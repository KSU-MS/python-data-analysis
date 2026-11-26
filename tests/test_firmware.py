"""The C++ VCU controller (controller/), run through accel.firmware."""
import ctypes
import dataclasses
import math

import numpy as np
import pandas as pd
import pytest

from accel import Car, MotorMap, PacejkaTire, quasi_static, simulate
from accel.firmware import (FirmwareController, Inputs, Outputs, Sensors, load_library,
                            read_log, replay, warmup, write_log)
from accel.sysid import fit_plant
from accel.uncertainty import RANGES


@pytest.fixture(scope="module")
def motor():
    return MotorMap("228HV")


def _controller(mode="passthrough", **overrides):
    fc = FirmwareController(mode=mode, overrides=overrides)
    lib = load_library()
    state = ctypes.create_string_buffer(lib.vcu_state_size())
    cfg = fc.config(MotorMap("228HV"))
    lib.vcu_init(state, ctypes.byref(cfg), None, 0)
    return lib, state


def test_wheel_speed_estimator_tracks_and_decays():
    lib, state = _controller()
    teeth, omega = 19, 20.0                     # rad/s
    period_us = 2 * math.pi / teeth / omega * 1e6
    inp, out = Inputs(), Outputs()
    edge = 0
    for ms in range(1, 301):
        now = ms * 1000
        n = 0
        while (edge + 1) * period_us <= now:
            edge += 1
            n += 1
        inp.now_us = now
        inp.ws_edges[0] = n
        if n:
            inp.ws_last_edge_us[0] = int(edge * period_us)
        lib.vcu_step(state, ctypes.byref(inp), ctypes.byref(out))
    assert out.wheel_omega[0] == pytest.approx(omega, rel=0.02)

    # wheel stops: estimate decays (bounded by one tooth per time since last edge)
    for ms in range(301, 1000):
        inp.now_us = ms * 1000
        inp.ws_edges[0] = 0
        lib.vcu_step(state, ctypes.byref(inp), ctypes.byref(out))
        if ms == 400:
            assert out.wheel_omega[0] < 5.0
    assert out.wheel_omega[0] == 0.0


def test_passthrough_matches_open_loop(motor):
    car = Car()
    plain = simulate(car, motor)
    vcu = simulate(car, motor, controller=FirmwareController(mode="passthrough"))
    assert vcu.accel_time == pytest.approx(plain.accel_time, rel=0.01)


@pytest.mark.parametrize("mode", ["slip_pi", "mu_adaptive"])
def test_closed_loop_with_real_sensors(motor, mode):
    car = Car()
    ideal = quasi_static(car, motor).accel_time
    r = simulate(car, motor, controller=FirmwareController(mode=mode))
    assert r.accel_time == pytest.approx(ideal, rel=0.015)
    assert r.df.query("t > 0.5")["slip"].max() < 0.25
    log = r.controller_log
    assert np.abs(log["v_est"].to_numpy() - np.interp(log["t"], r.df["t"], r.df["v"])).max() < 0.3


def test_replay_reproduces_outputs_exactly(motor):
    fc = FirmwareController(mode="mu_adaptive", sensors=Sensors(seed=3))
    r = simulate(Car(), motor, controller=fc)
    again = replay(fc.input_frame(), fc.config(motor), t_offset=fc.sensors.staged_time)
    pd.testing.assert_frame_equal(again, r.controller_log, check_exact=True)


def test_binary_log_round_trip(motor, tmp_path):
    fc = FirmwareController(mode="slip_pi")
    r = simulate(Car(), motor, controller=fc)
    write_log(tmp_path / "run.vcu", fc.records())
    inputs, logged = read_log(tmp_path / "run.vcu")
    pd.testing.assert_frame_equal(inputs, fc.input_frame(), check_dtype=False, check_exact=True)
    # the log alone is enough to reproduce what the controller did
    again = replay(inputs, fc.config(motor), t_offset=fc.sensors.staged_time)
    np.testing.assert_array_equal(again["torque_cmd"].to_numpy(np.float32),
                                  logged["torque_cmd_nm"].to_numpy())
    np.testing.assert_array_equal(again["torque_cmd"], r.controller_log["torque_cmd"])


def test_warmup_calibrates_radius_and_bias(motor):
    true = dataclasses.replace(Car(), tire_radius=Car().tire_radius * 0.98)
    learned = warmup(true, FirmwareController(sensors=Sensors(vn_accel_bias=0.15)), motor)
    assert learned["r_front"] == pytest.approx(true.tire_radius, rel=0.003)
    assert learned["r_rear"] == pytest.approx(true.tire_radius, rel=0.003)
    assert learned["accel_bias"] == pytest.approx(0.15, abs=0.03)


def test_mu_learning_carries_between_runs(motor):
    car, tire = Car(), PacejkaTire(scale=0.6 * 0.85)
    ideal = quasi_static(car, motor, tire=tire).accel_time
    first = FirmwareController(mode="mu_adaptive")
    simulate(car, motor, tire=tire, controller=first)
    assert first.learned["mu"] < 1.2                       # learned the low grip
    second = FirmwareController(mode="mu_adaptive", learned=first.learned)
    r = simulate(car, motor, tire=tire, controller=second)
    assert r.accel_time == pytest.approx(ideal, rel=0.015)


def test_sysid_recovers_plant_from_log(motor):
    nominal = Car()
    true = dataclasses.replace(nominal, mass=nominal.mass * 1.04,
                               tire_radius=nominal.tire_radius * 0.985)
    fc = FirmwareController(mode="slip_pi", sensors=Sensors(vn_accel_bias=0.1))
    simulate(true, motor, tire=PacejkaTire(scale=0.6 * 0.9), relaxation_length=0.2,
             controller=fc)
    fit = fit_plant(fc.input_frame(), nominal, motor)
    truth = {"grip": 0.9, "mass": 1.04, "tire_radius": 0.985, "relaxation_length": 0.2}
    for k, v in truth.items():
        assert fit.params[k] == pytest.approx(v, abs=2 * fit.std[k]), k
        lo, hi = fit.ranges(RANGES)[k]
        assert lo <= v <= hi, k
    assert fit.ranges(RANGES)["grip"][1] - fit.ranges(RANGES)["grip"][0] < 0.5 * (
        RANGES["grip"][1] - RANGES["grip"][0])
