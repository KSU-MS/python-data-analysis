"""The C++ VCU controller (controller/), run through accel.firmware."""
import ctypes
import math

import numpy as np
import pandas as pd
import pytest

from accel import Car, MotorMap, quasi_static, simulate
from accel.firmware import (FirmwareController, Inputs, Outputs, Sensors, load_library,
                            replay)


@pytest.fixture(scope="module")
def motor():
    return MotorMap("228HV")


def _controller(mode="passthrough", **overrides):
    fc = FirmwareController(mode=mode, overrides=overrides)
    lib = load_library()
    state = ctypes.create_string_buffer(lib.vcu_state_size())
    cfg = fc.config(MotorMap("228HV"))
    lib.vcu_init(state, ctypes.byref(cfg), 0)
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
    again = replay(fc.input_frame(), fc.config(motor))
    pd.testing.assert_frame_equal(again, r.controller_log, check_exact=True)
