"""Double-track model (vehicle.py), drivers and the drive() harness."""
import dataclasses
import math

import numpy as np
import pytest

from kmslap import Car, simulate
from kmslap.models import twotrack as vehicle
from kmslap.driving.drive import drive
from kmslap.driving.paths import Path, SKIDPAD_RADIUS
from kmslap.driving.follower import PathFollower, ScriptedDriver
from kmslap.vcu.firmware import FirmwareController
from kmslap.car.powertrain import PACKS, MotorMap, Pack
from kmslap.events.accel import LENGTH, STAGE
from kmslap.car.tire import default_tire


@pytest.fixture(scope="module")
def motor():
    return MotorMap("228HV")


@pytest.fixture(scope="module")
def pack():
    return Pack(PACKS["cosmx600v"])


def _crossing_time(df, x):
    return float(np.interp(x, df.X, df.t))


def test_compiled_step_matches_casadi(motor, pack):
    car, tire = Car(), default_tire()
    f = vehicle.step_function(motor, tire, pack)
    p = vehicle.params(car, motor, pack, tire)
    rng = np.random.default_rng(1)
    for _ in range(30):
        s = vehicle.initial_state(car, pack, vx=rng.uniform(0, 25), soc=rng.uniform(0.2, 1),
                                  pack_temp=rng.uniform(10, 50))
        for k, lo, hi in [("vy", -1, 1), ("r", -1.5, 1.5), ("kappa_rl", -0.2, 1), ("kappa_rr", -0.2, 1),
                          ("alpha_fl", -0.2, 0.2), ("alpha_rr", -0.2, 0.2), ("ay_filt", -12, 12)]:
            s[vehicle.STATES.index(k)] = rng.uniform(lo, hi)
        u = [rng.uniform(0, 230), rng.uniform(-0.3, 0.3), rng.uniform(0, 800), rng.uniform(0, 1)]
        st = vehicle.car_step(motor, tire, pack, p, s)
        st(*u)
        nxt, out = f(s, u, p)
        np.testing.assert_allclose(st.state, np.array(nxt).ravel(), rtol=1e-11, atol=1e-11)
        np.testing.assert_allclose(st.out, np.array(out).ravel(), rtol=1e-11, atol=1e-11)


def test_straight_line_matches_accel_sim(motor, pack):
    # no steer: the four-wheel model with its diff and front wheel states reduces to
    # the 1D accel model
    car = Car()
    full = ScriptedDriver(pedal=lambda t: min(t / 0.1, 1.0))
    d = drive(full, car, motor, pack, driver_period=2e-4, log_period=2e-4, t_max=5.0).df
    t2d = _crossing_time(d, STAGE + LENGTH) - _crossing_time(d, STAGE)
    assert t2d == pytest.approx(simulate(car, motor, pack=pack).accel_time, rel=2e-3)
    assert d.Y.abs().max() == 0 and d.psi.abs().max() == 0   # stays exactly symmetric


def test_left_right_mirror(motor, pack):
    steer = lambda sign: ScriptedDriver(pedal=lambda t: 0.3,
                                        steering_wheel=lambda t: sign * math.radians(40) * min(t, 1))
    left = drive(steer(1), Car(), motor, pack, t_max=4.0).df
    right = drive(steer(-1), Car(), motor, pack, t_max=4.0).df
    assert left.Y.iloc[-1] > 1.0
    np.testing.assert_allclose(left.Y, -right.Y, atol=1e-9)
    np.testing.assert_allclose(left.r, -right.r, atol=1e-9)
    np.testing.assert_allclose(left.X, right.X, atol=1e-9)


def test_steady_circle(motor, pack):
    car, v = Car(), 9.0
    circle = Path.from_segments([("arc", SKIDPAD_RADIUS, 540.0)])
    d = drive(PathFollower(circle, speed=v, car=car), car, motor, pack, vx0=v, t_max=10).df
    ss = d.query("t > 6")
    radius = np.hypot(ss.X, ss.Y - SKIDPAD_RADIUS)
    assert ss.speed.mean() == pytest.approx(v, rel=0.01)
    assert radius.mean() == pytest.approx(SKIDPAD_RADIUS, rel=0.05)
    # steady state: lateral accel = v^2 / R and yaw rate = v / R at the driven radius
    assert ss.ay.mean() == pytest.approx((ss.speed ** 2 / radius).mean(), rel=0.02)
    assert ss.r.mean() == pytest.approx((ss.speed / radius).mean(), rel=0.02)
    # outside wheels carry more load, the car leans on them
    assert (ss.fz_fr > ss.fz_fl).all() and (ss.fz_rr > ss.fz_rl).all()


def test_pack_charge_and_heat(motor):
    # no cooling: the pack's temperature rise is its I^2 R heat over its thermal mass,
    # and its SOC drop is the charge drawn
    pack = Pack(PACKS["cosmx600v"], h_a=0.0, resistance_temp_coeff=0.0)
    full = ScriptedDriver(pedal=lambda t: min(t / 0.1, 1.0))
    d = drive(full, Car(), motor, pack, driver_period=2e-4, log_period=2e-4, t_max=4.0).df
    dt = 2e-4
    charge_ah = (d.bus_i * dt).sum() / 3600
    heat_j = (d.pack_heat * dt).sum()
    assert d.soc.iloc[0] - d.soc.iloc[-1] == pytest.approx(charge_ah / pack.capacity_ah, rel=1e-3)
    assert d.pack_temp.iloc[-1] - d.pack_temp.iloc[0] == pytest.approx(heat_j / pack.thermal_mass, rel=1e-3)
    assert d.bus_v.min() < d.pack_v_oc.min()      # sags under load


def test_pack_cools_to_ambient(motor):
    pack = Pack(PACKS["cosmx600v"], ambient_c=20.0)
    idle = ScriptedDriver(pedal=lambda t: 0.0)
    d = drive(idle, Car(), motor, pack, pack_temp0=45.0, t_max=120.0, dt=2e-3,
              driver_period=0.1, log_period=1.0).df
    tau = pack.thermal_mass / pack.h_a
    expected = 20.0 + 25.0 * math.exp(-d.t.iloc[-1] / tau)
    assert d.pack_temp.iloc[-1] == pytest.approx(expected, abs=0.05)


def test_limited_slip_diff_shares_torque(motor, pack):
    # power on in a tight corner: the unloaded inside (left) rear spins up. A locked-up
    # LSD holds the two rears together much more than an open diff.
    def run(**diff):
        car = dataclasses.replace(Car(), **diff)
        corner = ScriptedDriver(pedal=lambda t: 0.15 if t < 3 else 0.7,
                                steering_wheel=lambda t: math.radians(90))
        return drive(corner, car, motor, pack, vx0=8.0, t_max=3.6).df.query("t > 3.2")
    open_ = run(diff_preload=0.0, diff_lock_drive=0.0)
    lsd = run(diff_preload=50.0, diff_lock_drive=0.8)
    assert open_.rear_speed_diff.mean() > 2 * lsd.rear_speed_diff.mean() > 0
    assert (lsd.diff_torque.abs() > 0).any()


def test_controller_in_the_loop(motor, pack):
    # the VCU controller fed per-wheel sensor edges from the double-track model
    fc = FirmwareController(mode="mu_adaptive")
    full = ScriptedDriver(pedal=lambda t: min(t / 0.1, 1.0))
    res = drive(full, Car(), motor, pack, controller=fc, t_max=5.0)
    d = res.df
    t2d = _crossing_time(d, STAGE + LENGTH) - _crossing_time(d, STAGE)
    t1d = simulate(Car(), motor, pack=pack, controller=FirmwareController(mode="mu_adaptive")).accel_time
    assert t2d == pytest.approx(t1d, rel=0.01)
    assert len(res.controller_log) > 4000
