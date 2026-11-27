"""The CasADi plant model: compiled step, rollout gradients, cache reuse."""
import dataclasses

import casadi as ca
import numpy as np
import pytest

from accel import Car, CurveTire, MotorMap, simulate
from accel import model
from accel.powertrain import PACKS, Pack
from accel.tire import PacejkaTire, default_tire


@pytest.fixture(scope="module")
def motor():
    return MotorMap("228HV")


@pytest.fixture(scope="module")
def pack():
    return Pack(PACKS["cosmx600v"])


@pytest.mark.parametrize("tire", [default_tire(), CurveTire()], ids=["pacejka", "curve"])
def test_compiled_step_matches_casadi(motor, pack, tire):
    # the generated C and CasADi's own evaluation of the step agree, across states
    # that hit the branches (stopped, spinning, braking slip, power limited)
    f = model.step_function(motor, tire)
    p = model.params(Car(), motor, pack, tire, 0.15, 2e-4)
    rng = np.random.default_rng(0)
    for _ in range(50):
        s = np.array([rng.uniform(0, 75), rng.uniform(0, 30), rng.uniform(0, 150),
                      rng.uniform(-0.3, 1.0), rng.uniform(-5, 15), rng.uniform(0, 230),
                      rng.uniform(450, 600)])
        s[rng.integers(0, 3)] = 0.0
        cmd = rng.uniform(0, 230)
        st = model.accel_step(motor, tire, p, s)
        st(cmd)
        nxt, out = f(s, cmd, p)
        np.testing.assert_allclose(st.state, np.array(nxt).ravel(), rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(st.out, np.array(out).ravel(), rtol=1e-12, atol=1e-12)


def test_rollout_matches_simulate(motor, pack):
    tire = default_tire()
    res = simulate(Car(), motor, pack=pack, tire=tire, t_max=1.0)
    n = len(res.df)
    cmd = np.minimum(motor.peak_torque * np.arange(n) * 2e-4 / 0.1, motor.peak_torque)
    states, _ = model.rollout(motor, tire, n)(model.initial_state(pack), cmd,
                                              model.params(Car(), motor, pack, tire, 0.15, 2e-4))
    v = np.array(states[model.STATES.index("v"), :]).ravel()
    np.testing.assert_allclose(v[:-1], res.df.v.to_numpy()[1:], rtol=1e-12, atol=1e-12)


def test_accel_time_gradient_matches_finite_difference(motor, pack):
    car, tire = Car(), default_tire()
    t0, grad = model.accel_time_sensitivity(car, motor, pack, tire)
    assert t0 == pytest.approx(simulate(car, motor, pack=pack, tire=tire).accel_time, abs=1e-9)
    for name, tweak in [("mass", lambda c, h: (dataclasses.replace(c, mass=c.mass + h), tire)),
                        ("tire_scale", lambda c, h: (c, PacejkaTire(scale=tire.scale + h)))]:
        h = 1e-3 * (car.mass if name == "mass" else tire.scale)
        up = simulate(*tweak(car, h)[:1], motor, pack=pack, tire=tweak(car, h)[1]).accel_time
        dn = simulate(*tweak(car, -h)[:1], motor, pack=pack, tire=tweak(car, -h)[1]).accel_time
        assert grad[name] == pytest.approx((up - dn) / (2 * h), rel=0.02)


def test_grip_is_a_parameter_not_a_rebuild(motor):
    # Monte Carlo plants differ in tire scale; they must share one compiled model
    keys = {PacejkaTire(scale=s).table_key() for s in (0.5, 0.6, 0.7)}
    assert len(keys) == 1
    assert CurveTire(1.2).table_key() == CurveTire(1.6).table_key()
