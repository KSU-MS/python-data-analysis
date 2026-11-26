import math

import pytest

from accel import Car, CurveTire, MotorMap, Pack, PACKS, PacejkaTire, SlipTC, quasi_static, simulate
from accel.params import G


@pytest.fixture(scope="module")
def motor():
    return MotorMap("228HV")


def test_static_loads_sum_to_weight():
    car = Car()
    fz_r = car.rear_load(0.0, 0.0)
    assert fz_r == pytest.approx(car.mass * G * (1 - car.front_weight))
    # load transfer moves weight rearward without changing the total
    assert car.rear_load(5.0, 0.0) > fz_r
    assert car.total_load(0.0) == pytest.approx(car.mass * G)


def test_pack_solve():
    pack = Pack(PACKS["energus300v"])
    assert pack.solve(0.0) == (pack.v_oc, 0.0, 0.0)
    v, i, p = pack.solve(30e3)
    assert p == pytest.approx(30e3)
    assert v == pytest.approx(pack.v_oc - i * pack.resistance)
    # asking for more than the pack can give is clipped at its current limit
    v, i, p = pack.solve(1e6)
    assert i == pytest.approx(pack.max_current)
    assert p < 1e6


def test_motor_map(motor):
    assert motor.max_torque(0, 300) == pytest.approx(motor.peak_torque, rel=1e-3)
    # field weakening: torque falls with speed, and more voltage helps
    assert motor.max_torque(5000, 300) < motor.max_torque(3000, 300)
    assert motor.max_torque(4000, 300) >= motor.max_torque(4000, 220)
    assert motor.max_torque(motor.max_rpm + 1, 300) == 0.0


@pytest.mark.parametrize("tire", [CurveTire(), PacejkaTire()], ids=["curve", "pacejka"])
def test_tire_shape(tire):
    k_pk, mu_pk = tire.peak(700)
    assert 0.05 < k_pk < 0.3
    assert 1.0 < mu_pk < 2.0
    assert tire.fx(0.0, 700) == pytest.approx(0.0, abs=50)
    assert tire.fx(k_pk, 700) == pytest.approx(mu_pk * 700, rel=0.02)
    assert tire.fx(0.5 * k_pk, 700) < tire.fx(k_pk, 700)


def test_motor_limited_matches_quasi_static(motor):
    """With huge grip, wheel slip dynamics shouldn't change the result."""
    car, tire = Car(final_drive=2.4), CurveTire(mu_peak=3.0)
    dyn = simulate(car, motor, tire=tire)
    qs = quasi_static(car, motor, tire=tire)
    assert dyn.accel_time == pytest.approx(qs.accel_time, rel=0.01)


def test_no_tc_wheelspin_and_tc_recovers(motor):
    car = Car(final_drive=3.5)
    ideal = quasi_static(car, motor).accel_time
    no_tc = simulate(car, motor)
    tc = simulate(car, motor, tc=SlipTC())
    assert no_tc.df["slip"].max() > 2          # spins up
    assert no_tc.accel_time > ideal * 1.1
    assert tc.accel_time < no_tc.accel_time
    assert tc.accel_time == pytest.approx(ideal, rel=0.025)

    # after launch: no overshoot, slip tracks target while TC is cutting torque
    # (later the car is motor-limited and slip falls below target on its own)
    target = SlipTC().target_slip
    after = tc.df[tc.df["t"] > 0.5]
    assert after["slip"].max() < target + 0.03
    # (< 0.95: ignore the slow integral unwind as the car goes motor-limited)
    cutting = after[after["tc_scale"] < 0.95]
    assert len(cutting) > 1000
    assert cutting["slip"].sub(target).abs().max() < 0.03
    assert after["tc_scale"].diff().abs().max() < 0.05


def test_accel_time_is_plausible(motor):
    r = simulate(Car(), motor, tc=SlipTC())
    assert 3.5 < r.accel_time < 5.5
    assert not math.isnan(r.accel_time)
