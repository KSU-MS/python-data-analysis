"""Regen: off is exactly motoring-only, and when on it obeys its limits and the
energy chain (wheels -> diff -> chain -> motor map -> inverter -> cells)."""
import numpy as np
import pandas as pd
import pytest

from accel import Car
from accel.drive import drive
from accel.driver import RegenPolicy, ScriptedDriver
from accel.powertrain import PACKS, Pack, motor_efficiency_map

# drive 3 s, then off the throttle
DRIVE_THEN_LIFT = lambda: ScriptedDriver(pedal=lambda t: 0.6 if t < 3.0 else 0.0)


def test_off_is_bit_identical():
    a = drive(DRIVE_THEN_LIFT(), Car(), t_max=6.0).df
    b = drive(DRIVE_THEN_LIFT(), Car(), t_max=6.0, regen=RegenPolicy(lift_torque=0.0)).df
    pd.testing.assert_frame_equal(a, b, check_exact=True)


def test_lift_regen_charges_through_the_efficiency_chain():
    car, pack = Car(), Pack(PACKS["cosmx600v"])
    d = drive(DRIVE_THEN_LIFT(), car, pack=pack, t_max=6.0, soc0=0.6,
              regen=RegenPolicy(lift_torque=60.0), log_period=2e-4, driver_period=2e-4).df
    gen = d[(d.t > 3.3) & (d.motor_torque < -1)]
    assert len(gen) > 1000
    assert (gen.bus_i < 0).all() and gen.soc.is_monotonic_increasing
    # DC power = shaft power x motor map x inverter (shaft power negative)
    rpms, torques, eff = motor_efficiency_map("228HV")
    from scipy.interpolate import RegularGridInterpolator
    em = RegularGridInterpolator((rpms, torques), eff)(np.column_stack([gen.motor_rpm, -gen.motor_torque]))
    shaft = gen.motor_torque * gen.motor_rpm * 2 * np.pi / 60
    expected = shaft * em * car.regen_inverter_eff
    np.testing.assert_allclose(gen.bus_v * gen.bus_i, expected, rtol=2e-3)
    assert (-gen.bus_i <= pack.max_charge_current + 1e-9).all()


def test_no_regen_below_min_speed():
    car = Car()
    d = drive(ScriptedDriver(pedal=lambda t: 0.0), car, vx0=6.0, t_max=12.0,
              regen=RegenPolicy(lift_torque=60.0)).df
    slow = d[d.vx < car.regen_min_speed]
    assert len(slow) and (slow.motor_torque.abs() < 1e-9).all()
    assert d.vx.iloc[-1] < car.regen_min_speed            # it did slow down to there


def test_charge_limits():
    car = Car()
    hard = RegenPolicy(lift_torque=230.0)
    # current: a strong regen at mid charge is held at the pack's charge current
    pack = Pack(PACKS["cosmx600v"])
    d = drive(ScriptedDriver(pedal=lambda t: 0.0), car, pack=pack, vx0=25.0, soc0=0.5,
              t_max=1.0, regen=hard).df.iloc[30:]
    assert d.bus_i.min() == pytest.approx(-pack.max_charge_current, rel=1e-6)
    # voltage: a full pack with a 4.2 V cell ceiling can't take any charge
    full = Pack(PACKS["cosmx600v"], v_cell_max=4.2)
    d = drive(ScriptedDriver(pedal=lambda t: 0.0), car, pack=full, vx0=25.0, soc0=1.0,
              t_max=1.0, regen=hard).df
    assert d.bus_i.min() >= -1e-9 and (d.bus_v <= d.pack_v_oc + 1e-6).all()   # never pushed above OCV


def test_blend_rear_share_cap():
    car = Car()
    pol = RegenPolicy(brake_fraction=1.0, max_rear_brake_share=0.5)
    for brake in (0.1, 0.5, 1.0):
        regen_wheel = pol.blend_torque(brake, car) * car.final_drive / car.driveline_eff
        friction = brake * car.max_brake_torque
        rear = friction * (1 - car.brake_bias_front) + regen_wheel
        assert rear / (friction + regen_wheel) == pytest.approx(0.5, abs=1e-9)
