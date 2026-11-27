"""Drivetrain variants: masses, and hub motor cars driving within the rules."""
import numpy as np
import pytest

from kmslap.car.drivetrain import VARIANTS, HubMotor, mass_breakdown, variant
from kmslap.car.params import Car
from kmslap.driving.drive import drive
from kmslap.driving.follower import ScriptedDriver


def test_the_current_car_is_the_rwd_variant():
    assert variant("rwd_emrax") == Car()


def test_variant_masses_follow_their_parts():
    awd, fwd = mass_breakdown("awd_emrax_dti"), mass_breakdown("4wd_dti")
    assert not awd["removed"] and awd["mass"] > Car().mass + 10          # hubs added, nothing out
    assert any("EMRAX" in p.name for p in fwd["removed"])                 # no EMRAX, chain or diff
    assert fwd["front_weight"] > Car().front_weight                       # weight moves forward
    for name in VARIANTS:
        assert 270 < variant(name).mass < 320


def test_hub_motor_curves():
    dti = HubMotor("dti")
    assert dti.max_torque(5000) == pytest.approx(31.6)
    assert dti.max_torque(20000) == 0.0
    assert 0.9 < dti.efficiency(10000, 20) < 0.97


@pytest.mark.parametrize("name", ["awd_emrax_dti", "4wd_dti"])
def test_hub_cars_drive_within_80_kw(name):
    car = variant(name)
    d = drive(ScriptedDriver(), car, t_max=3.0, log_period=0.005).df
    assert d.speed.iloc[-1] > 20                                          # it went
    assert (d.bus_v * d.bus_i).max() <= 80.5e3                            # the rule
    front = d[["hub_torque_fl", "hub_torque_fr"]].to_numpy()
    assert front.max() > 5                                                # the front wheels drive


def test_torque_vectoring_turns_the_car():
    """In yaw mode, asking for more yaw puts more torque on the outside wheels."""
    from kmslap.car.powertrain import MotorMap
    from kmslap.vcu.torque_control import TorqueController, VehicleState
    car = variant("4wd_dti")
    ctl = TorqueController(car, MotorMap("228HV"), "yaw", yaw_kp=3000, yaw_ki=0)
    om = (15 / car.tire_radius,) * 4
    _, (fl, fr, rl, rr) = ctl(VehicleState(0.5, 0.0, 0.1, om, 15.0, 0.0, 0.0, 580.0))   # steered left, not yawing
    assert fr > fl and rr > rl
    _, (fl, fr, rl, rr) = TorqueController(car, MotorMap("228HV"), "load")(VehicleState(0.5, 0.0, 0.1, om, 15.0, 0.0, 0.0, 580.0))
    assert fr == pytest.approx(fl) and rr == pytest.approx(rl)


def test_mode_manager_switches_only_when_slow():
    from kmslap.car.powertrain import MotorMap
    from kmslap.vcu.torque_control import ModeManager, TorqueController, VehicleState
    car, m = variant("4wd_dti"), MotorMap("228HV")
    mm = ModeManager({k: TorqueController(car, m, k) for k in ("load", "yaw")}, "load")
    mm.request("yaw")
    mm(VehicleState(0.5, 0.0, 0.0, (20 / car.tire_radius,) * 4, 20.0, 0.0, 0.0, 580.0))
    assert mm.active == "load"                                             # 20 m/s: not now
    mm(VehicleState(0.0, 0.0, 0.0, (2 / car.tire_radius,) * 4, 2.0, 0.0, 0.0, 580.0))
    assert mm.active == "yaw"
