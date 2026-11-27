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
