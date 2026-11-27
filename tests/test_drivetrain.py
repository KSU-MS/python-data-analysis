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


def test_cli_lists_match_the_code():
    from kmslap import cli
    from kmslap.vcu.torque_control import MODES
    assert [d.value for d in cli.Drivetrain] == list(VARIANTS)
    assert [m.value for m in cli.TorqueMode] == list(MODES)


@pytest.mark.parametrize("mode", ["qp", "qp_lc"])
def test_qp_vectors_and_cuts_a_spinning_wheel(mode):
    """The QP puts more torque outside when the car should turn more, and takes it off a
    wheel spinning past its peak slip; with or without load cells."""
    from kmslap.car.powertrain import MotorMap
    from kmslap.vcu.torque_control import TorqueController, VehicleState
    car = variant("4wd_dti")
    fz = (700.0,) * 4
    v = 15.0
    om = (v / car.tire_radius,) * 4
    ctl = TorqueController(car, MotorMap("228HV"), mode)
    _, (fl, fr, rl, rr) = ctl(VehicleState(0.5, 0.0, 0.1, om, v, 0.0, 0.0, 580.0, fz=fz))
    assert fr > fl and rr > rl
    ctl = TorqueController(car, MotorMap("228HV"), mode)
    even = ctl(VehicleState(0.3, 0.0, 0.0, om, v, 0.0, 0.0, 580.0, fz=fz))[1]
    spin = ctl(VehicleState(0.3, 0.0, 0.0, (om[0], om[1], om[2] * 1.3, om[3]), v, 0.0, 0.0, 580.0, fz=fz))[1]
    assert even[2] > 5 and spin[2] < 0.5 * even[2]


def test_qp_holds_the_power_limit_and_drives():
    car = variant("4wd_dti")
    d = drive(ScriptedDriver(), car, t_max=3.0, log_period=0.005, torque_mode="qp").df
    assert d.speed.iloc[-1] > 20
    assert (d.bus_v * d.bus_i).max() <= 80.5e3


def test_wheel_load_estimate_without_load_cells():
    """The IMU / speed estimate sums to the weight plus downforce and moves load the
    right ways."""
    from kmslap.car.params import G
    from kmslap.car.powertrain import MotorMap
    from kmslap.vcu.torque_control import TorqueController, VehicleState
    car = variant("4wd_dti")
    ctl = TorqueController(car, MotorMap("228HV"), "qp")
    fz = ctl.estimate_fz(VehicleState(0, 0, 0, (0,) * 4, 20.0, 0.5, 5.0, 580.0, ay=10.0))
    assert fz.sum() == pytest.approx(car.mass * G + 0.5 * car.rho * 400 * car.cla)
    assert fz[2] + fz[3] > fz[0] + fz[1] - 1e-9 or car.front_weight > 0.5     # accelerating: rearward
    assert fz[1] > fz[0] and fz[3] > fz[2]                                   # turning left: onto the right


def test_dti_current_mode_steps_and_times_out():
    from kmslap.vcu.inverter import DtiCurrentMode
    inv = DtiCurrentMode(timeout=0.5)
    assert inv(0.0, 1e-3, 1000) == 0.0                     # nothing sent yet
    inv.command(0.0, 20.0)
    assert inv(0.1, 1e-3, 1000) == pytest.approx(20.0, abs=0.1 * inv.kt)
    inv.command(0.2, 1e3)
    assert inv(0.2, 1e-3, 1000) == pytest.approx(inv.i_max * inv.kt, abs=0.1 * inv.kt)   # SetMaxAcCurrent
    assert inv(0.8, 1e-3, 1000) == 0.0                     # no command for 0.6 s: free running


def test_amk_speed_mode_gives_torque_by_saturating():
    """Speed setpoint far above the motor with the torque limit: the limit comes out;
    regen likewise, through the negative limit; the slip encoder holds the speed."""
    from kmslap.vcu.inverter import AmkSpeedMode
    inv = AmkSpeedMode(wheel_rpm_per_ms=500.0)
    inv.command(0.0, 15.0, 5000.0)
    assert inv(0.001, 1e-3, 5000.0) == pytest.approx(15.0, abs=inv.unit)
    inv.command(0.01, -8.0, 5000.0)
    assert inv(0.011, 1e-3, 5000.0) == pytest.approx(-8.0, abs=inv.unit)
    slip = AmkSpeedMode(encoder="slip", slip_target=0.1)
    slip.command(0.0, 15.0, 5000.0)
    assert slip(0.001, 1e-3, 5500.0) == pytest.approx(0.0, abs=0.05)    # at the setpoint: no torque
    assert slip(0.002, 1e-3, 6000.0) == 0.0                               # past it: cut (no negative limit)


def test_drivetrain_drag_race_runs():
    from kmslap.reporting.variants import accel_run
    r = accel_run("4wd_dti", "qp")
    assert 3.0 < r.accel_time < 4.5
    assert {"x", "v", "a", "slip", "motor_torque"} <= set(r.df.columns)


def test_qp_keeps_power_oversteer_in_check():
    """A power-on sine steer at 20 m/s: the load split spins the car; the QP's yaw
    control (and, on the AWD car, its EMRAX cut) keeps the body slip small."""
    from kmslap.vcu.torque_bench import sine
    for name in ("4wd_dti", "awd_emrax_dti"):
        assert sine(name, "load", {})["sine_beta_deg"] > 30
        assert sine(name, "qp", {})["sine_beta_deg"] < 10


def test_qp_launches_with_little_slip():
    from kmslap.vcu.torque_bench import accel
    r = accel("4wd_dti", "qp", {})
    assert r["accel_slip"] < 0.3
    assert r["accel_s"] < accel("4wd_dti", "load", {})["accel_s"]


def test_lap_plan_drives_the_hub_motors():
    from kmslap.car.powertrain import MotorMap
    from kmslap.car.tire import default_tire
    from kmslap.models import mintime_dt
    car = variant("4wd_dti")
    assert mintime_dt.controls(car)[-4:] == ["hub_fl", "hub_fr", "hub_rl", "hub_rr"]
    assert mintime_dt.controls(Car()) == mintime_dt.CONTROLS
    model = mintime_dt._model(car, MotorMap("228HV"), default_tire(), 550.0, 1.0)
    x = [15.0, 0.0, 0.0, 0.0, 0.0]
    u = lambda hub: [0.0] * 9 + [hub, 0.0, 0.0, 0.0]          # straight, no slip: no tire force
    _, _, eq0, ineq0, _ = (np.asarray(a).ravel() for a in model(x, u(0.0), 0.0))
    _, _, eq1, ineq1, _ = (np.asarray(a).ravel() for a in model(x, u(20.0), 0.0))
    # front left's torque balance is off by the hub's wheel torque; the others aren't
    assert eq1[2] - eq0[2] == pytest.approx(-20.0 * car.hub_ratio * car.hub_gear_eff)
    assert np.allclose(np.delete(eq1 - eq0, 2), 0.0)
    # its power comes out of the 80 kW: more than the shaft power (losses)
    shaft = 20.0 * 15.0 / car.tire_radius * car.hub_ratio
    assert ineq0[1] - ineq1[1] > shaft * car.motor_eff
    assert ineq1[2] == pytest.approx(ineq0[2] - 20.0, abs=1e-6)  # fl's torque curve headroom
