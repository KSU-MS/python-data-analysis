"""The drivetrain variants (car/drivetrain.py) side by side in the multi-car views: the
drag race (report drag GIF, `kmslap viz --drivetrains`), saved laps (`kmslap laptime
--drivetrain`, then `kmslap race`) and the endurance bird's-eye (`kmslap birdseye
--drivetrains`). Each runs in the double-track model (models/twotrack) with its own
torque controller (vcu/torque_control); the current car with the VCU's traction control.

    runs = accel_runs()              # name -> events.accel.Result, as simulate() gives
"""
import numpy as np

from kmslap.car.drivetrain import variant
from kmslap.car.params import Car
from kmslap.events.accel import Result

# (drivetrain, torque mode): the lineup. The current car's mode is the VCU firmware.
LINEUP = [("rwd_emrax", "vcu"), ("awd_emrax_dti", "qp"), ("awd_emrax_amk", "qp"),
          ("4wd_dti", "qp"), ("4wd_amk", "qp")]


def label(drivetrain: str, mode: str, inverters=None) -> str:
    return f"{drivetrain} {mode}" + (" +inv" if inverters else "")


def as_accel_result(df, car: Car, name: str = "", controller_log=None) -> Result:
    """A double-track straight-line run in the 1D accel sim's columns (x, v, a, slip,
    motor_torque, wheel_speed, fx, fz_rear, bus_v), for the drag race views. slip: the
    driven wheel slipping most; motor_torque: the EMRAX's, or the hub motors' mean."""
    r = car.tire_radius
    v = df.vx.to_numpy()
    driven = [w for w, on in (("fl", car.hub_front), ("fr", car.hub_front),
                              ("rl", car.hub_rear or car.central_drive),
                              ("rr", car.hub_rear or car.central_drive)) if on]
    slips = np.column_stack([(df[f"omega_{w}"].to_numpy() * r - v) / np.maximum(np.abs(v), 1.0) for w in driven])
    hubs = [f"hub_torque_{w}" for w in ("fl", "fr", "rl", "rr") if f"hub_torque_{w}" in df]
    torque = df.motor_torque.to_numpy() if car.central_drive else df[hubs].to_numpy().mean(axis=1)
    out = df[["t", "bus_v"]].copy()
    out["x"], out["v"], out["a"] = df.X.to_numpy(), v, df.ax.to_numpy()
    out["slip"] = slips.max(axis=1)
    out["motor_torque"] = torque
    out["wheel_speed"] = 0.5 * (df.omega_rl + df.omega_rr).to_numpy() * r
    if "fx_rl" in df:
        out["fx"] = (df.fx_rl + df.fx_rr).to_numpy()
    out["fz_rear"] = (df.fz_rl + df.fz_rr).to_numpy()
    return Result(out, name, controller_log)


def accel_run(drivetrain: str, mode: str = "qp", inverters=None, base: Car = None) -> Result:
    """75 m accel, full pedal (0.1 s ramp), wing open, in the double-track model."""
    from kmslap.driving.drive import drive
    from kmslap.driving.follower import ScriptedDriver
    from kmslap.vcu.firmware import FirmwareController

    car = variant(drivetrain, base or Car().open_aero())
    controller = FirmwareController(mode="mu_adaptive", belief=car) if mode == "vcu" else None
    res = drive(ScriptedDriver(pedal=lambda t: min(t / 0.1, 1.0)), car, controller=controller,
                t_max=5.0, log_period=1e-3, torque_mode="load" if mode == "vcu" else mode,
                inverters=inverters)
    return as_accel_result(res.df, car, label(drivetrain, mode, inverters), res.controller_log)


def accel_runs(lineup=None, inverters=None) -> dict:
    return {label(d, m, inverters): accel_run(d, m, inverters) for d, m in (lineup or LINEUP)}
