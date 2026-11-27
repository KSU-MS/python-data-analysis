"""A benchmark for the multi-motor torque controllers (vcu/torque_control.py): the same
three tests for every car and mode, to tune against.

    kmslap torque-bench                                  # the default cars x modes
    kmslap torque-bench -c 4wd_dti -m qp -p yaw_kp=8000  # one setting

  accel    75 m, full pedal (0.1 s ramp), wing open: time and peak driven-wheel slip
  skidpad  the fastest speed the path follower holds on the skidpad circle (radius
           and speed within 0.3 m / 2 %), as steady lateral g
  sine     a 1 Hz steering wheel sine (3 periods) at 20 m/s with the pedal held at
           0.35: the car speeds up through it and power oversteer shows; peak body
           slip angle (stability) and the speed at the end (drive kept)

inverters: None (torque straight to the motors), "native" (vcu/inverter.py: DTI current
mode, AMK speed mode saturated) or "slip" (AMK with the setpoint at the ground speed
plus the slip target: the inverter's speed loop as traction control).
"""
import itertools
import math
from multiprocessing import Pool

import numpy as np

from kmslap.car.drivetrain import variant
from kmslap.car.params import Car

CARS = ("4wd_dti", "4wd_amk", "awd_emrax_dti", "awd_emrax_amk")
MODES = ("load", "yaw", "qp")
SKID_SPEEDS = np.arange(11.0, 13.76, 0.25)


def _controller(car, mode, params):
    from kmslap.car.powertrain import MotorMap
    from kmslap.vcu.torque_control import TorqueController
    return TorqueController(car, MotorMap("228HV"), mode, **params)


def _inverters(car, inverters):
    if inverters is None or not (car.hub_front or car.hub_rear):
        return None
    from kmslap.vcu.inverter import Inverters
    if inverters == "slip":
        return Inverters(car, encoder="slip") if car.hub_motor == "amk" else Inverters(car)
    return Inverters(car)


def accel(name, mode, params, inverters=None):
    from kmslap.driving.drive import drive
    from kmslap.driving.follower import ScriptedDriver
    from kmslap.reporting.variants import as_accel_result
    car = variant(name, Car().open_aero())
    d = drive(ScriptedDriver(pedal=lambda t: min(t / 0.1, 1.0)), car, t_max=5.0, log_period=1e-3,
              torque_controller=_controller(car, mode, params), inverters=_inverters(car, inverters)).df
    r = as_accel_result(d, car)
    return {"accel_s": r.accel_time, "accel_slip": float(r.df.slip.max())}


def skid_at(name, mode, params, v, inverters=None):
    """(held, lateral g) at this speed on the skidpad circle."""
    from kmslap.driving.drive import drive
    from kmslap.driving.follower import PathFollower
    from kmslap.driving.paths import SKIDPAD_RADIUS, Path
    car = variant(name)
    circle = Path.from_segments([("arc", SKIDPAD_RADIUS, 900.0)])
    d = drive(PathFollower(circle, speed=v, car=car), car, vx0=v, t_max=10,
              torque_controller=_controller(car, mode, params), log_period=0.02,
              inverters=_inverters(car, inverters)).df
    ss = d.query("t > 6")
    rad = np.hypot(ss.X, ss.Y - SKIDPAD_RADIUS)
    held = abs(rad.mean() - SKIDPAD_RADIUS) < 0.3 and rad.std() < 0.3 and abs(ss.speed.mean() / v - 1) < 0.02
    return held, float(ss.ay.mean() / 9.81)


def sine(name, mode, params, amp_deg: float = 60.0, inverters=None):
    from kmslap.driving.drive import drive
    from kmslap.driving.follower import ScriptedDriver
    car = variant(name)
    sw = lambda t: math.radians(amp_deg) * math.sin(2 * math.pi * (t - 1.0)) if 1.0 < t < 4.0 else 0.0
    d = drive(ScriptedDriver(pedal=lambda t: 0.35, steering_wheel=sw), car, vx0=20.0, t_max=5.0,
              torque_controller=_controller(car, mode, params), log_period=0.005,
              inverters=_inverters(car, inverters)).df
    w = d.query("1.0 < t < 4.5")
    return {"sine_beta_deg": float(np.degrees(np.abs(w.beta).max())), "sine_v_end": float(w.vx.iloc[-1])}


def _job(job):
    kind, name, mode, params, arg, inv = job
    if kind == "accel":
        return accel(name, mode, params, inverters=inv)
    if kind == "sine":
        return sine(name, mode, params, inverters=inv)
    return skid_at(name, mode, params, arg, inverters=inv)


def run(cars=CARS, modes=MODES, params: dict | None = None, workers: int | None = None, inverters=None) -> list:
    """Every car x mode (params for each): a list of result dicts, printed as a table."""
    params = params or {}
    cases = list(itertools.product(cars, modes))
    jobs = ([("accel", n, m, params, None, inverters) for n, m in cases]
            + [("sine", n, m, params, None, inverters) for n, m in cases]
            + [("skid", n, m, params, v, inverters) for n, m in cases for v in SKID_SPEEDS])
    from kmslap import progress
    with Pool(workers) as p, progress.task(f"torque bench: {len(cars)} cars x {len(modes)} modes, "
                                           "accel / skidpad / sine steer, double-track model",
                                           total=len(jobs)) as task:
        out = []
        for r in p.imap(_job, jobs):
            out.append(r)
            task.advance()
    k = len(cases)
    rows = []
    for i, (n, m) in enumerate(cases):
        row = {"car": n, "mode": m, **out[i], **out[k + i]}
        skid = out[2 * k + i * len(SKID_SPEEDS): 2 * k + (i + 1) * len(SKID_SPEEDS)]
        row["skidpad_g"] = max((g for held, g in skid if held), default=float("nan"))
        rows.append(row)
        print(f"{n:14s} {m:6s} accel {row['accel_s']:.3f} s (slip {row['accel_slip']:.2f})  "
              f"skidpad {row['skidpad_g']:.3f} g  sine: beta {row['sine_beta_deg']:4.1f} deg, "
              f"v_end {row['sine_v_end']:.1f} m/s", flush=True)
    return rows
