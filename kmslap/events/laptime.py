"""Lap time on a track: the double-track minimum-time plan (models/mintime_dt), driven by
the PathFollower with the plan's feedforward in the time-domain car, scored for FSAE
penalties, the flying lap saved (reporting/runs) to race against other versions.

    kmslap laptime ac18 --grip 0.85 --margin 0.35
    kmslap laptime ac18 --drivetrain 4wd_dti --torque-mode qp
    kmslap race ac18

A drivetrain variant (car/drivetrain.py) plans with its own mass, weight split and CG,
and its hub motors (each one's torque: torque vectoring). Regen: a brake pedal whose
first travel is regen only (FSAE T.3.2.4), planned and driven (on by default with hub
motors, where some motors can drive while others generate; an option for the
rear-drive car, whose regen brakes the rear axle alone).

    kmslap laptime michigan2026 --drivetrain 4wd_dti --soc 0.8
    kmslap laptime michigan2026 --regen-travel 0.9          # the rear-drive car with regen
"""
import dataclasses
import hashlib
import math
import pickle

import numpy as np

from kmslap import dirs
from kmslap.car.params import Car
from kmslap.events import track as T

CACHE = dirs.BUILD / "mintime"


def plan(track: T.Track, grip: float = 0.85, margin: float = T.EDGE_MARGIN, step: float = 2.0,
         car: Car = None, verbose: bool = False, regen: bool = False, soc: float = 1.0, label: str = ""):
    """mintime_dt.solve, cached in build/mintime/ by the track, car, settings and the
    solvers' sources (a solve takes ~5-7 minutes)."""
    from kmslap.models import envelope, mintime, mintime_dt

    car = car or Car()
    key = (dataclasses.astuple(car), grip, margin, step) + ((regen, soc) if regen or soc != 1.0 else ())
    h = hashlib.sha1(repr(key).encode())
    for a in (track.center.x, track.center.y, track.w_left, track.w_right):
        # rounded: a track built by spline fits differs in the last bits between
        # machines, which would keep plans solved on one from being used on another
        h.update(np.round(np.asarray(a), 6).tobytes())
    for mod in (mintime_dt, mintime, envelope):         # the track itself is hashed above
        h.update(open(mod.__file__, "rb").read())
    path = CACHE / f"{h.hexdigest()[:16]}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    result = mintime_dt.solve(track, car=car, grip=grip, margin=margin, step=step, verbose=verbose,
                              regen=regen, soc=soc, label=label)
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(result))
    return result


def regen_wheel_torque(car: Car, regen) -> float:
    """N m at the wheels the end of the brake pedal's regen travel asks for: every
    motor's most (the EMRAX's to the inverter's regen limit)."""
    t = regen.max_torque * car.final_drive / car.driveline_eff if car.central_drive else 0.0
    if car.hub_front or car.hub_rear:
        from kmslap.car.drivetrain import HubMotor
        n = 2 * car.hub_front + 2 * car.hub_rear
        t += n * HubMotor(car.hub_motor).peak_torque * car.hub_ratio / car.hub_gear_eff
    return t


def regen_power(car: Car, pack, soc: float) -> float:
    """W at the wheels the pack takes at this state of charge: its charge peak, within
    the cells' max voltage (as mintime_dt plans with)."""
    v_oc = pack.series * float(np.interp(soc, pack.ocv_soc, pack.ocv_cell))
    i = min(pack.peak_charge_current, max(pack.series * pack.v_cell_max - v_oc, 0.0) / pack.resistance)
    return i * (v_oc + i * pack.resistance) / (car.motor_eff * car.regen_inverter_eff)


def run(track_name: str = "michigan2026", grip: float = 0.85, margin: float = T.EDGE_MARGIN,
        step: float = 2.0, label: str | None = None, verbose: bool = False,
        width: float | None = None, drivetrain: str = "rwd_emrax", torque_mode: str = "qp",
        inverters: bool = False, regen_travel: float | None = None, soc: float = 1.0) -> dict:
    """width: the track's width (m), for a track that's a line only (michigan2026).
    drivetrain: a car/drivetrain.VARIANTS name; torque_mode / inverters: its hub
    motors' control (vcu/torque_control.MODES; through vcu/inverter's models).
    regen_travel: the brake pedal's regen-only travel (driving.policies.RegenPolicy);
    None: 0.9 with hub motors, 0 (no regen) without. soc: the pack's, planned and driven."""
    from kmslap.car.drivetrain import variant
    from kmslap.driving.drive import drive
    from kmslap.driving.follower import PathFollower
    from kmslap.reporting import runs

    car = variant(drivetrain)
    hubs = car.hub_front or car.hub_rear
    track = T.TRACK_NAMES[track_name](width) if width is not None else T.TRACK_NAMES[track_name]()
    if regen_travel is None:
        regen_travel = 0.9 if hubs else 0.0
    if not 0.0 <= regen_travel <= 0.9:
        raise ValueError(f"regen_travel {regen_travel}: 0..0.9 (FSAE T.3.2.4)")
    p = plan(track, grip, margin, step, car, verbose, regen=regen_travel > 0, soc=soc,
             label=f"{track.name} {drivetrain}")
    if not p.stats["success"]:
        print(f"warning: the solver stopped without converging ({p.stats['status']})")
    regen, kw = None, {}
    if regen_travel > 0:
        from kmslap.driving.policies import RegenPolicy
        from kmslap.car.powertrain import PACKS, Pack
        regen = RegenPolicy(regen_travel=regen_travel)
        kw = dict(regen_travel=regen_travel, regen_wheel_torque=regen_wheel_torque(car, regen),
                  regen_power=regen_power(car, Pack(PACKS["cosmx600v"]), soc))
    f = PathFollower.for_plan(p, car=car, laps=3, plan_grip=grip, **kw)
    line = p.line
    res = drive(f, car, psi0=float(line.heading[0] - p.beta_ff[0]), X0=float(line.x[0]),
                Y0=float(line.y[0]), t_max=3 * 4 * p.lap_time, log_period=0.02, torque_mode=torque_mode,
                inverters="native" if inverters and hubs else None, regen=regen, soc0=soc,
                progress=f"{track.name} {drivetrain}: driving the plan, double-track model"
                         + (f", {torque_mode} torque control" if hubs else "")
                         + (f", regen {regen_travel:.0%} travel" if regen_travel > 0 else ""))
    lt = f.lap_times
    finished = len(lt) > 1
    t1, t2 = (lt[0], lt[1]) if finished else (lt[0] if lt else 0.0, float(res.df.t.iloc[-1]))
    d = res.df[(res.df.t >= t1) & (res.df.t <= t2)]
    pen = T.penalties(track, d.X, d.Y, d.psi, car)
    label = label or (f"min-time {grip:.0%} grip, {margin:g} m from the edge"
                      + (f", {width:g} m wide" if width is not None else "")
                      + (f", {drivetrain} {torque_mode}" + (" +inv" if inverters else "") if hubs else "")
                      + (f", regen {regen_travel:.0%} travel" if regen_travel > 0 and not hubs else "")
                      + (f", {soc:.0%} soc" if soc != 1.0 else ""))
    runs.save_lap(track_name, label, res.df, t1, t2, finished=finished, penalties=str(pen),
                  penalty_s=pen.seconds, plan_lap_time=p.lap_time, grip=grip, margin=margin, width=width,
                  drivetrain=drivetrain, torque_mode=torque_mode if hubs else None,
                  regen_travel=regen_travel, soc=soc,
                  max_beta_deg=math.degrees(float(np.abs(d.beta).max())))
    lap = t2 - t1
    print(f"{track.name}\n  plan {p.lap_time:.2f} s | driven flying lap "
          f"{f'{lap:.2f} s' if finished else 'DNF'} | {pen} | saved as '{label}' (kmslap race {track_name})")
    return {"plan": p.lap_time, "lap": lap if finished else None, "penalties": pen}
