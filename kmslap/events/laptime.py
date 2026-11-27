"""Lap time on a track: the double-track minimum-time plan (models/mintime_dt), driven by
the PathFollower with the plan's feedforward in the time-domain car, scored for FSAE
penalties, the flying lap saved (reporting/runs) to race against other versions.

    kmslap laptime ac18 --grip 0.85 --margin 0.35
    kmslap race ac18
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
         car: Car = None, verbose: bool = False):
    """mintime_dt.solve, cached in build/mintime/ by the track, car, settings and the
    solvers' sources (a solve takes ~5-7 minutes)."""
    from kmslap.models import envelope, mintime, mintime_dt

    car = car or Car()
    h = hashlib.sha1(repr((dataclasses.astuple(car), grip, margin, step)).encode())
    for a in (track.center.x, track.center.y, track.w_left, track.w_right):
        h.update(np.asarray(a).tobytes())
    for mod in (mintime_dt, mintime, envelope, T):
        h.update(open(mod.__file__, "rb").read())
    path = CACHE / f"{h.hexdigest()[:16]}.pkl"
    if path.exists():
        return pickle.loads(path.read_bytes())
    result = mintime_dt.solve(track, car=car, grip=grip, margin=margin, step=step, verbose=verbose)
    CACHE.mkdir(parents=True, exist_ok=True)
    path.write_bytes(pickle.dumps(result))
    return result


def run(track_name: str = "michigan2026", grip: float = 0.85, margin: float = T.EDGE_MARGIN,
        step: float = 2.0, label: str | None = None, verbose: bool = False,
        width: float | None = None) -> dict:
    """width: the track's width (m), for a track that's a line only (michigan2026)."""
    from kmslap.driving.drive import drive
    from kmslap.driving.follower import PathFollower
    from kmslap.reporting import runs

    car = Car()
    track = T.TRACK_NAMES[track_name](width) if width is not None else T.TRACK_NAMES[track_name]()
    p = plan(track, grip, margin, step, car, verbose)
    if not p.stats["success"]:
        print(f"warning: the solver stopped without converging ({p.stats['status']})")
    f = PathFollower.for_plan(p, car=car, laps=3)
    line = p.line
    res = drive(f, car, psi0=float(line.heading[0] - p.beta_ff[0]), X0=float(line.x[0]),
                Y0=float(line.y[0]), t_max=3 * 4 * p.lap_time, log_period=0.02)
    lt = f.lap_times
    finished = len(lt) > 1
    t1, t2 = (lt[0], lt[1]) if finished else (lt[0] if lt else 0.0, float(res.df.t.iloc[-1]))
    d = res.df[(res.df.t >= t1) & (res.df.t <= t2)]
    pen = T.penalties(track, d.X, d.Y, d.psi, car)
    label = label or (f"min-time {grip:.0%} grip, {margin:g} m from the edge"
                      + (f", {width:g} m wide" if width is not None else ""))
    runs.save_lap(track_name, label, res.df, t1, t2, finished=finished, penalties=str(pen),
                  penalty_s=pen.seconds, plan_lap_time=p.lap_time, grip=grip, margin=margin, width=width,
                  max_beta_deg=math.degrees(float(np.abs(d.beta).max())))
    lap = t2 - t1
    print(f"{track.name}\n  plan {p.lap_time:.2f} s | driven flying lap "
          f"{f'{lap:.2f} s' if finished else 'DNF'} | {pen} | saved as '{label}' (kmslap race {track_name})")
    return {"plan": p.lap_time, "lap": lap if finished else None, "penalties": pen}
