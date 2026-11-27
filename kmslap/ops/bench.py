"""Benchmarks: how fast a machine runs our sims, alone and in parallel, and the
cluster across machines.

    accel bench single  --out bench/        # single-thread timings
    accel bench parallel --n 96 --out bench/  # strategy evaluations, all cores
    accel bench cluster --n 192 --remote mathewos@desktop-17id03o-2 --out bench/

single:   1D accel run, VCU controller in the loop, one double-track lap of the
          endurance line, a small grip envelope solve (best of a few repeats).
parallel: a fixed set of kmslap.strategy candidates (2-lap endurance evaluations,
          ~4 s each), all cores, through a process pool: evaluations per second.
cluster:  the same candidates through kmslap.ops.cluster across this machine and the
          remotes (live dashboard in a terminal); also saves a timeline plot.
Each writes a JSON (machine, cores, timings) into --out.
"""
import itertools
import json
import math
import os
import platform
import socket
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def _best(fn, repeats=3):
    best = math.inf
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _machine():
    cpu = "?"
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    return {"host": socket.gethostname(), "cores": os.cpu_count(), "cpu": cpu,
            "kernel": platform.release()}


def single() -> dict:
    from kmslap.driving.drive import drive
    from kmslap.driving.follower import PathFollower
    from kmslap.models.envelope import gg_envelope
    from kmslap.vcu.firmware import FirmwareController
    from kmslap.car.params import Car
    from kmslap.events.accel import simulate
    from kmslap.events.track import michigan_2026_endurance

    car = Car().open_aero()
    simulate(car)                                   # warm caches / compiled models
    path = michigan_2026_endurance()
    lap = lambda: drive(PathFollower(path, speed=12.0, laps=1), Car(), t_max=200,
                        psi0=float(path.heading[0]))
    lap()
    return {
        "accel_1d_s": _best(lambda: simulate(car)),
        "accel_1d_vcu_s": _best(lambda: simulate(car, controller=FirmwareController(belief=car))),
        "endurance_lap_2d_s": _best(lap, 2),
        "envelope_small_s": _best(lambda: gg_envelope(Car(), speeds=[10.0, 20.0], n_angles=9,
                                                      cache=False), 1),
    }


def candidates(n: int) -> list[dict]:
    """n strategy candidates, the same on every machine."""
    grid = itertools.product([0.8, 0.85, 0.9, 0.95, 1.0], [0.0, 0.5, 1.0, 1.5, 2.0],
                             [None, {"min_speed": 8.0, "max_lateral": 8.8,
                                     "max_steering_wheel": math.radians(20)}],
                             [None, {"lift_torque": 0.0, "brake_fraction": 1.0}])
    out = [{"margin": m, "coast_time": c, "drs": d, "regen": r} for m, c, d, r in grid]
    return (out * (n // len(out) + 1))[:n]


def _warm():
    from kmslap.strategy import _speed
    for m in (0.8, 0.85, 0.9, 0.95, 1.0):
        _speed(m)


def parallel(n: int, workers: int | None = None) -> dict:
    from kmslap.strategy import evaluate

    _warm()
    cands = candidates(n)
    workers = workers or os.cpu_count()
    t0 = time.perf_counter()
    with ProcessPoolExecutor(workers) as ex:
        list(ex.map(evaluate, cands))
    el = time.perf_counter() - t0
    return {"n": n, "workers": workers, "wall_s": el, "evals_per_s": n / el}


def cluster(n: int, remotes: list[str], local_workers: int | None, timeline: Path | None) -> dict:
    from kmslap.ops.cluster import Cluster

    _warm()
    cands = candidates(n)
    labels = [f"pace {c['margin']:.2f} coast {c['coast_time']:.1f}"
              f"{' DRS' if c['drs'] else ''}{' regen' if c['regen'] else ''}" for c in cands]
    with Cluster(local_workers=local_workers, remotes=remotes) as c:
        t0 = time.perf_counter()
        c.map("kmslap.strategy:evaluate", cands, labels=labels, title=f"bench: {n} strategy evaluations")
        el = time.perf_counter() - t0
    if timeline:
        c.save_timeline(timeline)
    per_host = {h: sum(1 for x in c.history if x[1] == h) for h in c.hosts}
    return {"n": n, "hosts": c.hosts, "tasks_per_host": per_host, "wall_s": el, "evals_per_s": n / el}


def record(what: str, n: int = 96, workers: int | None = None, remote: list[str] = (),
           out: str = "bench") -> dict:
    """Run one benchmark (single / parallel / cluster), save and print its JSON."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    m = _machine()
    if what == "single":
        res = single()
    elif what == "parallel":
        res = parallel(n, workers)
    else:
        res = cluster(n, list(remote), workers, out / "cluster_timeline.png")
    rec = {"machine": m, "bench": what, **res}
    (out / f"{what}_{m['host']}.json").write_text(json.dumps(rec, indent=1, default=str))
    print(json.dumps(rec, indent=1, default=str))
    return rec

