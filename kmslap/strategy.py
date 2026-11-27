"""Tune the endurance driving strategy: pace (fraction of the grip envelope), lift-and-
coast, and the DRS policy, for the fastest event that still finishes.

A candidate is scored on 2 laps from a standing start, projected to the event (two
stints: 2 standing laps + the rest flying): time, and charge used. Objective: event
time, plus 30 s for every 1% of charge the projection ends below SOC_FLOOR (the
margin before the BMS undervoltage limit cuts power). Search: a coarse grid of pace x
coast x DRS on/off, then the DRS parameters around the best, then (--regen) regen
strategies on top, run in parallel; the winners are checked with full events.

    accel strategy --out strategy/ [--regen] [--remote mathewos@desktop-17id03o-2]
"""
import itertools
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

SOC_FLOOR = 0.10
PENALTY_S_PER_SOC = 3000.0   # 30 s per 1% short
EVENT_LAPS = 18

_profiles = {}


def _speed(margin):
    from .envelope import gg_envelope, speed_profile
    from .params import Car
    from .track import michigan_2026_endurance

    if margin not in _profiles:
        path = michigan_2026_endurance()
        _profiles[margin] = (path, speed_profile(path, gg_envelope(Car()), margin=margin))
    return _profiles[margin]


def _soc_after(energy_j: float) -> float:
    """SOC left after drawing energy_j from the cells, from full, through the pack's
    open-circuit voltage curve."""
    from .powertrain import PACKS, Pack

    pack = Pack(PACKS["cosmx600v"])
    soc = np.linspace(1.0, 0.0, 1001)
    ocv = pack.series * np.interp(soc, pack.ocv_soc, pack.ocv_cell)
    # energy to go from full down to each soc: integral of Voc dQ
    drawn = np.concatenate([[0.0], np.cumsum(0.5 * (ocv[1:] + ocv[:-1]) * -np.diff(soc))])
    drawn *= pack.capacity_ah * 3600
    return float(np.interp(energy_j, drawn, soc))


def evaluate(c: dict) -> dict:
    """c: margin, coast_time, drs (None or DrsPolicy kwargs), regen (None or
    RegenPolicy kwargs). Two laps, projected."""
    from .driver import DrsPolicy, RegenPolicy
    from .endurance import endurance
    from .params import Car

    path, speed = _speed(c["margin"])
    res = endurance(Car(), laps=2, path=path, speed=speed, coast_time=c["coast_time"],
                    drs=DrsPolicy(**c["drs"]) if c["drs"] is not None else None,
                    regen=RegenPolicy(**c["regen"]) if c.get("regen") else None,
                    driver_change=False, log_period=0.01)
    d, laps = res.df, res.lap_times
    row = {**{k: v for k, v in c.items() if k not in ("drs", "regen", "soc_floor")},
           "drs": c["drs"] is not None, **{f"drs_{k}": v for k, v in (c["drs"] or {}).items()},
           "regen": bool(c.get("regen")), **{f"regen_{k}": v for k, v in (c.get("regen") or {}).items()}}
    if len(laps) < 2:
        return {**row, "objective": math.inf, "finished_2": False}
    # energy from the cells (open-circuit voltage x current: includes the internal
    # losses) is ~constant per lap; charge isn't (current rises as the voltage sags)
    t1, t2 = laps[0], laps[0] + laps[1]
    e_cells = np.cumsum(d.pack_v_oc * d.bus_i) * 0.01
    e = lambda t: float(np.interp(t, d.t, e_cells))
    e1, e2 = e(t1), e(t2) - e(t1)
    stints = 2
    flying = EVENT_LAPS - stints
    time = stints * laps[0] + flying * laps[1]
    soc_end = _soc_after(stints * e1 + flying * e2)
    short = max(0.0, c.get("soc_floor", SOC_FLOOR) - soc_end)
    return {**row, "finished_2": True, "lap_standing_s": laps[0], "lap_flying_s": laps[1],
            "event_time_s": time, "soc_end": soc_end, "drs_open": float((d.drs > 0.5).mean()),
            "objective": time + PENALTY_S_PER_SOC * short}


_cluster = None   # an accel.cluster.Cluster to run on instead of a local pool


def _map(fn, cands, workers):
    if _cluster is not None:
        labels = [f"pace {c['margin']:.2f} coast {c['coast_time']:.1f}"
                  f"{' DRS' if c['drs'] else ''}{' regen' if c.get('regen') else ''}" for c in cands]
        return _cluster.map(f"accel.strategy:{fn.__name__}", cands, labels=labels,
                            title=f"strategy: {fn.__name__}")
    with ProcessPoolExecutor(workers) as ex:
        return list(ex.map(fn, cands))


def _run(cands, workers):
    return pd.DataFrame(_map(evaluate, cands, workers))


def full_event(c: dict) -> dict:
    from .driver import DrsPolicy, RegenPolicy
    from .endurance import endurance
    from .params import Car

    path, speed = _speed(c["margin"])
    res = endurance(Car(), laps=EVENT_LAPS, path=path, speed=speed, coast_time=c["coast_time"],
                    drs=DrsPolicy(**c["drs"]) if c["drs"] is not None else None,
                    regen=RegenPolicy(**c["regen"]) if c.get("regen") else None)
    return {**res.summary, "candidate": c}


def _candidate(row) -> dict:
    """A results row back into the candidate it scored."""
    drs = regen = None
    if row.drs:
        drs = {k: float(row[f"drs_{k}"]) for k in ("min_speed", "max_lateral", "max_steering_wheel")}
    if row.get("regen", False):
        regen = {k: float(row[f"regen_{k}"]) for k in ("lift_torque", "brake_fraction")}
    return {"margin": float(row.margin), "coast_time": float(row.coast_time), "drs": drs, "regen": regen}


def tune(workers: int | None = None, soc_floor: float = SOC_FLOOR, regen: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(every candidate scored, the picks confirmed by full events).

    regen: also search regen strategies (lift-off torque x brake blending, with and
    without the best DRS policy), on top of pace and coasting."""
    default_drs = {"min_speed": 12.0, "max_lateral": 0.3 * 9.81, "max_steering_wheel": math.radians(180)}
    margins = [0.7, 0.75, 0.8, 0.85, 0.9, 0.93, 0.96, 1.0]
    coasts = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    stage1 = [{"margin": m, "coast_time": ct, "drs": d, "soc_floor": soc_floor}
              for m, ct, d in itertools.product(margins, coasts, [None, default_drs])]
    r1 = _run(stage1, workers)
    r1["stage"] = 1
    # DRS parameters around the best few pace / coast pairs
    best = r1.sort_values("objective").drop_duplicates(["margin", "coast_time"]).head(3)
    stage2 = [{"margin": m, "coast_time": ct, "soc_floor": soc_floor,
               "drs": {"min_speed": v, "max_lateral": g * 9.81, "max_steering_wheel": math.radians(sw)}}
              for (m, ct) in zip(best.margin, best.coast_time)
              for v, g, sw in itertools.product([8.0, 11.0, 14.0, 17.0], [0.2, 0.4, 0.6, 0.9],
                                                [10.0, 20.0, 40.0, 180.0])]
    r2 = _run(stage2, workers)
    r2["stage"] = 2
    stages = [r1, r2]
    if regen:
        best_drs = _candidate(r2.sort_values("objective").iloc[0])["drs"]
        stage3 = [{"margin": m, "coast_time": ct, "drs": d, "soc_floor": soc_floor,
                   "regen": {"lift_torque": lift, "brake_fraction": bf}}
                  for m, ct, d, lift, bf in itertools.product(
                      [0.9, 0.95, 1.0], [0.0, 0.5, 1.0, 1.5], [None, best_drs],
                      [0.0, 20.0, 40.0, 80.0], [0.0, 0.5, 1.0])]
        r3 = _run(stage3, workers)
        r3["stage"] = 3
        stages.append(r3)
    results = pd.concat(stages, ignore_index=True).sort_values("objective")
    results["regen"] = results.get("regen", False)
    results["regen"] = results["regen"].fillna(False).astype(bool)
    no_regen = results[~results.regen]
    picks = {"best": results.iloc[0], "best without regen": no_regen.iloc[0],
             "best without DRS or regen": no_regen[~no_regen.drs].iloc[0],
             "best with default DRS": results[(results.stage == 1) & results.drs].iloc[0]}
    if not regen:
        picks.pop("best without regen")
    confirmed = pd.DataFrame(_map(full_event, [_candidate(p) for p in picks.values()], workers))
    confirmed.insert(0, "pick", list(picks))
    return results, confirmed


def run(out: str = "strategy", workers: int | None = None, regen: bool = False,
        remote: list[str] = (), soc_floor: float = SOC_FLOOR):
    """Tune, write candidates / confirmed CSVs and the trade-off plot into `out`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .envelope import gg_envelope
    from .params import Car

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    gg_envelope(Car())       # solve / cache once before the workers need it
    global _cluster
    if remote:
        from .cluster import Cluster
        _cluster = Cluster(local_workers=workers, remotes=list(remote)).__enter__()
    try:
        results, confirmed = tune(workers, soc_floor, regen=regen)
    finally:
        if _cluster is not None:
            _cluster.__exit__(None, None, None)
            _cluster.save_timeline(out / "cluster_timeline.png")
    results.to_csv(out / "candidates.csv", index=False)
    confirmed.to_csv(out / "confirmed.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(results.head(15).round(3).to_string(index=False))
        print()
        print(confirmed.round(3).T.to_string())

    fig, ax = plt.subplots(figsize=(9, 6))
    for drs, c in ((False, "#468ce6"), (True, "#e6504a")):
        r = results[(results.drs == drs) & results.finished_2]
        ax.scatter(100 * r.soc_end, r.event_time_s, s=12, color=c, alpha=0.6,
                   label="DRS" if drs else "no DRS")
    ax.axvline(100 * soc_floor, color="k", ls=":", lw=1, label="SOC floor")
    ax.set(xlabel="projected SOC at the end of the event (%)", ylabel="projected event time (s)",
           title="Endurance strategies: pace x lift-and-coast x DRS (2-lap projections)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "tradeoff.png")

