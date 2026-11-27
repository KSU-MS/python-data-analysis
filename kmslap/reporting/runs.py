"""Saved laps: the time series of driven flying laps, kept to race them against each
other later (birdseye.lap_gif) and compare what changed between versions of a line,
driver or strategy.

    save_lap("ac18", "min-time (double track)", df, t_start, t_end, lap_time=..., ...)
    laps = load_laps("ac18")          # {label: lap DataFrame}, in the order saved

Each lap is out/laps/<track>/<slug>.csv.gz (time from the lap's start line, distance
along it, position, heading, speed, inputs, body slip, what the car is doing) with
<slug>.json beside it (label, when, and whatever metadata the caller gives: lap time,
penalties, settings).
"""
import json
import re
import time

import numpy as np
import pandas as pd

from kmslap import dirs

RUNS = dirs.LAPS
COLUMNS = ["t_lap", "dist", "X", "Y", "psi", "speed", "beta", "r", "ax", "ay", "steering_wheel",
           "pedal", "brake", "motor_torque", "drs", "soc", "pack_temp", "mode"]


def lap_frame(df: pd.DataFrame, t_start: float, t_end: float) -> pd.DataFrame:
    """One lap of a drive log: time and distance from the lap's start, and the mode
    (drive / coast / brake / regen) at each sample."""
    d = df[(df.t >= t_start) & (df.t <= t_end)].copy()
    d["t_lap"] = d.t - t_start
    d["dist"] = np.r_[0, np.cumsum(np.hypot(np.diff(d.X), np.diff(d.Y)))]
    # every motor's torque: the EMRAX's and any hub motors'
    torque = d.motor_torque + sum(d[c] for c in ("hub_torque_fl", "hub_torque_fr", "hub_torque_rl",
                                                  "hub_torque_rr") if c in d)
    d["mode"] = np.select([torque < -1, d.brake > 0.02, d.pedal > 0.02],
                          ["regen", "brake", "drive"], "coast")
    return d


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-")


def save_lap(track: str, label: str, df: pd.DataFrame, t_start: float, t_end: float, **meta):
    """Save one lap (t_start..t_end of the drive log df) under the track's name."""
    folder = RUNS / track
    folder.mkdir(parents=True, exist_ok=True)
    d = lap_frame(df, t_start, t_end)
    d[[c for c in COLUMNS if c in d.columns]].to_csv(folder / f"{_slug(label)}.csv.gz", index=False,
                                                     float_format="%.5g")
    info = {"label": label, "saved": time.strftime("%Y-%m-%d %H:%M:%S"),
            "lap_time": float(t_end - t_start), **meta}
    (folder / f"{_slug(label)}.json").write_text(json.dumps(info, indent=1, default=str))
    return folder / f"{_slug(label)}.csv.gz"


def load_laps(track: str, labels=None) -> dict:
    """{label: lap DataFrame} for the track, by label (or the given labels';
    labels may also be slugs). DataFrame.attrs holds the metadata."""
    folder = RUNS / track
    infos = sorted((json.loads(p.read_text()) for p in folder.glob("*.json")), key=lambda i: i["label"])
    if labels is not None:
        want = {_slug(l) for l in labels}
        infos = [i for i in infos if _slug(i["label"]) in want]
    laps = {}
    for info in infos:
        d = pd.read_csv(folder / f"{_slug(info['label'])}.csv.gz")
        d.attrs = info
        laps[info["label"]] = d
    return laps
