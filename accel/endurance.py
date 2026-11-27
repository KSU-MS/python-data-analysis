"""The FSAE endurance event in the double-track model: ~22 km of laps, a driver change
at half distance (car stopped 3 minutes), one pack with no active cooling. Where pack
temperature and capacity, driving style and drag matter most.

    res = endurance(Car(), laps=18)
    res.summary        # time, energy, SOC, pack temperature, drag energy, lap times
    res.df             # the whole run, both stints

    accel endurance --out endurance.png
        # wing closed, wing set open by hand, DRS, lift-and-coast
"""
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .drive import drive
from .driver import DrsPolicy, PathFollower
from .envelope import gg_envelope, speed_profile
from .params import Car
from .powertrain import PACKS, Pack
from .track import michigan_2026_endurance

DRIVER_CHANGE_S = 180.0


@dataclass
class EnduranceResult:
    df: pd.DataFrame
    summary: dict
    lap_times: list


def endurance(car: Car = None, *, laps: int = 18, margin: float = 0.8, coast_time: float = 0.0,
              drs: DrsPolicy | None = None, regen=None, pack: Pack | str = "cosmx600v", path=None,
              speed=None, driver_change: bool = True, soc0: float = 1.0,
              pack_temp0: float | None = None, log_period: float = 0.02, label: str = "") -> EnduranceResult:
    """Run the event: two stints of laps/2 from a standing start, the pack cooling
    to ambient between them for DRIVER_CHANGE_S.

    margin: drive at this fraction of the car's grip envelope (endurance pace).
    speed: target speed along the path instead of the envelope's (m/s per point).
    """
    car = car or Car()
    pack = Pack(PACKS[pack]) if isinstance(pack, str) else pack
    path = path or michigan_2026_endurance()
    if speed is None:
        speed = speed_profile(path, gg_envelope(car), margin=margin)
    stints = [laps // 2, laps - laps // 2] if driver_change and laps > 1 else [laps]
    soc, temp = soc0, pack.ambient_c if pack_temp0 is None else pack_temp0
    frames, lap_times, t_offset = [], [], 0.0
    for n, stint_laps in enumerate(stints):
        if n > 0:   # driver change: stopped, no current, the pack cools toward ambient
            tau = pack.thermal_mass / pack.h_a if pack.h_a > 0 else math.inf
            temp = pack.ambient_c + (temp - pack.ambient_c) * math.exp(-DRIVER_CHANGE_S / tau)
        follower = PathFollower(path, speed=speed, car=car, laps=stint_laps, coast_time=coast_time,
                                brake_regen_fraction=regen.brake_fraction if regen else 0.0,
                                pedal_regen_torque=regen.lift_torque if regen else 0.0)
        res = drive(follower, car, pack=pack, drs=drs, regen=regen, soc0=soc, pack_temp0=temp,
                    psi0=float(path.heading[0]), t_max=stint_laps * 200.0, log_period=log_period)
        d = res.df.copy()
        d["stint"] = n
        d["t_event"] = d.t + t_offset
        frames.append(d)
        starts = [0.0] + follower.lap_times
        lap_times += list(np.diff(starts))
        soc, temp = float(d.soc.iloc[-1]), float(d.pack_temp.iloc[-1])
        t_offset += d.t.iloc[-1] + (DRIVER_CHANGE_S if n + 1 < len(stints) else 0.0)
        if follower.laps_done < stint_laps:
            break    # ran out of time: off the track, or the pack ran flat
    df = pd.concat(frames, ignore_index=True)
    dt = log_period
    power = df.bus_v * df.bus_i
    summary = {
        "label": label,
        "finished": len(lap_times) >= laps,
        "laps": len(lap_times),
        "driving_time_s": float(sum(lap_times)),
        "best_lap_s": float(min(lap_times)) if lap_times else math.nan,
        "energy_kwh": float(power.sum() * dt / 3.6e6),                 # net, from the pack
        "regen_kwh": float(-power.clip(upper=0).sum() * dt / 3.6e6),   # put back by regen
        "drag_energy_kwh": float((df.drag * df.speed).sum() * dt / 3.6e6),
        "soc_end": float(df.soc.iloc[-1]),
        "pack_temp_max_c": float(df.pack_temp.max()),
        "pack_temp_end_c": float(df.pack_temp.iloc[-1]),
        "pack_heat_kj": float(df.pack_heat.sum() * dt / 1e3),
        "min_bus_v": float(df.bus_v.min()),
        "drs_open_fraction": float((df.drs > 0.5).mean()),
    }
    return EnduranceResult(df, summary, lap_times)


def compare(laps: int = 18, margin: float = 0.8, out: str = "endurance.png"):
    """Wing closed / open / DRS / lift-and-coast over the event, plotted into `out`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    car, open_car = Car(), Car().open_aero()
    path = michigan_2026_endurance()
    # each aero setup drives to its own envelope; DRS closes for corners, so the
    # closed wing's
    speed = speed_profile(path, gg_envelope(car), margin=margin)
    speed_open = speed_profile(path, gg_envelope(open_car), margin=margin)
    common = dict(laps=laps, path=path)
    runs = {
        "wing closed": endurance(car, speed=speed, label="wing closed", **common),
        "wing open (set by hand)": endurance(open_car, speed=speed_open, label="wing open", **common),
        "DRS": endurance(car, speed=speed, drs=DrsPolicy(), label="DRS", **common),
        "wing closed, lift and coast 1 s": endurance(car, speed=speed, coast_time=1.0,
                                                     label="closed, coast", **common),
    }

    table = pd.DataFrame([r.summary for r in runs.values()]).set_index("label")
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print(table.round(3).T)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for (name, r), c in zip(runs.items(), ["#468ce6", "#e6aa32", "#e6504a", "#5abe6e"]):
        d = r.df
        axes[0, 0].plot(d.t_event / 60, d.pack_temp, color=c, label=name)
        axes[0, 1].plot(d.t_event / 60, 100 * d.soc, color=c)
        axes[1, 0].plot(np.arange(1, len(r.lap_times) + 1), r.lap_times, "o-", color=c)
        lap1 = d[(d.stint == 0) & (d.t < r.lap_times[0])] if r.lap_times else d
        axes[1, 1].plot(lap1.X, lap1.Y, color=c, lw=1)
    axes[1, 1].plot(path.x, path.y, "k:", lw=0.6)
    axes[1, 1].set_aspect("equal")
    axes[0, 0].legend()
    for ax, (title, x, y) in zip(axes.flat, [("pack temperature", "min", "C"), ("state of charge", "min", "%"),
                                             ("lap times", "lap", "s"), ("lap 1 line", "m", "m")]):
        ax.set(title=title, xlabel=x, ylabel=y)
        ax.grid(alpha=0.3)
    fig.suptitle(f"FSAE Michigan 2026 EV endurance, {laps} laps at {margin:.0%} of the grip envelope")
    fig.tight_layout()
    fig.savefig(out)

