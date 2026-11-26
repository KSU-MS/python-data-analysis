"""Reference data to validate the sim against, and matching sim configs.

- OptimumLap accel run (olapsims/): 208MV, total ratio 4.0, 285 kg (from its static Fz).
- KS5e launches (cardata/ks5e/): inverter/VCU logs only, no wheel speed or IMU. Motor
  rpm is rear wheel speed, so it shows wheelspin directly; the rpm-derived v/x assume
  no slip and are only meaningful once the car hooks up.
  Motor (228HV) and pack (60s8p HE2) are inferred from the logs: torque falls off
  with rpm like the 228HV map at the logged bus voltage, and bus sag matches 0.24 ohm.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .params import Car

ROOT = Path(__file__).resolve().parent.parent
OPTIMUMLAP_CSV = ROOT / "olapsims" / "208mvTestStrip.csv"
KS5E_DIR = ROOT / "cardata" / "ks5e"

# matching sim setups: pass as simulate(**OPTIMUMLAP_SETUP)
OPTIMUMLAP_SETUP = dict(car=Car(final_drive=4.0), motor="208MV", pack="energus300v")
# KS5e final drive and mass are ASSUMPTIONS: the old notebook scaled ks5e torque by 3,
# and integrating rpm at FD 3.0 puts the driver's lift just past 75 m (FD >= ~4 would
# not reach the finish before the lift)
KS5E_SETUP = dict(car=Car(final_drive=3.0), motor="228HV", pack="energus250v")

_KS5E_COLS = {
    "D1_Commanded_Torque": "torque_request",
    "D2_Torque_Feedback": "motor_torque",
    "D2_Motor_Speed": "motor_rpm",
    "D1_DC_Bus_Voltage": "bus_v",
    "D4_DC_Bus_Current": "bus_i",
}


def load_optimumlap(path=OPTIMUMLAP_CSV) -> pd.DataFrame:
    """OptimumLap export in the sim's column names and SI units.

    motor_torque is net of OptimumLap's traction control cut. Note OptimumLap starts
    at 3.6 km/h and times from x = 0 (no 0.3 m rollout).
    """
    o = pd.read_csv(path)
    o = o.loc[:, ~o.columns.str.startswith("Unnamed")]
    return pd.DataFrame({
        "t": o["elapsedTime"],
        "x": o["elapsedDistance"],
        "v": o["speed"] / 3.6,
        "a": o["longitudinalAcceleration"],
        "motor_rpm": o["engineSpeed"],
        "motor_torque": o["torque"] * (1 - o["tractionControl"] / 100),
        "fz_total": o["fz"],
        "downforce": o["downForce"],
        "drag": o["dragForce"],
    })


def load_ks5e_runs(car: Car = KS5E_SETUP["car"], directory=KS5E_DIR,
                   gap_s: float = 1.0, min_peak_torque: float = 200.0,
                   max_start_rpm: float = 500.0) -> list[pd.DataFrame]:
    """Launches from the KS5e logs, lined up with t = 0 at torque application.

    The logger writes a separate segment per run (> gap_s between them), so each
    segment that reaches min_peak_torque from below max_start_rpm is a launch
    (same filter as the old accel_weight_transfer notebook). Short pulls are kept;
    each run ends when the driver lifts.
    """
    runs = []
    for path in sorted(Path(directory).glob("*.csv")):
        log = pd.read_csv(path, usecols=["Time", *_KS5E_COLS], parse_dates=["Time"])
        log = log.rename(columns=_KS5E_COLS)
        log["t"] = (log["Time"] - log["Time"].iloc[0]).dt.total_seconds()
        segment = (log["Time"].diff().dt.total_seconds() > gap_s).cumsum()
        for _, seg in log.groupby(segment):
            if seg["torque_request"].max() < min_peak_torque:
                continue
            if seg["motor_rpm"].min() > max_start_rpm:
                continue
            t0 = seg.loc[seg["torque_request"] > 20, "t"].min()
            lift = seg.loc[seg["torque_request"] > 100, "t"].max()
            run = seg[(seg["t"] >= t0) & (seg["t"] <= lift)].drop(columns="Time")
            run = run.reset_index(drop=True)
            run["t"] -= t0
            run["v"] = run["motor_rpm"] / car.final_drive * 2 * math.pi / 60 * car.tire_radius
            dt = np.diff(run["t"].to_numpy(), prepend=0.0)
            run["x"] = np.cumsum(run["v"].to_numpy() * dt)
            run.attrs["label"] = f"{path.stem.removeprefix('parseddata')} @ {t0:.1f}s"
            runs.append(run)
    return runs


def time_to(df: pd.DataFrame, distance: float, start: float = 0.0) -> float:
    """Time between crossing `start` and `distance` (linear interpolation)."""
    t, x = df["t"].to_numpy(), df["x"].to_numpy()
    if x[-1] < distance:
        return float("nan")
    return float(np.interp(distance, x, t) - np.interp(start, x, t))
