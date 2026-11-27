"""Dyno pulls against the car's inverter log and against kmslap's motor model.

    kmslap dyno --out dyno/            # every run in RUNS: plots + summary

Each run pairs a Dynojet export (roller speed and power at the wheels) with the
inverter / BMS log from the same pull (cardata/dyno/, trimmed from the team's log
archive). The two clocks are unrelated, so the logs are aligned by speed: the motor
speed trace is slid along the dyno's roller speed until they correlate best, which
also gives the ratio between them (final drive over the effective tire radius on the
roller).

Then, on the dyno's time base (the inverter signals interpolated, not held: the
2023 log has only 2-4 samples a second per signal, and holding them makes steps the
car didn't have; points without a fresh speed and DC current sample nearby are left
out of the summary):
    wheel power       dyno, from the roller's acceleration: the reference
    DC power          DC bus voltage x current (pack voltage when the bus voltage
                      signal is bad, as in the 2023 log)
    motor torque      from the dyno: wheel power / motor speed / driveline efficiency.
                      The inverter's own torque feedback (an estimate from its
                      currents) is shown against it: within ~5% on both pulls.
    model             kmslap's motor map: max torque at that speed and bus voltage; and
                      DC power for the dyno's wheel power through driveline, the EMRAX
                      efficiency map and the inverter
    overall           wheels / DC, measured vs model: the efficiency chain the sim
                      uses, checked end to end

The 2023 file came with a third export (DJDataExport_DYNORUN_015) that is the 2024
pull again, raw; it's in legacy/dynoruns/ with an empty summary-only export.
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from kmslap import dirs

DYNO = dirs.REPO / "cardata" / "dyno"
MPH = 0.44704
HP = 745.7
INVERTER_EFF = 0.97   # same assumption as Car.regen_inverter_eff, ESTIMATE
FRESH_S = 0.3         # a point counts if speed and DC current were sampled this close


@dataclass
class Run:
    name: str
    dyno_file: str
    inverter_file: str
    motor: str             # MotorMap name
    final_drive: float     # motor rev / wheel rev
    notes: str = ""


RUNS = {
    "ks6e_2023-04-26": Run("ks6e_2023-04-26", "ks6e_2023-04-26_dyno.txt", "ks6e_2023-04-26_inverter.csv",
                           motor="228MV", final_drive=3.0,
                           notes="KS6e (Energus 72s8p), 10/30 sprockets; the 4-26-23 dyno & regen day. "
                                 "DC bus voltage signal is bad in this log: pack voltage used."),
    "228hv_2024-03-03": Run("228hv_2024-03-03", "228hv_2024-03-03_dyno.txt", "228hv_2024-03-03_inverter.csv",
                            motor="228HV", final_drive=3.0,
                            notes="EMRAX 228HV pull, ~245 V pack; final drive assumed 3.0."),
}


def load_dynojet(path) -> pd.DataFrame:
    """A Dynojet text export (any of the layouts we have: plain, padded with N/A,
    leading -INF power) as t (s), v (m/s), wheel_power (W; NaN where none)."""
    rows = []
    for line in open(path, errors="replace"):
        parts = line.split()
        if len(parts) != 3:
            continue
        try:
            rows.append([np.nan if p in ("N/A", "-INF", "INF") else float(p) for p in parts])
        except ValueError:
            continue    # header
    d = pd.DataFrame(rows, columns=["s", "mph", "hp"]).dropna(subset=["s", "mph"])
    return pd.DataFrame({"t": d.s.to_numpy(), "v": d.mph.to_numpy() * MPH,
                         "wheel_power": d.hp.to_numpy() * HP})


def load_inverter(path) -> pd.DataFrame:
    """The inverter / BMS log as logged: t (s) and each CAN signal as its own sparse
    column (NaN between its messages), glitches dropped."""
    d = pd.read_csv(path)
    d.insert(0, "t", (d.Time - d.Time.iloc[0]) / 1000.0)
    d = d.drop(columns="Time")
    for col, lo, hi in [("D1_DC_Bus_Voltage", 50, 800), ("Pack_Inst_Voltage", 50, 800),
                        ("D2_Motor_Speed", -100, 8000), ("D4_DC_Bus_Current", -400, 600)]:
        if col in d:
            d.loc[(d[col] < lo) | (d[col] > hi), col] = np.nan
    return d


def signal(inv: pd.DataFrame, col: str):
    """(sample times, values) of one signal."""
    s = inv[["t", col]].dropna()
    return s.t.to_numpy(), s[col].to_numpy()


def staleness(t, sample_t):
    """Distance from each t to the nearest sample time."""
    i = np.clip(np.searchsorted(sample_t, t), 1, len(sample_t) - 1)
    return np.minimum(np.abs(t - sample_t[i - 1]), np.abs(sample_t[i] - t))


def align(dyno: pd.DataFrame, inv: pd.DataFrame, step: float = 0.01):
    """(offset s, rpm per m/s): inverter time = dyno time + offset, where the motor
    speed best correlates with the roller speed."""
    tv = dyno.t.to_numpy()
    v = dyno.v.to_numpy()
    ts, rs = signal(inv, "D2_Motor_Speed")
    best = (-np.inf, 0.0)
    offsets = np.arange(ts[0] - tv[0], ts[-1] - tv[-1], step)
    for off in offsets:
        rpm = np.interp(tv + off, ts, rs)
        if rpm.std() < 1:
            continue
        c = np.corrcoef(rpm, v)[0, 1]
        if c > best[0]:
            best = (c, off)
    off = best[1]
    rpm = np.interp(tv + off, ts, rs)
    k = float(np.sum(rpm * v) / np.sum(v * v))
    return float(off), k, float(best[0])


@dataclass
class Comparison:
    run: Run
    df: pd.DataFrame          # on the dyno's time base
    offset: float
    rpm_per_mps: float
    correlation: float
    summary: dict = field(default_factory=dict)


def compare(run: Run | str) -> Comparison:
    from kmslap.car.powertrain import MotorMap, motor_efficiency_map
    from scipy.interpolate import RegularGridInterpolator

    run = RUNS[run] if isinstance(run, str) else run
    dyno = load_dynojet(DYNO / run.dyno_file)
    inv = load_inverter(DYNO / run.inverter_file)
    off, k, corr = align(dyno, inv)

    d = dyno.copy()
    t = d.t.to_numpy() + off

    def at(col):
        st, sv = signal(inv, col)
        return np.interp(t, st, sv)

    bus_ok = inv.D1_DC_Bus_Voltage.notna().mean() > 0 and inv.D1_DC_Bus_Voltage.dropna().median() > 50
    vcol = "D1_DC_Bus_Voltage" if bus_ok else "Pack_Inst_Voltage"
    d["motor_rpm"] = at("D2_Motor_Speed")
    d["torque_feedback"] = at("D2_Torque_Feedback")
    d["torque_command"] = at("D1_Commanded_Torque")
    d["dc_voltage"] = at(vcol)
    d["dc_power"] = d.dc_voltage * at("D4_DC_Bus_Current")
    d["pack_power"] = at("Pack_Inst_Voltage") * at("Pack_Current")
    d["fresh"] = ((staleness(t, signal(inv, "D2_Motor_Speed")[0]) < FRESH_S)
                  & (staleness(t, signal(inv, "D4_DC_Bus_Current")[0]) < FRESH_S))
    w = (d.motor_rpm * 2 * np.pi / 60).clip(lower=1.0)

    from kmslap.car.params import Car
    eta_dl = Car.driveline_eff
    d["dyno_motor_torque"] = d.wheel_power / eta_dl / w          # what the motor delivered
    d["feedback_power"] = d.torque_feedback * w                  # the inverter's own estimate
    motor = MotorMap(run.motor)
    d["model_max_torque"] = [motor.max_torque(r, v) for r, v in zip(d.motor_rpm, d.dc_voltage)]
    rpms, torques, eff = motor_efficiency_map(run.motor)
    em = RegularGridInterpolator((rpms, torques), eff, bounds_error=False, fill_value=None)
    d["model_motor_eff"] = em(np.column_stack([d.motor_rpm.clip(rpms[0], rpms[-1]),
                                               d.dyno_motor_torque.clip(torques[0], torques[-1])]))
    d["model_overall_eff"] = d.model_motor_eff * INVERTER_EFF * eta_dl
    d["model_dc_power"] = d.wheel_power / d.model_overall_eff
    d["measured_overall_eff"] = d.wheel_power / d.dc_power

    # summary over the pull's high-power part (above half the dyno's peak)
    hot = d[d.fresh & (d.wheel_power > 0.5 * d.wheel_power.max())]
    r_eff = run.final_drive * 60 / (2 * np.pi * k)
    c = Comparison(run, d, off, k, corr)
    c.summary = {
        "run": run.name, "motor": run.motor,
        "speed_correlation": round(corr, 4),
        "effective_tire_radius_m": round(r_eff, 4),
        "peak_wheel_kw": round(d.wheel_power.max() / 1e3, 1),
        "peak_dc_kw": round(d.dc_power.max() / 1e3, 1),
        "points scored": int(len(hot)),
        "overall wheels/DC measured": round(float(hot.measured_overall_eff.median()), 3),
        "overall wheels/DC model": round(float(hot.model_overall_eff.median()), 3),
        "DC power model/measured": round(float((hot.model_dc_power / hot.dc_power).median()), 3),
        "inverter torque feedback / dyno torque": round(float((hot.torque_feedback / hot.dyno_motor_torque).median()), 3),
        "dyno torque / model max torque": round(float((hot.dyno_motor_torque / hot.model_max_torque).median()), 3),
    }
    return c


def plot(c: Comparison, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = c.df
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    a = ax[0, 0]
    a.plot(d.t, d.v / MPH, color="k", label="dyno roller (mph)")
    a2 = a.twinx()
    a2.plot(d.t, d.motor_rpm, color="#468ce6", label="inverter motor speed (rpm)")
    a.set(xlabel="dyno time (s)", ylabel="mph",
          title=f"Aligned by speed (r = {c.correlation:.4f}, offset {c.offset:.2f} s)")
    a2.set_ylabel("rpm")
    a = ax[0, 1]
    for col, lab, color in [("wheel_power", "wheels (dyno)", "k"), ("dc_power", "DC (measured)", "#e6504a"),
                            ("model_dc_power", "DC the model needs for the dyno's wheel power", "#e6aa32"),
                            ("feedback_power", "shaft, from inverter torque feedback", "#468ce6"),
                            ("pack_power", "pack (BMS)", "#5abe6e")]:
        a.plot(d.motor_rpm, d[col] / 1e3, ".", ms=2, color=color, label=lab)
    a.set(xlabel="motor rpm", ylabel="kW", title="Power along the pull")
    a.legend(fontsize=8, markerscale=4)
    a = ax[1, 0]
    a.plot(d.motor_rpm, d.dyno_motor_torque, ".", ms=2, color="k", label="from dyno power (reference)")
    a.plot(d.motor_rpm, d.torque_feedback, ".", ms=2, color="#468ce6", label="inverter torque feedback")
    a.plot(d.motor_rpm, d.torque_command, ".", ms=2, color="0.6", label="torque command")
    a.plot(d.motor_rpm, d.model_max_torque, ".", ms=2, color="#e6aa32", label="model max torque at this bus voltage")
    a.set(xlabel="motor rpm", ylabel="Nm", title="Motor torque")
    a.legend(fontsize=8, markerscale=4)
    a = ax[1, 1]
    hot = d[d.fresh & (d.wheel_power > 0.3 * d.wheel_power.max())]
    a.plot(hot.motor_rpm, hot.measured_overall_eff, "o", ms=3, color="#e6504a", label="measured wheels / DC")
    a.plot(hot.motor_rpm, hot.model_overall_eff, "o", ms=3, color="#e6aa32",
           label="model: driveline x motor map x inverter")
    a.set(xlabel="motor rpm", ylabel="efficiency", ylim=(0.5, 1.1),
          title="DC -> wheels efficiency (fresh samples, >30% of peak power)")
    a.legend(fontsize=8)
    for a in ax.flat:
        a.grid(alpha=0.3)
    fig.suptitle(f"{c.run.name}: {c.run.motor}, {c.run.notes}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run_all(out: str | None = None) -> list[dict]:
    out = dirs.out("dyno", out)          # None: out/dyno
    rows = []
    for name in RUNS:
        c = compare(name)
        plot(c, out / f"{name}.png")
        rows.append(c.summary)
    table = pd.DataFrame(rows).set_index("run").T
    table.to_csv(out / "summary.csv")
    print(table.to_string())
    return rows
