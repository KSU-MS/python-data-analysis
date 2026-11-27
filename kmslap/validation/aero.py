"""The car's aero from on-track tests: drag from coast-downs, downforce and balance from
the shock pots. Reported, not applied: Car()'s aero is unchanged by this.

    kmslap aero --out aero/        # fits, plots, summary

Data: the KS7e aero tests of 6 July 2024, the same day with and without the aero
package (cardata/aero/, trimmed from the team log archive to when the car was moving,
25 Hz): VectorNav velocity / acceleration / yaw rate, inverter torque, front / rear
brake pressure, the four shock pots.

Drag. Coast-downs: no torque (feedback and command), no brakes, nearly straight, above
6 m/s, at least 2 s. Deceleration from the VectorNav speed, corrected for road grade
(its vertical velocity). The model, -m a = F0 + 1/2 rho CdA_config v^2, shares F0
(rolling + driveline drag: the same car) between configurations: over 7-13 m/s the v^2
and constant terms can't be told apart per configuration alone, so the package's drag
(the difference) is much firmer than either absolute CdA.

Downforce. No spring rates or motion ratios needed: each axle's shock pots are
calibrated in counts per newton from longitudinal load transfer (an axle's load changes
by m a h / L per m/s^2), in the runs without aero below 9 m/s; then axle compression vs
v^2 at near-zero acceleration (offsets per run) is its downforce. The front pots count
up in extension, the rear ones in compression (checked: corners and acceleration agree
on it). Total downforce scales with the mass and CG height estimates; the front / rear
balance doesn't. Anti-dive / anti-squat geometry would make the calibration, and so the
downforce, read somewhat high.

Speeds only reached ~19 m/s, so these rest on 7-19 m/s; there was no DRS test.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from kmslap import dirs

AERO = dirs.REPO / "cardata" / "aero"
G = 9.81
RHO = 1.20            # kg/m^3, a July morning
DT = 0.04             # s, the trimmed data's sample period


def runs() -> dict[str, str]:
    """run name -> configuration ("aero" / "noaero")."""
    return {p.name.replace(".csv.gz", ""): ("aero" if "_aero_" in p.name else "noaero")
            for p in sorted(AERO.glob("2024-07-06_*.csv.gz"))}


def load(name: str) -> pd.DataFrame:
    from scipy.signal import savgol_filter

    d = pd.read_csv(AERO / f"{name}.csv.gz")
    # contiguous moving stretches only (the trim left gaps where the car sat)
    gap = np.r_[True, np.diff(d.t) > 2 * DT]
    d["stretch"] = np.cumsum(gap)
    parts = []
    for _, s in d.groupby("stretch"):
        if len(s) < 51:
            continue
        s = s.copy()
        v = np.hypot(s.v_n, s.v_e).to_numpy()
        sg = lambda x, w, **kw: savgol_filter(np.asarray(x, float), w, 2, **kw)
        s["v"] = sg(v, 25)                                         # 1 s
        s["accel"] = sg(v, 25, deriv=1, delta=DT)
        s["grade"] = -sg(s.v_d, 25) / np.maximum(v, 1.0)          # v_d is down-positive
        s["ax_s"], s["ay_s"] = sg(s.ax, 13), sg(s.ay, 13)         # 0.5 s (pots are ~17 Hz)
        s["front"] = -sg(s.pot_fl + s.pot_fr, 13)                 # compression, counts
        s["rear"] = sg(s.pot_rl + s.pot_rr, 13)
        parts.append(s)
    out = pd.concat(parts, ignore_index=True)
    out["run"], out["cfg"] = name, runs()[name]
    return out


@dataclass
class Drag:
    f0: float                   # N, rolling + driveline, shared
    cda: dict                   # config -> m^2
    cda_se: dict
    package_cda: float
    package_cda_se: float
    samples: pd.DataFrame


def coastdown(data: pd.DataFrame, mass: float) -> Drag:
    rows = []
    for (run, _), s in data.groupby(["run", "stretch"]):
        bf, br = s.brake_f.median(), s.brake_r.median()
        coast = ((s.torque_fb.abs() < 2) & (s.torque_cmd.abs() < 2) & (s.brake_f < bf + 30)
                 & (s.brake_r < br + 30) & (s.v > 6) & (s.wz.abs() < 0.15)).to_numpy()
        seg = np.cumsum(np.r_[True, coast[1:] != coast[:-1]])
        for k in np.unique(seg[coast]):
            idx = np.nonzero(seg == k)[0]
            if len(idx) * DT < 2.0:
                continue
            q = s.iloc[idx[12:-12:2]]                 # drop 0.5 s at each end (filters, transitions)
            rows.append(q.assign(seg=f"{run}/{k}", a_flat=q.accel + G * q.grade))
    c = pd.concat(rows, ignore_index=True)
    X = np.column_stack([np.ones(len(c)), 0.5 * RHO * c.v ** 2 * (c.cfg == "aero"),
                         0.5 * RHO * c.v ** 2 * (c.cfg == "noaero")])
    y = -mass * c.a_flat
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ coef
    cov = np.linalg.inv(X.T @ X) * r.var() * 10       # samples within a coast aren't independent
    se = np.sqrt(np.diag(cov))
    return Drag(coef[0], {"aero": coef[1], "noaero": coef[2]}, {"aero": se[1], "noaero": se[2]},
                coef[1] - coef[2], float(np.sqrt(cov[1, 1] + cov[2, 2] - 2 * cov[1, 2])), c)


@dataclass
class Downforce:
    counts_per_n: dict          # axle -> counts/N
    cla: dict                   # (axle, cfg) -> m^2
    cla_se: dict
    samples: pd.DataFrame

    def total(self, cfg):
        return self.cla["front", cfg] + self.cla["rear", cfg]

    def balance(self, cfg):
        return self.cla["front", cfg] / self.total(cfg)

    @property
    def package(self):
        return self.total("aero") - self.total("noaero")

    @property
    def package_balance(self):
        return (self.cla["front", "aero"] - self.cla["front", "noaero"]) / self.package


def downforce(data: pd.DataFrame, mass: float, cg_height: float, wheelbase: float) -> Downforce:
    d = data[data.v > 3].iloc[::2].reset_index(drop=True)       # 12.5 Hz
    run_names = sorted(d.run.unique())
    offsets = np.column_stack([(d.run == r).to_numpy(float) for r in run_names])

    def fit(axle, sel, v2_terms):
        q, F = d[sel], offsets[sel.to_numpy()]
        cols = [F, q.ax_s.to_numpy()[:, None], (q.ay_s ** 2).to_numpy()[:, None]]
        for cfg in v2_terms:
            cols.append((q.v ** 2 * (q.cfg == cfg)).to_numpy()[:, None])
        X = np.hstack(cols)
        keep = X.any(0)
        X = X[:, keep]
        c, *_ = np.linalg.lstsq(X, q[axle].to_numpy(), rcond=None)
        r = q[axle].to_numpy() - X @ c
        se = np.sqrt(np.diag(np.linalg.pinv(X.T @ X) * r.var() * 10))
        names = [*np.array(run_names)[keep[:len(run_names)]], "ax", "ay2", *v2_terms]
        return dict(zip(names, c)), dict(zip(names, se))

    transfer = mass * cg_height / wheelbase                       # N per m/s^2
    cal = (d.cfg == "noaero") & (d.v < 9)
    steady = (d.ax_s.abs() < 1.5) & (d.ay_s.abs() < 3)
    cpn, cla, cla_se = {}, {}, {}
    for axle, sign in [("front", -1), ("rear", +1)]:              # braking (ax < 0) loads the front
        c, _ = fit(axle, cal, [])
        cpn[axle] = c["ax"] / (sign * transfer)
        c, se = fit(axle, steady, ["aero", "noaero"])
        for cfg in ("aero", "noaero"):
            cla[axle, cfg] = 2 * c[cfg] / cpn[axle] / RHO
            cla_se[axle, cfg] = abs(2 * se[cfg] / cpn[axle] / RHO)
    return Downforce(cpn, cla, cla_se, d[steady])


def run(out: str = "aero") -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from kmslap.car.params import Car

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    car = Car()
    data = pd.concat([load(n) for n in runs()], ignore_index=True)
    drag = coastdown(data, car.mass)
    down = downforce(data, car.mass, car.cg_height, car.wheelbase)

    summary = {
        "CdA with aero (m^2)": f"{drag.cda['aero']:.2f} +- {drag.cda_se['aero']:.2f}",
        "CdA without aero": f"{drag.cda['noaero']:.2f} +- {drag.cda_se['noaero']:.2f}",
        "package CdA": f"{drag.package_cda:.2f} +- {drag.package_cda_se:.2f}",
        "rolling + driveline F0 (N)": f"{drag.f0:.0f}",
        "ClA with aero (m^2)": f"{down.total('aero'):.2f} (front {down.cla['front', 'aero']:.2f}, rear {down.cla['rear', 'aero']:.2f})",
        "ClA without aero": f"{down.total('noaero'):.2f}",
        "package ClA": f"{down.package:.2f}",
        "aero balance with aero (front)": f"{down.balance('aero'):.2f}",
        "package balance (front)": f"{down.package_balance:.2f}",
        "package L/D": f"{down.package / drag.package_cda:.1f}",
        "Car() now: CdA / ClA / front": f"{car.cda:.2f} / {car.cla:.2f} / {car.aero_front:.2f}",
        "coast-down samples (aero / none)": f"{(drag.samples.cfg == 'aero').sum()} / {(drag.samples.cfg == 'noaero').sum()}",
        "speed range (m/s)": f"{data.v[data.v > 3].min():.0f}-{data.v.max():.0f}",
    }
    pd.Series(summary).to_csv(out / "aero_summary.csv", header=["value"])
    for k, v in summary.items():
        print(f"{k:36s} {v}")

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    colors = {"aero": "#e6504a", "noaero": "#468ce6"}
    c = drag.samples
    vv = np.linspace(6, 14, 50)
    for cfg in ("aero", "noaero"):
        q = c[c.cfg == cfg]
        ax[0].plot(q.v, -q.a_flat, ".", ms=2, color=colors[cfg], alpha=0.4)
        ax[0].plot(vv, (drag.f0 + 0.5 * RHO * drag.cda[cfg] * vv ** 2) / car.mass, color=colors[cfg],
                   label=f"{'with' if cfg == 'aero' else 'without'} aero: CdA {drag.cda[cfg]:.2f}")
    ax[0].set(xlabel="m/s", ylabel="deceleration on flat ground (m/s^2)", title="Coast-downs")
    ax[0].legend()
    s = down.samples
    for i, axle in enumerate(["front", "rear"]):
        a = ax[i + 1]
        for cfg in ("aero", "noaero"):
            q = s[s.cfg == cfg]
            # remove each run's offset and the acceleration terms for display: bin by v^2
            load_n = (q[axle] - q.groupby("run")[axle].transform(lambda x: x[q.loc[x.index, "v"] < 6].median())) \
                / down.counts_per_n[axle]
            bins = pd.cut(q.v ** 2, np.arange(0, 400, 20))
            m = load_n.groupby(bins, observed=True).median()
            centers = [b.mid for b in m.index]
            a.plot(centers, m.values, "o", color=colors[cfg])
            a.plot([0, 380], [0, 0.5 * RHO * down.cla[axle, cfg] * 380], color=colors[cfg],
                   label=f"{'with' if cfg == 'aero' else 'without'} aero: ClA {down.cla[axle, cfg]:.2f}")
        a.set(xlabel="v^2 (m^2/s^2)", ylabel="axle load vs at < 6 m/s (N)",
              title=f"{axle} axle from shock pots ({down.counts_per_n[axle]:.2f} counts/N)")
        a.legend(fontsize=8)
    for a in ax:
        a.grid(alpha=0.3)
    fig.suptitle("KS7e aero tests, 6 July 2024 (reported; Car() unchanged)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "aero.png", dpi=110)
    plt.close(fig)
    return {"drag": drag, "downforce": down, "summary": summary}
