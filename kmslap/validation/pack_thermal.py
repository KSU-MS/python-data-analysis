"""KS6e accumulator thermal tests: the cell temperatures as a top-down heatmap over the
run, and the lumped pack thermal model the sim uses fitted to them.

    kmslap pack-temps --out pack/      # GIFs, summary plot, fitted coefficients

The tests (cardata/ks6e/acc-test-{no-fans,with-fans}-data/temps_raw.csv): the KS6e
pack (Energus, 72s8p LG HE2) driven ~20 minutes at a few tens of amps, with every
series group's thermistor logged, without and with the fans. The current car's pack
(COSMX, no fans) is different, but "no fans" is the condition its model assumes.

Processing (from legacy/acc_heatmap/manipulate_df.py, checked against the converted
temps_real.csv to 0.0 K): the BMS's 5th-order calibration polynomial takes 8-bit ADC
counts (0.019607 V per count); readings logged in volts are converted to counts first; each signal is sparse (its own CAN message), so it's
interpolated onto a 1 s grid. The physical layout, looking down on the pack, is
legacy/cellarraycsv.csv (LAYOUT below).

Model (models/twotrack.py's pack): C dT/dt = I^2 R - hA (T - T_ambient), T the mean
cell temperature, ambient the starting temperature. Fitted per test: the heating
coefficient R/C and the cooling coefficient hA/C; with the pack's thermal mass C (an
estimate) they give R and hA. hA needs a cool-down to be identified: the no-fans test
has ~6 minutes of one, the with-fans test none, so its hA isn't meaningful.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from kmslap import dirs

TESTS = {
    "no-fans": dirs.REPO / "cardata" / "ks6e" / "acc-test-no-fans-data" / "temps_raw.csv",
    "with-fans": dirs.REPO / "cardata" / "ks6e" / "acc-test-with-fans-data" / "temps_raw.csv",
}
# series group (thermistor) numbers as they sit in the pack, looking down; columns are
# 12 .. 1 left to right (legacy/cellarraycsv.csv)
LAYOUT = np.array([
    [67, 66, 55, 54, 43, 42, 31, 30, 19, 18, 7, 6],
    [68, 65, 56, 53, 44, 41, 32, 29, 20, 17, 8, 5],
    [69, 64, 57, 52, 45, 40, 33, 28, 21, 16, 9, 4],
    [70, 63, 58, 51, 46, 39, 34, 27, 22, 15, 10, 3],
    [71, 62, 59, 50, 47, 38, 35, 26, 23, 14, 11, 2],
    [72, 61, 60, 49, 48, 37, 36, 25, 24, 13, 12, 1],
])
CELLS = [f"cell{i}temp" for i in range(1, 73)]
# KS6e pack: 576 LG HE2 cells (~47 g each, ~0.9 kJ/(kg K)) plus module structure; the
# thermal mass is an ESTIMATE used to turn the fitted coefficients into R and hA
KS6E_THERMAL_MASS = 576 * 0.047 * 900 * 1.15        # J/K
KS6E_PACK_R_DATASHEET = 0.0321 * 72 / 8             # ohm, battery.py's lgHE2 cell, 72s8p


def thermistor_c(v: np.ndarray) -> np.ndarray:
    """Thermistor reading (volts, or ADC counts if > 3) to deg C: the BMS's
    calibration polynomial, in ADC counts. Zero / missing stays NaN."""
    v = np.asarray(v, float).copy()
    v[v <= 0] = np.nan
    v = np.where(v > 3, v, v / 0.019607)   # volts -> counts
    return np.round(-0.000002416676401 * v ** 5 + 0.001082617446913 * v ** 4
                    - 0.194488265848684 * v ** 3 + 17.5197709028014 * v ** 2
                    - 792.865188960333 * v + 14494.8611005946, 4)


def load(test: str, rate_hz: float = 1.0) -> pd.DataFrame:
    """A test on a uniform clock: t (s), current (A), voltage (V), cell1temp..cell72temp
    (deg C), and t_mean / t_max / t_min across cells."""
    raw = pd.read_csv(TESTS[test], usecols=["Time", "Pack_Current", "Pack_Summed_Voltage", *CELLS])
    raw["t"] = (raw.Time - raw.Time.iloc[0]) / 1000.0
    for c in CELLS:
        raw[c] = thermistor_c(raw[c].to_numpy())
    grid = np.arange(0.0, raw.t.iloc[-1], 1.0 / rate_hz)
    out = {"t": grid}
    for col, name in [("Pack_Current", "current"), ("Pack_Summed_Voltage", "voltage"), *[(c, c) for c in CELLS]]:
        s = raw[["t", col]].dropna()
        out[name] = np.interp(grid, s.t, s[col]) if len(s) else np.full_like(grid, np.nan)
    d = pd.DataFrame(out)
    temps = d[CELLS]
    d["t_mean"], d["t_max"], d["t_min"] = temps.mean(1), temps.max(1), temps.min(1)
    return d


@dataclass
class Fit:
    test: str
    heat_per_c: float        # R / C, K per (A^2 s)
    cool_per_c: float        # hA / C, 1/s
    rms_k: float
    thermal_mass: float = KS6E_THERMAL_MASS

    @property
    def resistance(self) -> float:      # ohm
        return self.heat_per_c * self.thermal_mass

    @property
    def h_a(self) -> float:             # W/K
        return self.cool_per_c * self.thermal_mass

    @property
    def time_constant_min(self) -> float:
        return 1 / self.cool_per_c / 60 if self.cool_per_c > 0 else float("inf")


def simulate_lumped(d: pd.DataFrame, heat_per_c: float, cool_per_c: float) -> np.ndarray:
    t, i = d.t.to_numpy(), d.current.to_numpy()
    T = np.empty_like(t)
    T[0] = amb = d.t_mean.iloc[0]
    for k in range(1, len(t)):
        dt = t[k] - t[k - 1]
        T[k] = T[k - 1] + dt * (heat_per_c * i[k - 1] ** 2 - cool_per_c * (T[k - 1] - amb))
    return T


def fit(test: str, d: pd.DataFrame | None = None) -> Fit:
    from scipy.optimize import least_squares

    d = load(test) if d is None else d
    res = least_squares(lambda x: simulate_lumped(d, x[0] * 1e-6, x[1] * 1e-4) - d.t_mean.to_numpy(),
                        x0=[10.0, 5.0], bounds=([0, 0], [1e4, 1e3]))
    return Fit(test, res.x[0] * 1e-6, res.x[1] * 1e-4, float(np.sqrt(np.mean(res.fun ** 2))))


def cool_down_s(d: pd.DataFrame, idle_a: float = 5.0) -> float:
    """How long the test ends with the pack idle after having been loaded."""
    loaded = np.nonzero(d.current.to_numpy() > idle_a)[0]
    return float(d.t.iloc[-1] - d.t.iloc[loaded[-1]]) if len(loaded) else 0.0


def heatmap_gif(d: pd.DataFrame, path, title: str, frames: int = 120, fps: int = 12):
    """Top-down cell temperatures over the test, with current and min / mean / max
    temperature traces and a moving cursor."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    idx = np.linspace(0, len(d) - 1, frames).astype(int)
    vmin, vmax = np.floor(d[CELLS].min().min()), np.ceil(d[CELLS].max().max())
    grid = lambda k: d.iloc[k][[f"cell{n}temp" for n in LAYOUT.ravel()]].to_numpy(float).reshape(LAYOUT.shape)
    fig = plt.figure(figsize=(11, 6.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.3, 1])
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(grid(0), cmap="inferno", vmin=vmin, vmax=vmax)
    labels = [[ax.text(j, i, "", ha="center", va="center", fontsize=7) for j in range(12)] for i in range(6)]
    ax.set_xticks(range(12), [str(c) for c in range(12, 0, -1)])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, label="deg C", fraction=0.025)
    ax2 = fig.add_subplot(gs[1])
    tm = d.t / 60
    for col, c in [("t_max", "#e6504a"), ("t_mean", "k"), ("t_min", "#468ce6")]:
        ax2.plot(tm, d[col], color=c, label=col[2:])
    ax2.set(xlabel="min", ylabel="deg C")
    ax3 = ax2.twinx()
    ax3.plot(tm, d.current, color="0.7", lw=0.8, label="pack current")
    ax3.set_ylabel("A")
    ax2.legend(loc="upper left", fontsize=8)
    cursor = ax2.axvline(0, color="k", lw=1)
    ax2.grid(alpha=0.3)

    def draw(f):
        k = idx[f]
        g = grid(k)
        im.set_data(g)
        for i in range(6):
            for j in range(12):
                labels[i][j].set_text(f"{LAYOUT[i, j]}\n{g[i, j]:.0f}")
                labels[i][j].set_color("w" if g[i, j] < (vmin + vmax) / 2 else "k")
        cursor.set_xdata([tm.iloc[k]])
        ax.set_title(f"{title}: t = {tm.iloc[k]:4.1f} min, {d.current.iloc[k]:5.1f} A, "
                     f"cells {d.t_min.iloc[k]:.0f}-{d.t_max.iloc[k]:.0f} deg C\n"
                     "top-down; each cell: series group / deg C", fontsize=10)
        return [im, cursor]

    draw(0)
    fig.tight_layout()
    FuncAnimation(fig, draw, frames=len(idx)).save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def summary_plot(data: dict, fits: dict, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    for col, (name, d) in enumerate(data.items()):
        f = fits[name]
        rise = (d.iloc[-1][[f"cell{n}temp" for n in LAYOUT.ravel()]].to_numpy(float)
                - d.iloc[0][[f"cell{n}temp" for n in LAYOUT.ravel()]].to_numpy(float)).reshape(LAYOUT.shape)
        a = ax[0, col]
        im = a.imshow(rise, cmap="inferno")
        for i in range(6):
            for j in range(12):
                a.text(j, i, f"{rise[i, j]:.0f}", ha="center", va="center", fontsize=8,
                       color="w" if rise[i, j] < np.nanmean(rise) else "k")
        a.set_xticks(range(12), [str(c) for c in range(12, 0, -1)])
        a.set_yticks([])
        a.set_title(f"{name}: temperature rise over the test (K), top-down")
        fig.colorbar(im, ax=a, fraction=0.025)
        a = ax[1, col]
        a.plot(d.t / 60, d.t_mean, "k", label="measured mean cell")
        a.fill_between(d.t / 60, d.t_min, d.t_max, color="0.85", label="cell min-max")
        a.plot(d.t / 60, simulate_lumped(d, f.heat_per_c, f.cool_per_c), "--", color="#e6504a",
               label=f"lumped model fit (RMS {f.rms_k:.2f} K)")
        a.set(xlabel="min", ylabel="deg C",
              title=f"R ~ {f.resistance * 1e3:.0f} mOhm, hA ~ {f.h_a:.1f} W/K, tau {f.time_constant_min:.0f} min "
                    f"(C ~ {f.thermal_mass / 1e3:.0f} kJ/K)")
        a.legend(fontsize=8)
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def run(out: str = "pack", gif: bool = True) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    data = {name: load(name) for name in TESTS}
    fits = {name: fit(name, d) for name, d in data.items()}
    summary_plot(data, fits, out / "pack_thermal.png")
    if gif:
        for name, d in data.items():
            heatmap_gif(d, out / f"pack_{name}.gif", f"KS6e pack, {name}")
    rows = {name: {"duration_min": round(d.t.iloc[-1] / 60, 1), "mean_current_a": round(d.current.mean(), 1),
                   "rms_current_a": round(float(np.sqrt((d.current ** 2).mean())), 1),
                   "start_c": round(d.t_mean.iloc[0], 1), "end_mean_c": round(d.t_mean.iloc[-1], 1),
                   "end_max_c": round(d.t_max.iloc[-1], 1),
                   "spread_end_k": round(d.t_max.iloc[-1] - d.t_min.iloc[-1], 1),
                   "fit_R_mohm": round(f.resistance * 1e3, 1), "datasheet_R_mohm": round(KS6E_PACK_R_DATASHEET * 1e3, 1),
                   "fit_hA_w_per_k": round(f.h_a, 2), "time_constant_min": round(f.time_constant_min, 1),
                   # hA is only identifiable from cooling: without a cool-down it's barely constrained
                   "cool_down_min": round(cool_down_s(d) / 60, 1),
                   "fit_rms_k": round(f.rms_k, 2)}
            for (name, d), f in zip(data.items(), fits.values())}
    table = pd.DataFrame(rows)
    table.to_csv(out / "pack_thermal.csv")
    print(table.to_string())
    return rows
