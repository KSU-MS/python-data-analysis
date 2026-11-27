"""Run the headline sims and write viewable artifacts plus a markdown summary.

    accel report --out report/

Writes into --out:
    drag_race.gif        no TC vs the VCU controllers racing the 75 m, rear wheels
                         colored by slip, speed and slip traces drawing in
    launch.png           first second: torque, slip, accel
    fd_sweep.png         accel time vs final drive (ideal TC, no TC, VCU)
    robustness.png       Monte Carlo: % over ideal with every coefficient wrong
    controller_view.png  what the C++ controller saw vs the truth
    sysid.png            log -> fit of a hidden car: tire force model vs measured
    gear_ratio_check.png controller set up for the wrong final drive, with and
                         without its gear ratio plausibility check
    sensitivity.png      seconds of accel time each uncertain coefficient is worth
                         (exact gradients through the CasADi model)
    summary.md           tables + Mermaid charts (GitHub renders these inline; CI
                         appends this to the job summary)
    metrics.json         headline numbers; `accel report-diff` compares
                         two of these (CI posts that on pull requests)
"""
import dataclasses
import json
import math
import tempfile
import textwrap
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from kmslap.models import accel1d as model  # noqa: E402
from kmslap.vcu.firmware import FAULTS, FirmwareController, Sensors, read_log, replay, write_log  # noqa: E402
from kmslap.car.params import G, Car  # noqa: E402
from kmslap.car.powertrain import PACKS, MotorMap, Pack  # noqa: E402
from kmslap.events.accel import LENGTH, STAGE, quasi_static, simulate  # noqa: E402
from kmslap.vcu.sysid import fit_plant  # noqa: E402
from kmslap.car.tire import default_tire  # noqa: E402
from kmslap.vcu.uncertainty import ABSOLUTE, RANGES, monte_carlo, sample_plants  # noqa: E402

# One plot style for every figure in the report. Categorical colors in a fixed order
# (blue, orange, aqua, yellow; checked for colorblind separation), ink and hairline
# greys for text, truth/reference lines and the grid.
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# a run keeps its color in every plot (and the GIF); anything else takes the next free one
RUN_COLORS = {"VCU mu_adaptive": COLORS[0], "VCU": COLORS[0], "no TC": COLORS[1],
              "VCU slip_pi": COLORS[2], "VCU, no ratio check": COLORS[2]}
plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 150, "figure.facecolor": "white",
    "font.size": 10, "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlecolor": INK, "axes.labelsize": 10, "axes.labelcolor": INK2,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "axes.prop_cycle": plt.cycler(color=COLORS), "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 1.8,
    "legend.fontsize": 9, "legend.frameon": False,
})


def _color(name, i=0):
    return RUN_COLORS.get(name, COLORS[i % len(COLORS)])


def _finish(fig, path, title, caption):
    """Title and a short "what to read from it" caption above the axes, then save."""
    w, h = fig.get_size_inches()
    caption = textwrap.fill(caption, width=int(w * 13.5))
    lines = caption.count("\n") + 1
    fig.text(0.012, 1 - 0.12 / h, title, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.012, 1 - 0.42 / h, caption, ha="left", va="top", fontsize=9.5, color=INK2,
             linespacing=1.35)
    fig.tight_layout(rect=(0, 0, 1, 1 - (0.5 + 0.19 * lines) / h))
    fig.savefig(path)
    plt.close(fig)


def _slip_color(slip):
    """Green at no slip through yellow to red at 0.3+."""
    k = min(max(slip / 0.3, 0.0), 1.0)
    return (min(1.0, 2 * k), min(1.0, 2 * (1 - k)) * 0.8, 0.2)


def drag_race_gif(runs: dict, car: Car, path, fps: int = 20, width_px: int = 960):
    """Animate runs (name -> Result) down the strip, in real time."""
    names = list(runs)
    finish = {n: r.accel_time for n, r in runs.items()}
    t_end = max(r.df["t"].iloc[-1] for r in runs.values()) + 0.4
    frames = np.arange(0.0, t_end, 1.0 / fps)
    data, full = {}, {}
    for n, r in runs.items():
        d = r.df
        data[n] = {k: np.interp(frames, d["t"], d[k], right=d[k].iloc[-1])
                   for k in ("x", "v", "slip", "motor_torque")}
        # traces draw from the full-rate sim (every 5 ms), not the frames, so fast
        # slip oscillations aren't aliased
        sub = d.iloc[::25]
        full[n] = (sub["t"].to_numpy(), sub["v"].to_numpy() * 3.6,
                   np.clip(sub["slip"].to_numpy(), -0.05, 0.6))

    fig = plt.figure(figsize=(width_px / 100, width_px / 100 * 0.62))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1], hspace=0.35, wspace=0.22,
                          left=0.16, right=0.97, top=0.9, bottom=0.08)
    track = fig.add_subplot(gs[0, :])
    speed = fig.add_subplot(gs[1, 0])
    slip_ax = fig.add_subplot(gs[1, 1])

    lane_h, car_len, car_w = 1.0, 2.8, 0.55
    track.set_xlim(-2, STAGE + LENGTH + 3)
    track.set_ylim(-0.6, lane_h * len(names) - 0.4)
    track.set_yticks([lane_h * i for i in range(len(names))], names[::-1])
    track.set_xlabel("distance (m)")
    track.grid(False)
    for x, label in ((STAGE, "start"), (STAGE + LENGTH, "finish")):
        track.axvline(x, color="k", lw=2)
        track.text(x, lane_h * len(names) - 0.45, label, ha="center", va="bottom", fontsize=8)
    for i in range(len(names) + 1):
        track.axhline(lane_h * i - 0.5, color="0.7", lw=0.8)

    cars, wheels, labels = {}, {}, {}
    for i, n in enumerate(names):
        y = lane_h * (len(names) - 1 - i)
        body = Rectangle((0, y - car_w / 2), car_len, car_w, color=_color(n, i), zorder=3)
        track.add_patch(body)
        # markers (sized in points) so wheels stay round on the unequal-scale axes
        rear = track.plot([], [], "o", ms=10, mec="k", zorder=4)[0]
        front = track.plot([], [], "o", ms=10, color="0.2", zorder=4)[0]
        cars[n], wheels[n] = body, (rear, front)
        labels[n] = track.text(0, y + car_w / 2 + 0.05, "", fontsize=8, va="bottom")

    lines = {}
    for i, n in enumerate(names):
        c = _color(n, i)
        lines[n] = (speed.plot([], [], color=c, label=n)[0], slip_ax.plot([], [], color=c)[0])
    speed.set(xlim=(0, t_end), ylim=(0, max(d["v"].max() for d in data.values()) * 3.6 * 1.08),
              xlabel="time (s)", ylabel="speed (km/h)")
    slip_ax.set(xlim=(0, 1.5), ylim=(-0.05, 0.6), xlabel="time (s)", ylabel="rear slip ratio (first 1.5 s)")
    speed.legend(fontsize=8, loc="lower right")
    cursor = [ax.axvline(0, color="k", lw=0.8) for ax in (speed, slip_ax)]
    title = fig.suptitle("")

    def draw(k):
        t = frames[k]
        for i, n in enumerate(names):
            d = data[n]
            x, s = d["x"][k], d["slip"][k]
            y = lane_h * (len(names) - 1 - i)
            cars[n].set_x(x - car_len + 0.2)
            wheels[n][0].set_data([x - car_len + 0.6], [y - car_w / 2])
            wheels[n][0].set_markerfacecolor(_slip_color(s))
            wheels[n][1].set_data([x - 0.2], [y - car_w / 2])
            done = t >= finish[n] + 0.0 and math.isfinite(finish[n])
            labels[n].set_position((x - car_len + 0.2, y + car_w / 2 + 0.05))
            labels[n].set_text(f"{finish[n]:.3f} s" if done else
                               f"{d['v'][k] * 3.6:5.1f} km/h  slip {s:4.2f}")
            ft, fv, fs = full[n]
            upto = ft <= t
            lines[n][0].set_data(ft[upto], fv[upto])
            lines[n][1].set_data(ft[upto], fs[upto])
        for c in cursor:
            c.set_xdata([t, t])
        title.set_text(f"75 m accel, current car (228HV, FD {car.final_drive:g})   t = {t:4.2f} s")
        return []

    anim = FuncAnimation(fig, draw, frames=len(frames), interval=1000 / fps)
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def _fig_launch(runs, path):
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True)
    for i, (n, r) in enumerate(runs.items()):
        d = r.df.query("t < 1.0")
        axes[0].plot(d.t, d.motor_torque, color=_color(n, i), label=n)
        axes[1].plot(d.t, d.slip.clip(-0.1, 1.0), color=_color(n, i))
        axes[2].plot(d.t, d.a / G, color=_color(n, i))
    for ax, title, y in zip(axes, ["Torque command", "Rear wheel slip", "Longitudinal acceleration"],
                            ["motor torque (Nm)", "slip ratio (-)", "accel (g)"]):
        ax.set_title(title)
        ax.set_ylabel(y)
    axes[0].legend(loc="lower right", ncols=len(runs))
    axes[-1].set_xlabel("time since launch (s)")
    _finish(fig, path, "Launch: the first second",
            "Traction control trims torque so rear slip settles to a small value instead of running "
            "away into wheelspin (no TC, clipped at 1); holding more g here is most of the 75 m time.")


def _fd_sweep(car, motor, fds):
    rows = []
    for fd in fds:
        c = dataclasses.replace(car, final_drive=fd)
        rows.append((fd, quasi_static(c, motor).accel_time, simulate(c, motor).accel_time,
                     # the controller must be configured with the car's final drive: with
                     # the wrong ratio its rear wheel speed, and so its slip, is garbage
                     simulate(c, motor, controller=FirmwareController(mode="mu_adaptive",
                                                                      belief=c)).accel_time))
    return np.array(rows)


def _fig_fd(sweep, current_fd, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    sweep = np.where(np.isfinite(sweep), sweep, np.nan)   # a run that didn't finish: a gap
    for j, (label, c) in enumerate([("ideal TC (quasi-static)", INK), ("no TC", _color("no TC")),
                                    ("VCU mu_adaptive", _color("VCU mu_adaptive"))], start=1):
        ax.plot(sweep[:, 0], sweep[:, j], "o-", color=c, ms=6, label=label,
                ls="--" if j == 1 else "-", lw=1.4 if j == 1 else 1.8)
    ax.axvline(current_fd, color=MUTED, lw=1, zorder=0)
    ax.annotate(f"current car (FD {current_fd:g})", (current_fd, 1), xycoords=("data", "axes fraction"),
                xytext=(4, -4), textcoords="offset points", va="top", fontsize=8.5, color=INK2)
    ax.set(xlabel="final drive ratio (motor rev / wheel rev)", ylabel="75 m accel time (s)")
    ax.margins(y=0.12)
    ax.legend(loc="center right")
    _finish(fig, path, "Accel time vs final drive",
            "Lower is faster. The dashed ideal-TC line is the best any controller could do at each "
            "ratio; the gap from VCU to ideal is what traction control still leaves on the table.")


def _fig_robustness(mc, cols, path):
    fig, ax = plt.subplots(figsize=(8, 1.6 + 0.9 * len(cols)))
    rng = np.random.default_rng(0)   # vertical jitter only, so dots don't stack
    for i, c in enumerate(cols):
        y = len(cols) - 1 - i
        ax.scatter(mc[c], y + rng.uniform(-0.18, 0.18, len(mc)), s=30, color=_color(c, i),
                   alpha=0.75, edgecolors="white", linewidths=0.8, zorder=3)
        mean = mc[c].mean()
        ax.plot([mean, mean], [y - 0.32, y + 0.32], color=INK, lw=2, zorder=4)
        ax.annotate(f"mean {mean:+.1f}%", (mean, y + 0.32), xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_yticks(range(len(cols)), cols[::-1])
    ax.set_ylim(-0.6, len(cols) - 0.25)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(left=min(0, mc[cols].min().min()) - 0.2)
    ax.set_xlabel("75 m time over that car's ideal-TC time (%)")
    _finish(fig, path, f"Robustness: {len(mc)} cars with every coefficient wrong",
            "Each dot is one randomly perturbed car (grip, mass, sensors...) driven by a controller "
            "tuned for the nominal car; the black tick is the mean. Good: dots bunched near 0%.")


def _fig_controller(run, path):
    d, log = run.df.query("t < 2"), run.controller_log.query("0 <= t < 2")
    fig, axes = plt.subplots(3, 1, figsize=(9, 8.5), sharex=True)
    axes[0].plot(d.t, d.v, color=INK, label="true speed")
    axes[0].plot(log.t, log.v_est, color=COLORS[0], label="v_est (VectorNav + wheels)")
    axes[0].plot(log.t, log.omega_fl * Car.tire_radius, color=COLORS[1], lw=1.2, label="front wheel sensor")
    axes[1].plot(d.t, d.slip.clip(-0.3, 1), color=INK, label="true slip")
    axes[1].plot(log.t, log.slip.clip(-0.3, 1), color=COLORS[0], label="measured slip")
    axes[2].plot(d.t, d.motor_torque, color=INK, label="motor torque")
    axes[2].plot(log.t, log.torque_ff, color=COLORS[2], label="map torque from learned mu")
    for ax, title, y in zip(axes, ["Vehicle speed", "Rear wheel slip", "Torque"],
                            ["speed (m/s)", "slip ratio (-)", "torque (Nm)"]):
        ax.set_title(title)
        ax.set_ylabel(y)
        ax.legend(loc="best")
    axes[-1].set_xlabel("time since launch (s)")
    _finish(fig, path, "What the VCU saw vs the truth (mu_adaptive)",
            "Black is the simulated truth, color is the controller's own estimate from its sensors. "
            "Good: the estimates track the black lines closely and settle quickly after launch.")


def _fig_sysid(fit, path):
    s = fit.signals
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(s.t, s.fx_meas, color=MUTED, lw=1, label="from measured accel (noisy)")
    ax.plot(s.t, s.fx_model, color=COLORS[0], label="fitted tire model")
    ax.set(xlabel="time since launch (s)", ylabel="rear tire force Fx (N)", xlim=(0, 1.5))
    ax.legend(loc="lower right")
    _finish(fig, path, "System ID: rear tire force, measured vs fitted",
            "Tire force backed out of a logged run on a hidden car (grey) and the model fitted to "
            "that log (blue). Good: blue runs through the middle of the grey, including the peak.")


def _fig_gear_ratio(runs, true_fd, belief_fd, path):
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for i, (n, r) in enumerate(runs.items()):
        d = r.df.query("t < 4")
        axes[0].plot(d.t, d.v, color=_color(n, i), label=n)
        if r.controller_log is None:
            continue
        log = r.controller_log.query("0 <= t < 4")
        m = log[log.gear_ratio > 0]
        axes[1].step(m.t, m.gear_ratio, where="post", color=_color(n, i), label=n)
        f = log[log.faults > 0]
        if len(f):
            for ax in axes:
                ax.axvline(f.t.iloc[0], color=_color(n, i), ls=":", lw=1.4)
            axes[0].annotate("fault latched, TC off", (f.t.iloc[0], 0), xycoords=("data", "axes fraction"),
                             xytext=(4, 4), textcoords="offset points", fontsize=8.5, color=INK2)
    axes[1].axhline(belief_fd, color=INK, lw=1, label=f"configured FD {belief_fd:g}")
    axes[1].axhline(true_fd, color=MUTED, lw=1, ls="--", label=f"actual FD {true_fd:g}")
    axes[0].set_title("Vehicle speed")
    axes[1].set_title("Gear ratio the VCU measures (wheel teeth vs motor revs)")
    axes[0].set_ylabel("speed (m/s)")
    axes[1].set_ylabel("ratio (motor rev / wheel rev)")
    axes[0].legend(loc="upper left")
    axes[1].legend(loc="best", ncols=2)
    axes[-1].set_xlabel("time since launch (s)")
    _finish(fig, path, f"Plausibility check: VCU set up for FD {belief_fd:g}, car geared {true_fd:g}",
            "With the wrong ratio the controller's slip estimate is garbage and it chokes the torque. "
            "Good: the measured ratio disagrees early, the fault (dotted line) latches, and the car "
            "accelerates again; without the check it never recovers.")


def _sensitivity(car, motor):
    """(accel time, rows of (coefficient, nominal, range, dT/dp, seconds across range))
    for an open-loop driver launch, ranked by seconds."""
    tire = default_tire()
    accel_time, grad = model.accel_time_sensitivity(car, motor, Pack(PACKS["cosmx600v"]), tire)
    nominal = dict(zip(model.PARAMS, model.params(car, motor, Pack(PACKS["cosmx600v"]), tire,
                                                  0.15, 2e-4)))
    rows = []
    for k, (lo, hi) in RANGES.items():
        p = "tire_scale" if k == "grip" else k
        if p not in grad:
            continue   # sensor errors aren't model parameters
        span = (hi - lo) if k in ABSOLUTE else (hi - lo) * nominal[p]
        rows.append((k, nominal[p], (lo, hi), grad[p], abs(grad[p]) * span))
    return accel_time, sorted(rows, key=lambda r: -r[4])


def _fig_sensitivity(rows, path):
    fig, ax = plt.subplots(figsize=(8, 1.8 + 0.38 * len(rows)))
    names = [r[0] for r in rows][::-1]
    ms = [r[4] * 1000 for r in rows][::-1]
    bars = ax.barh(names, ms, height=0.62, color=COLORS[0])
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8.5, color=INK2)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(0, max(ms + [1]) * 1.12)
    ax.set_xlabel("spread in 75 m time across the coefficient's uncertainty range (ms)")
    _finish(fig, path, "What to measure next: driver launch, no TC",
            "How much accel time each uncertain coefficient could swing, from exact gradients "
            "through the CasADi model. The longest bars are the measurements worth making first.")


def _mermaid_lines(title, x, series, y_label):
    if not all(np.isfinite(v).all() for v in series.values()):
        return ["_(chart skipped: a run didn't finish the 75 m; see fd_sweep.png)_"]
    lo = min(min(v) for v in series.values())
    hi = max(max(v) for v in series.values())
    pad = (hi - lo) * 0.1 or 0.1
    out = ["```mermaid", "xychart-beta", f'  title "{title}"',
           f"  x-axis [{', '.join(f'{v:g}' for v in x)}]",
           f'  y-axis "{y_label}" {lo - pad:.2f} --> {hi + pad:.2f}']
    out += [f"  line [{', '.join(f'{v:.3f}' for v in vals)}]" for vals in series.values()]
    out.append("```")
    return out


def _mermaid_bars(title, labels, values, y_label):
    names = ", ".join(f'"{v}"' for v in labels)
    return ["```mermaid", "xychart-beta", f'  title "{title}"', f"  x-axis [{names}]",
            f'  y-axis "{y_label}" 0 --> {max(values + [1]) * 1.1:.1f}',
            f"  bar [{', '.join(f'{v:.1f}' for v in values)}]", "```"]


def run(out: str = "report", plants: int = 20):
    """Write the report into `out`; plants: Monte Carlo plants."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    # accel: wing set open (lowest drag), and the controller configured to match
    car, motor = Car().open_aero(), MotorMap("228HV")
    md = ["# Accel sim report", "",
          "Deterministic sims of the 75 m acceleration event with the VCU's C++ traction control "
          "in the loop, run on every push. Lower times are better throughout; on pull requests "
          "the run summary also compares these numbers against the base branch.", "",
          "> Plots (`*.png`) and the drag race animation (`drag_race.gif`) are in the "
          "**`accel-report`** artifact at the bottom of this run's page; the job summary can't "
          "show images inline. Each section below names the plot that goes with it.", ""]
    metrics = {}

    def metric(key, value, unit, better="lower"):
        metrics[key] = {"value": None if not np.isfinite(value) else float(value),
                        "unit": unit, "better": better}

    # headline runs + GIF
    ideal = quasi_static(car, motor)
    runs = {
        "no TC": simulate(car, motor),
        "VCU slip_pi": simulate(car, motor, controller=FirmwareController(mode="slip_pi", belief=car)),
        "VCU mu_adaptive": simulate(car, motor, controller=FirmwareController(mode="mu_adaptive",
                                                                              belief=car)),
    }
    md += ["## Headline: 75 m accel", "",
           f"Current car: EMRAX 228HV, COSMX 144s1p, FD {car.final_drive:g}, LC0 tire, wing open. "
           "Ideal TC is a quasi-static bound (perfect traction control: the tire at its peak "
           "every instant); the other rows are full sims. Good: VCU rows within a percent or "
           "two of ideal, and peak slip ratio far below the wheelspin no TC shows.", "",
           "| run | 75 m time (s) | vs ideal (%) | trap speed (km/h) | peak slip ratio (-) |",
           "|---|--:|--:|--:|--:|",
           f"| ideal TC (quasi-static) | {ideal.accel_time:.3f} | | "
           f"{ideal.df.v.iloc[-1] * 3.6:.1f} | |"]
    metric("accel time: ideal TC", ideal.accel_time, "s")
    for n, r in runs.items():
        s = r.summary()
        metric(f"accel time: {n}", r.accel_time, "s")
        metric(f"peak slip: {n}", s["max_slip"], "")
        md.append(f"| {n} | {r.accel_time:.3f} | {100 * (r.accel_time / ideal.accel_time - 1):+.2f} | "
                  f"{s['trap_kmh']:.1f} | {s['max_slip']:.2f} |")
    md += ["", "Plots: `launch.png` (torque, slip and g over the first second), "
               "`controller_view.png` (the VCU's estimates vs the truth), `drag_race.gif`."]
    drag_race_gif(runs, car, out / "drag_race.gif")
    _fig_launch(runs, out / "launch.png")
    _fig_controller(runs["VCU mu_adaptive"], out / "controller_view.png")

    # final drive sweep
    sweep = _fd_sweep(car, motor, sorted({2.4, 2.7, 3.0, 3.5, 4.0, 4.5, 5.0, car.final_drive}))
    _fig_fd(sweep, car.final_drive, out / "fd_sweep.png")
    for fd, *_, vcu_time in sweep:
        metric(f"accel time: VCU mu_adaptive, FD {fd:g}", vcu_time, "s")
    series = {"no TC": sweep[:, 2], "VCU mu_adaptive": sweep[:, 3], "ideal TC": sweep[:, 1]}
    # Mermaid charts have no legend: name the lines in the order they sit at the current ratio
    k = int(np.argmin(abs(sweep[:, 0] - car.final_drive)))
    order = sorted(series, key=lambda n: -series[n][k] if np.isfinite(series[n][k]) else -math.inf)
    md += ["", "## Final drive", "",
           "75 m time for each final drive ratio, with the VCU configured for the ratio it's run "
           "at. A higher ratio launches harder but hits the motor's speed limit sooner. Good: "
           "the VCU line hugging ideal TC, and the current ratio near the bottom of the curve.", "",
           f"Lines, top to bottom at FD {sweep[k, 0]:g}: {', '.join(order)}. Plot: `fd_sweep.png`.", ""]
    md += _mermaid_lines("75 m time (s) vs final drive ratio", sweep[:, 0], series, "time (s)")
    md += ["", "<details><summary>Final drive sweep as a table</summary>", "",
           "| final drive | ideal TC (s) | no TC (s) | VCU mu_adaptive (s) |", "|--:|--:|--:|--:|"]
    md += [f"| {fd:g} | " + " | ".join(f"{v:.3f}" if np.isfinite(v) else "didn't finish" for v in row) + " |"
           for fd, *row in sweep]
    md += ["", "</details>"]

    # robustness
    fw = lambda mode: (lambda pl: dict(controller=FirmwareController(
        mode=mode, belief=car, sensors=Sensors(vn_accel_bias=pl.vn_accel_bias))))
    strategies = {"no TC": lambda pl: {}, "VCU slip_pi": fw("slip_pi"), "VCU mu_adaptive": fw("mu_adaptive")}
    mc = monte_carlo(strategies, sample_plants(plants, car), motor)
    cols = list(strategies)
    _fig_robustness(mc, cols, out / "robustness.png")
    md += ["", f"## Robustness ({plants} plants, every coefficient off)", "",
           f'{plants} random cars ("plants"), each with grip, mass, inertias, aero, sensor bias and '
           "the rest drawn from their uncertainty ranges, driven by controllers tuned for the "
           "nominal car. Numbers are % slower than that car's own ideal-TC time. Good: small mean "
           "and, more importantly, a small worst case. Plot: `robustness.png`.", "",
           "| controller | mean (% over ideal) | p90 (% over ideal) | worst (% over ideal) |",
           "|---|--:|--:|--:|"]
    for c in cols:
        md.append(f"| {c} | {mc[c].mean():+.2f} | {mc[c].quantile(0.9):+.2f} | {mc[c].max():+.2f} |")
        metric(f"robustness mean: {c}", mc[c].mean(), "% over ideal")
        metric(f"robustness worst: {c}", mc[c].max(), "% over ideal")

    # plausibility: controller configured with the wrong final drive (the two values in
    # circulation: the VCU's 4.18, kmslap's old 3.0; which the car has isn't measured)
    true_fd, belief_fd = car.final_drive, 3.0
    wrong = car
    belief = dataclasses.replace(car, final_drive=belief_fd)
    ratio_runs = {
        "no TC": simulate(wrong, motor),
        "VCU, no ratio check": simulate(wrong, motor, controller=FirmwareController(
            mode="mu_adaptive", belief=belief, overrides=dict(ratio_tolerance=1e9))),
        "VCU": simulate(wrong, motor, controller=FirmwareController(mode="mu_adaptive", belief=belief)),
    }
    _fig_gear_ratio(ratio_runs, true_fd, belief_fd, out / "gear_ratio_check.png")
    log = ratio_runs["VCU"].controller_log
    faulted = log[(log.faults & FAULTS["gear_ratio"]) > 0]
    d = ratio_runs["VCU"].df
    fault_x = d.x[d.t <= faulted.t.iloc[0]].iloc[-1] if len(faulted) else math.inf
    fmt = lambda r: f"{r.accel_time:.3f}" if np.isfinite(r.accel_time) else "didn't finish"
    md += ["", f"## Plausibility: VCU set up for FD {belief_fd:g}, car geared {true_fd:g}", "",
           "The controller counts rear wheel sensor teeth against motor revs; when the ratio "
           "disagrees with its configured final drive it latches a fault and passes the "
           "driver's torque straight through. Here it's deliberately misconfigured. Good: the "
           "fault latches within the first few metres so the VCU run still finishes (slower than "
           "no TC, having fought a wrong slip estimate until then), while the unchecked controller "
           "is much slower or doesn't finish. "
           "Plot: `gear_ratio_check.png`.", "",
           "| run | 75 m time (s) |", "|---|--:|"]
    md += [f"| {n} | {fmt(r)} |" for n, r in ratio_runs.items()]
    md += ["", f"Fault latched after **{fault_x:.1f} m** (measured ratio "
               f"{log.gear_ratio[log.gear_ratio > 0].iloc[-1]:.2f})." if len(faulted)
           else "**No fault latched.**"]
    metric("gear ratio fault: distance to detect", fault_x, "m")
    metric("gear ratio fault: accel time", ratio_runs["VCU"].accel_time, "s")

    # sensitivity: exact gradients through the CasADi model
    t_open, sens = _sensitivity(car, motor)
    _fig_sensitivity(sens, out / "sensitivity.png")
    md += ["", "## What to measure next", "",
           f"Driver launch with no TC ({t_open:.3f} s): exact d(accel time)/d(coefficient) "
           "through the CasADi model, times each coefficient's uncertainty range "
           "(`uncertainty.RANGES`). Biggest first: the ones worth measuring, since pinning them "
           "down shrinks the model's error bars the most. Plot: `sensitivity.png`.", ""]
    md += _mermaid_bars("Spread in 75 m time across each range (ms)", [r[0] for r in sens],
                        [1000 * r[4] for r in sens], "ms")
    md += ["", "| coefficient | nominal | range | d time / d coeff (s per unit) | across range (ms) |",
           "|---|--:|---|--:|--:|"]
    for k, nom, (lo, hi), g, spread in sens:
        rng = f"{lo:g} to {hi:g}" + ("" if k in ABSOLUTE else " x nominal")
        md.append(f"| {k} | {nom:.4g} | {rng} | {g:+.4g} | {1000 * spread:.1f} |")
        metric(f"sensitivity across range: {k}", 1000 * spread, "ms", better="neither")

    # log -> replay -> fit loop on a hidden car
    real = sample_plants(1, car, seed=123)[0]
    vcu = FirmwareController(mode="slip_pi", belief=car, sensors=Sensors(vn_accel_bias=real.vn_accel_bias))
    simulate(real.car, motor, tire=real.tire, relaxation_length=real.relaxation_length, controller=vcu)
    with tempfile.TemporaryDirectory() as tmp:
        write_log(Path(tmp) / "run.vcu", vcu.records())
        inputs, logged = read_log(Path(tmp) / "run.vcu")
    again = replay(inputs, vcu.config(motor), t_offset=vcu.sensors.staged_time)
    identical = bool((again["torque_cmd"].to_numpy(np.float32) == logged["torque_cmd_nm"].to_numpy()).all())
    fit = fit_plant(inputs, car, motor)
    _fig_sysid(fit, out / "sysid.png")
    narrowed = fit.ranges(RANGES)
    md += ["", "## Log replay and system ID (hidden car)", "",
           "A run on a randomly drawn car is logged in the VCU's binary format, replayed through "
           "the controller (it must reproduce every torque command bit for bit), then fitted to "
           "recover the car. Good: **identical** replay, fit close to truth, and the narrowed "
           "range well inside the prior. Plot: `sysid.png`.", "",
           f"Binary log ({len(inputs)} records) replayed through the controller: "
           f"**{'identical' if identical else 'DIFFERENT'}** torque commands.", "",
           "| coefficient | unit | truth | fit | error (%) | prior range | narrowed |",
           "|---|---|--:|--:|--:|---|---|"]
    for k in ("grip", "mass", "tire_radius", "relaxation_length"):
        lo, hi = narrowed[k]
        unit = "m" if k == "relaxation_length" else "x nominal"
        md.append(f"| {k} | {unit} | {real.draw[k]:.4f} | {fit.params[k]:.4f} | "
                  f"{100 * abs(fit.params[k] / real.draw[k] - 1):.2f} | "
                  f"{RANGES[k][0]:g} to {RANGES[k][1]:g} | {lo:.4f} to {hi:.4f} |")
        metric(f"sysid error: {k}", 100 * abs(fit.params[k] / real.draw[k] - 1), "%")

    md += ["", f"_Generated in {time.perf_counter() - t0:.0f} s; headline numbers in `metrics.json`._", ""]
    metric("log replay identical", float(identical), "", better="higher")
    (out / "summary.md").write_text("\n".join(md))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print("\n".join(md))
    if not identical:
        raise SystemExit("log replay did not reproduce the controller's outputs")

