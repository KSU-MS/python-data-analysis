"""Run the headline sims and write viewable artifacts plus a markdown summary.

    python -m accel.report --out report/

Writes into --out:
    drag_race.gif        no TC vs the VCU controllers racing the 75 m, rear wheels
                         colored by slip, speed and slip traces drawing in
    launch.png           first second: torque, slip, accel
    fd_sweep.png         accel time vs final drive (ideal TC, no TC, VCU)
    robustness.png       Monte Carlo: % over ideal with every coefficient wrong
    controller_view.png  what the C++ controller saw vs the truth
    sysid.png            log -> fit of a hidden car: tire force model vs measured
    summary.md           tables + Mermaid charts (GitHub renders these inline; CI
                         appends this to the job summary)
"""
import argparse
import math
import tempfile
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from .firmware import FirmwareController, Sensors, read_log, replay, write_log  # noqa: E402
from .params import G, Car  # noqa: E402
from .powertrain import MotorMap  # noqa: E402
from .sim import LENGTH, STAGE, quasi_static, simulate  # noqa: E402
from .sysid import fit_plant  # noqa: E402
from .uncertainty import RANGES, monte_carlo, sample_plants  # noqa: E402

COLORS = ["#e6504a", "#468ce6", "#5abe6e", "#e6aa32"]
plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3})


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
        body = Rectangle((0, y - car_w / 2), car_len, car_w, color=COLORS[i % 4], zorder=3)
        track.add_patch(body)
        # markers (sized in points) so wheels stay round on the unequal-scale axes
        rear = track.plot([], [], "o", ms=10, mec="k", zorder=4)[0]
        front = track.plot([], [], "o", ms=10, color="0.2", zorder=4)[0]
        cars[n], wheels[n] = body, (rear, front)
        labels[n] = track.text(0, y + car_w / 2 + 0.05, "", fontsize=8, va="bottom")

    lines = {}
    for i, n in enumerate(names):
        c = COLORS[i % 4]
        lines[n] = (speed.plot([], [], color=c, label=n)[0], slip_ax.plot([], [], color=c)[0])
    speed.set(xlim=(0, t_end), ylim=(0, max(d["v"].max() for d in data.values()) * 3.6 * 1.08),
              xlabel="time (s)", ylabel="speed (km/h)")
    slip_ax.set(xlim=(0, 1.5), ylim=(-0.05, 0.6), xlabel="time (s)", ylabel="rear slip (first 1.5 s)")
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
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    for i, (n, r) in enumerate(runs.items()):
        d = r.df.query("t < 1.0")
        axes[0].plot(d.t, d.motor_torque, color=COLORS[i % 4], label=n)
        axes[1].plot(d.t, d.slip.clip(-0.1, 1.0), color=COLORS[i % 4])
        axes[2].plot(d.t, d.a / G, color=COLORS[i % 4])
    for ax, y in zip(axes, ["motor torque (Nm)", "rear slip", "accel (g)"]):
        ax.set_ylabel(y)
    axes[0].legend()
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("Launch: first second")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _fd_sweep(motor, fds):
    rows = []
    for fd in fds:
        c = Car(final_drive=fd)
        rows.append((fd, quasi_static(c, motor).accel_time, simulate(c, motor).accel_time,
                     # the controller must be configured with the car's final drive: with
                     # the wrong ratio its rear wheel speed, and so its slip, is garbage
                     simulate(c, motor, controller=FirmwareController(mode="mu_adaptive",
                                                                      belief=c)).accel_time))
    return np.array(rows)


def _fig_fd(sweep, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for j, (label, c) in enumerate([("ideal TC", "k"), ("no TC", COLORS[0]),
                                    ("VCU mu_adaptive", COLORS[1])], start=1):
        ax.plot(sweep[:, 0], sweep[:, j], "o-", color=c, label=label)
    ax.set(xlabel="final drive", ylabel="accel time (s)", title="Accel time vs final drive")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _fig_robustness(mc, cols, path):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, max(mc[cols].max().max(), 1) * 1.05, 40)
    for i, c in enumerate(cols):
        ax.hist(mc[c], bins=bins, alpha=0.6, color=COLORS[i % 4], label=c)
    ax.set(xlabel="% over the plant's ideal time", ylabel="plants",
           title=f"{len(mc)} plants, every coefficient off, controllers at nominal")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _fig_controller(run, path):
    d, log = run.df.query("t < 2"), run.controller_log.query("0 <= t < 2")
    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axes[0].plot(d.t, d.v, "k", label="true speed")
    axes[0].plot(log.t, log.v_est, color=COLORS[1], label="v_est (VectorNav + wheels)")
    axes[0].plot(log.t, log.omega_fl * Car().tire_radius, color=COLORS[3], label="front wheel sensor")
    axes[1].plot(d.t, d.slip.clip(-0.3, 1), "k", label="true slip")
    axes[1].plot(log.t, log.slip.clip(-0.3, 1), color=COLORS[1], label="measured slip")
    axes[2].plot(d.t, d.motor_torque, "k", label="motor torque")
    axes[2].plot(log.t, log.torque_ff, color=COLORS[2], label="map torque from learned mu")
    for ax, y in zip(axes, ["m/s", "slip", "Nm"]):
        ax.set_ylabel(y)
        ax.legend(fontsize=8)
    axes[-1].set_xlabel("time (s)")
    fig.suptitle("What the VCU controller saw vs the truth (mu_adaptive)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _fig_sysid(fit, path):
    s = fit.signals
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(s.t, s.fx_meas, color="0.6", lw=1, label="rear tire force from measured accel")
    ax.plot(s.t, s.fx_model, color=COLORS[1], label="fitted tire model")
    ax.set(xlabel="time (s)", ylabel="Fx (N)", xlim=(0, 1.5),
           title="System ID: tire force, measured vs fitted (first 1.5 s)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="report")
    p.add_argument("--plants", type=int, default=20, help="Monte Carlo plants")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    car, motor = Car(), MotorMap("228HV")
    md = ["# Accel sim report", ""]

    # headline runs + GIF
    ideal = quasi_static(car, motor)
    runs = {
        "no TC": simulate(car, motor),
        "VCU slip_pi": simulate(car, motor, controller=FirmwareController(mode="slip_pi")),
        "VCU mu_adaptive": simulate(car, motor, controller=FirmwareController(mode="mu_adaptive")),
    }
    md += [f"Current car: EMRAX 228HV, COSMX 144s1p, FD {car.final_drive:g}, LC0 tire. "
           "Download the `accel-report` artifact for `drag_race.gif` and the plots.", "",
           "| run | 75 m time | vs ideal | trap speed | peak slip |", "|---|---|---|---|---|",
           f"| ideal TC (quasi-static) | {ideal.accel_time:.3f} s | | "
           f"{ideal.df.v.iloc[-1] * 3.6:.1f} km/h | |"]
    for n, r in runs.items():
        s = r.summary()
        md.append(f"| {n} | {r.accel_time:.3f} s | {100 * (r.accel_time / ideal.accel_time - 1):+.2f}% | "
                  f"{s['trap_kmh']:.1f} km/h | {s['max_slip']:.2f} |")
    drag_race_gif(runs, car, out / "drag_race.gif")
    _fig_launch(runs, out / "launch.png")
    _fig_controller(runs["VCU mu_adaptive"], out / "controller_view.png")

    # final drive sweep
    sweep = _fd_sweep(motor, [2.4, 2.7, 3.0, 3.5, 4.0, 4.5, 5.0])
    _fig_fd(sweep, out / "fd_sweep.png")
    md += ["", "## Final drive", "", "Lines, top to bottom at FD 2.7: no TC, VCU mu_adaptive, ideal TC.", ""]
    md += _mermaid_lines("Accel time (s) vs final drive", sweep[:, 0],
                         {"no TC": sweep[:, 2], "VCU": sweep[:, 3], "ideal": sweep[:, 1]}, "s")

    # robustness
    fw = lambda mode: (lambda pl: dict(controller=FirmwareController(
        mode=mode, sensors=Sensors(vn_accel_bias=pl.vn_accel_bias))))
    strategies = {"no TC": lambda pl: {}, "VCU slip_pi": fw("slip_pi"), "VCU mu_adaptive": fw("mu_adaptive")}
    mc = monte_carlo(strategies, sample_plants(args.plants, car), motor)
    cols = list(strategies)
    _fig_robustness(mc, cols, out / "robustness.png")
    md += ["", f"## Robustness ({args.plants} plants, every coefficient off)", "",
           "| controller | mean | p90 | worst |", "|---|---|---|---|"]
    for c in cols:
        md.append(f"| {c} | {mc[c].mean():+.2f}% | {mc[c].quantile(0.9):+.2f}% | {mc[c].max():+.2f}% |")

    # log -> replay -> fit loop on a hidden car
    real = sample_plants(1, car, seed=123)[0]
    vcu = FirmwareController(mode="slip_pi", sensors=Sensors(vn_accel_bias=real.vn_accel_bias))
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
           f"Binary log ({len(inputs)} records) replayed through the controller: "
           f"**{'identical' if identical else 'DIFFERENT'}** torque commands.", "",
           "| coefficient | truth | fit | prior range | narrowed |", "|---|---|---|---|---|"]
    for k in ("grip", "mass", "tire_radius", "relaxation_length"):
        lo, hi = narrowed[k]
        md.append(f"| {k} | {real.draw[k]:.4f} | {fit.params[k]:.4f} | "
                  f"{RANGES[k][0]:g} to {RANGES[k][1]:g} | {lo:.4f} to {hi:.4f} |")

    md += ["", f"_Generated in {time.perf_counter() - t0:.0f} s._", ""]
    (out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))
    if not identical:
        raise SystemExit("log replay did not reproduce the controller's outputs")


if __name__ == "__main__":
    main()
