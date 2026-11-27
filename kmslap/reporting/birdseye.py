"""Bird's-eye view of endurance strategies racing the same flying lap.

    kmslap birdseye --out birdseye/        # lap GIF + a static speed map

Each strategy (PRESETS: the endurance tuning's picks) drives two laps of the Michigan
2026 EV line from a standing start; lap 2, the flying lap, is animated with every car
starting together: the cars on the track with speed-colored trails, what each is doing
(drive / coast / brake / regen, DRS), speed along the lap and the running gap to the
fastest. The line is the traced centerline here; laps saved with
reporting/runs.save_lap (any line, any track) race the same way:

    lap_gif(runs.load_laps("ac18"), track=michigan_ac18(), out="ac18.gif")
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from kmslap.reporting.runs import lap_frame

COLORS = ["#e6504a", "#468ce6", "#5abe6e", "#e6aa32", "#9b59b6", "#17becf", "#8c564b", "#e377c2",
          "#7f7f7f", "#bcbd22"]


@dataclass
class Strategy:
    name: str
    margin: float = 1.0                  # fraction of the grip envelope
    coast_time: float = 0.0              # s, lift-and-coast
    drs: dict | None = None              # DrsPolicy kwargs
    regen: dict | None = None            # RegenPolicy kwargs
    car: dict = field(default_factory=dict)   # Car overrides


# from `kmslap strategy --regen` on the measured aero (see the README)
PRESETS = [
    Strategy("best: regen, coast 0.5 s", 1.0, 0.5, regen={"lift_torque": 40.0, "brake_fraction": 1.0}),
    Strategy("no regen, coast 1.5 s", 1.0, 1.5),
    Strategy("no regen, coast 1.5 s, DRS", 1.0, 1.5, drs={}),
    Strategy("80% pace, no tricks", 0.8, 0.0),
]


def drive_lap(s: Strategy, path=None) -> pd.DataFrame:
    """Lap 2 (flying) of a two-lap run, time from its start line."""
    from kmslap.driving.policies import DrsPolicy, RegenPolicy
    from kmslap.events.endurance import endurance
    from kmslap.events.track import michigan_2026_endurance
    from kmslap.models.envelope import gg_envelope, speed_profile
    from kmslap.car.params import Car
    import dataclasses

    car = dataclasses.replace(Car(), **s.car)
    path = path or michigan_2026_endurance()
    speed = speed_profile(path, gg_envelope(car), margin=s.margin)
    res = endurance(car, laps=2, path=path, speed=speed, coast_time=s.coast_time,
                    drs=DrsPolicy(**s.drs) if s.drs is not None else None,
                    regen=RegenPolicy(**s.regen) if s.regen is not None else None,
                    driver_change=False, log_period=0.05)
    t1, t2 = res.lap_times[0], res.lap_times[0] + res.lap_times[1]
    return lap_frame(res.df, t1, t2)


def _rotation(path):
    """Rotate the track so its long axis is horizontal."""
    xy = np.column_stack([path.x, path.y])
    xy = xy - xy.mean(0)
    _, _, vt = np.linalg.svd(xy, full_matrices=False)
    ang = np.arctan2(vt[0, 1], vt[0, 0])
    c, s = np.cos(-ang), np.sin(-ang)
    center = np.array([path.x.mean(), path.y.mean()])
    return lambda x, y: (c * (x - center[0]) - s * (y - center[1]), s * (x - center[0]) + c * (y - center[1]))


def _color(i):
    return COLORS[i % len(COLORS)]


def lap_gif(laps: dict, path=None, out="flying_lap.gif", fps: int = 15, speedup: float = 6.0,
            track=None, title: str = "Michigan 2026 EV endurance, flying lap"):
    """laps: {name: lap DataFrame (lap_frame's columns)}. track: a events.track.Track
    to draw its edges and cones (else `path`, the centerline, as a band)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter
    from matplotlib.collections import LineCollection

    path = track.center if track is not None else path
    rot = _rotation(path)
    tx, ty = rot(path.x, path.y)
    finished = {n: d.attrs.get("finished", True) for n, d in laps.items()}
    # end with the last finisher (a lap that didn't finish would run on for its spin)
    t_end = max((d.t_lap.iloc[-1] for n, d in laps.items() if finished[n]),
                default=max(d.t_lap.iloc[-1] for d in laps.values())) + 2.0
    frames = np.arange(0, t_end + 1 / fps * speedup, speedup / fps)
    vmax = max(d.speed.max() for d in laps.values())
    fastest = min((n for n in laps if finished[n]), key=lambda n: laps[n].t_lap.iloc[-1], default=next(iter(laps)))
    ref = laps[fastest]
    n = len(laps)
    status_rows = (n + 1) // 2
    fig = plt.figure(figsize=(14, 9 + 0.25 * status_rows))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.6, 0.12 * status_rows + 0.1, 1], width_ratios=[1, 1])
    ax = fig.add_subplot(gs[0, :])
    if track is not None:
        lx, ly, rx, ry = track.edges()
        for ex, ey in ((lx, ly), (rx, ry)):
            ex, ey = rot(np.r_[ex, ex[0]], np.r_[ey, ey[0]])
            ax.plot(ex, ey, color="0.55", lw=0.8, zorder=0)
        if track.cones is not None:
            cx, cy = rot(track.cones[:, 0], track.cones[:, 1])
            ax.plot(cx, cy, "^", color="#f28e2b", ms=3, zorder=1)
    else:
        ax.plot(tx, ty, color="0.85", lw=9, solid_capstyle="round", zorder=0)
        ax.plot(tx, ty, color="0.6", lw=0.6, ls="--", zorder=1)
    ax.set_aspect("equal")
    ax.axis("off")
    sx, sy = rot(np.array([path.x[0]]), np.array([path.y[0]]))
    ax.plot(sx, sy, "k|", ms=18, mew=3, zorder=2)
    cmap = plt.get_cmap("viridis")
    trails, dots, badges = [], [], []
    for i, (name, d) in enumerate(laps.items()):
        lc = LineCollection([], cmap=cmap, norm=plt.Normalize(0, vmax), lw=3, alpha=0.8, zorder=3)
        ax.add_collection(lc)
        trails.append(lc)
        dots.append(ax.plot([], [], "o", ms=13, color=_color(i), mec="k", zorder=5)[0])
        # the car's number on its dot; the status panel below is the key
        badges.append(ax.text(0, 0, str(i + 1), fontsize=7, color="white", fontweight="bold",
                              ha="center", va="center", zorder=6))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, vmax))
    fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01, label="speed (m/s)")
    title_text = title
    title = ax.set_title("", fontsize=12)

    # status panel: number, name, what the car is doing (the legend, kept off the track)
    st = fig.add_subplot(gs[1, :])
    st.axis("off")
    status = []
    for i, name in enumerate(laps):
        col, row = divmod(i, status_rows)
        x0, y0 = 0.5 * col, 1 - (row + 0.5) / status_rows
        st.text(x0, y0, f" {i + 1} ", fontsize=8, color="white", fontweight="bold", va="center",
                transform=st.transAxes, bbox=dict(boxstyle="circle,pad=0.15", fc=_color(i), ec="k", lw=0.6))
        st.text(x0 + 0.025, y0, name, fontsize=8.5, color="0.15", va="center", transform=st.transAxes)
        status.append(st.text(x0 + 0.36, y0, "", fontsize=8.5, color=_color(i), fontweight="bold",
                              va="center", transform=st.transAxes, family="monospace"))

    a1 = fig.add_subplot(gs[2, 0])
    a2 = fig.add_subplot(gs[2, 1])
    marks1, marks2 = [], []
    for i, (name, d) in enumerate(laps.items()):
        a1.plot(d.dist, d.speed, color=_color(i), lw=1)
        marks1.append(a1.plot([], [], "o", color=_color(i), mec="k")[0])
        grid = np.linspace(0, min(d.dist.iloc[-1], ref.dist.iloc[-1]), 800)
        gap = np.interp(grid, d.dist, d.t_lap) - np.interp(grid, ref.dist, ref.t_lap)
        a2.plot(grid, gap, color=_color(i), lw=1.2)
        marks2.append(a2.plot([], [], "o", color=_color(i), mec="k")[0])
    lap_len = max(d.dist.iloc[-1] for nm, d in laps.items() if finished[nm]) if any(finished.values()) \
        else ref.dist.iloc[-1]
    a1.set_xlim(0, lap_len * 1.02)        # a lap that didn't finish would stretch it
    a1.set(xlabel="distance along the lap (m)", ylabel="speed (m/s)", title="Speed")
    a2.set(xlabel="distance along the lap (m)", ylabel="behind (s)", title=f"Gap to {fastest}")
    for a in (a1, a2):
        a.grid(alpha=0.3)

    def at(d, t, col):
        return np.interp(t, d.t_lap, d[col])

    def draw(k):
        t = frames[k]
        artists = []
        for i, (name, d) in enumerate(laps.items()):
            m = d.t_lap <= t
            x, y = rot(d.X[m].to_numpy(), d.Y[m].to_numpy())
            if len(x) > 1:
                pts = np.column_stack([x, y]).reshape(-1, 1, 2)
                trails[i].set_segments(np.concatenate([pts[:-1], pts[1:]], axis=1)[-400:])
                trails[i].set_array(d.speed[m].to_numpy()[1:][-400:])
            px, py = rot(np.array([at(d, t, "X")]), np.array([at(d, t, "Y")]))
            dots[i].set_data(px, py)
            badges[i].set_position((px[0], py[0]))
            row = d.iloc[min(int(np.searchsorted(d.t_lap, t)), len(d) - 1)]
            done = t >= d.t_lap.iloc[-1]
            status[i].set_text("DNF" if not finished[name] and done else
                               f"finished {d.t_lap.iloc[-1]:6.2f} s" if done else
                               f"{row.speed:4.1f} m/s  {row['mode']:5s}{'  DRS' if row.get('drs', 0) > 0.5 else ''}")
            dist = at(d, t, "dist")
            marks1[i].set_data([dist], [at(d, t, "speed")])
            g = np.interp(dist, d.dist, d.t_lap) - np.interp(dist, ref.dist, ref.t_lap)
            marks2[i].set_data([dist], [g])
            artists += [trails[i], dots[i], badges[i], status[i], marks1[i], marks2[i]]
        title.set_text(f"{title_text}: t = {t:5.1f} s  ({speedup:g}x)")
        return artists

    fig.tight_layout()
    # .mp4 through ffmpeg (much smaller and faster to write); anything else a GIF
    writer = FFMpegWriter(fps=fps, bitrate=3000) if str(out).endswith(".mp4") else PillowWriter(fps=fps)
    FuncAnimation(fig, draw, frames=len(frames), blit=False).save(out, writer=writer)
    plt.close(fig)


def speed_map(laps: dict, path, out):
    """Each strategy's flying lap on the track: color = speed, marks where it brakes,
    regens and has DRS open."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    rot = _rotation(path)
    n = len(laps)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.4 * n))
    vmax = max(d.speed.max() for d in laps.values())
    for a, (name, d) in zip(np.atleast_1d(axes), laps.items()):
        x, y = rot(d.X.to_numpy(), d.Y.to_numpy())
        pts = np.column_stack([x, y]).reshape(-1, 1, 2)
        lc = LineCollection(np.concatenate([pts[:-1], pts[1:]], axis=1), cmap="viridis",
                            norm=plt.Normalize(0, vmax), lw=4)
        lc.set_array(d.speed.to_numpy()[1:])
        a.add_collection(lc)
        for mode, c, mk in [("brake", "#e6504a", "v"), ("regen", "#ff00c8", "o")]:
            m = (d["mode"] == mode).to_numpy()
            a.plot(x[m][::6], y[m][::6], mk, color=c, ms=3.5, label=mode, alpha=0.9)
        m = (d.drs > 0.5).to_numpy()
        if m.any():
            a.plot(x[m][::6], y[m][::6], "s", color="#9b59b6", ms=3, label="DRS open")
        a.set_aspect("equal")
        a.axis("off")
        a.autoscale()
        a.set_title(f"{name}: {d.t_lap.iloc[-1]:.2f} s, {d.speed.max():.1f} m/s max", fontsize=10)
        a.legend(loc="upper left", fontsize=7, markerscale=2)
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, vmax))
    fig.colorbar(sm, ax=np.atleast_1d(axes).tolist(), fraction=0.015, label="speed (m/s)")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def run(out: str = "birdseye", strategies=None, gif: bool = True) -> dict:
    from kmslap.events.track import michigan_2026_endurance

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    path = michigan_2026_endurance()
    laps = {s.name: drive_lap(s, path) for s in (strategies or PRESETS)}
    for name, d in laps.items():
        print(f"{name:32s} flying lap {d.t_lap.iloc[-1]:6.2f} s, max {d.speed.max():4.1f} m/s, "
              f"drive {np.mean(d['mode'] == 'drive') * 100:3.0f}% coast {np.mean(d['mode'] == 'coast') * 100:3.0f}% "
              f"brake {np.mean(d['mode'] == 'brake') * 100:3.0f}% regen {np.mean(d['mode'] == 'regen') * 100:3.0f}% "
              f"DRS {np.mean(d.drs > 0.5) * 100:3.0f}%")
    speed_map(laps, path, out / "lap_speed_map.png")
    if gif:
        lap_gif(laps, path, out / "flying_lap.gif")
    return laps
