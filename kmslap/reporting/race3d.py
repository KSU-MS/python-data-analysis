"""Saved laps (reporting/runs) racing in 3D in Rerun: the track (surface, edges, cones),
each car as a body with steered front wheels on its own colour, its whole line coloured
by speed, and speed / body slip / steering / gap plots on the same timeline.

    kmslap race ac18 --rerun                 # open the viewer
    kmslap race ac18 --rerun --save ac18.rrd

Time is each lap's own, from its start line, so the cars start together (a time trial).
"""
import subprocess
import tempfile

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from kmslap.car.params import Car
from kmslap.events import track as T
from kmslap.reporting.viz import _path

COLORS = [(230, 80, 74), (70, 140, 230), (90, 190, 110), (230, 170, 50), (155, 89, 182), (23, 190, 207),
          (140, 86, 75), (227, 119, 194), (127, 127, 127), (188, 189, 34)]


def _quat_z(angle):
    half = np.asarray(angle, float) / 2
    z = np.zeros_like(half)
    return np.column_stack([z, z, np.sin(half), np.cos(half)])


def _speed_colors(v, vmax):
    import matplotlib
    return (matplotlib.colormaps["viridis"](np.clip(v / vmax, 0, 1))[:, :3] * 255).astype(np.uint8)


def _log_track(track: T.Track):
    lx, ly, rx, ry = track.edges()
    n = len(lx)
    # the surface: a ribbon between the edges
    verts = np.vstack([np.column_stack([lx, ly, np.zeros(n)]), np.column_stack([rx, ry, np.zeros(n)])])
    i = np.arange(n)
    j = (i + 1) % n
    tris = np.vstack([np.column_stack([i, j, n + i]), np.column_stack([j, n + j, n + i])])
    rr.log("track/surface", rr.Mesh3D(vertex_positions=verts - [0, 0, 0.01], triangle_indices=tris,
                                      vertex_colors=np.tile([70, 70, 74], (2 * n, 1))), static=True)
    rr.log("track/edges", rr.LineStrips3D([np.column_stack([np.r_[lx, lx[0]], np.r_[ly, ly[0]], np.zeros(n + 1)]),
                                           np.column_stack([np.r_[rx, rx[0]], np.r_[ry, ry[0]], np.zeros(n + 1)])],
                                          colors=[(235, 235, 235)], radii=0.04), static=True)
    c = track.center
    nx, ny = T.normals(c)
    start = [(c.x[0] + track.w_left[0] * nx[0], c.y[0] + track.w_left[0] * ny[0], 0.01),
             (c.x[0] - track.w_right[0] * nx[0], c.y[0] - track.w_right[0] * ny[0], 0.01)]
    rr.log("track/start", rr.LineStrips3D([start], colors=[(255, 255, 255)], radii=0.15, labels=["start / finish"]),
           static=True)
    if track.cones is not None:
        k = len(track.cones)
        rr.log("track/cones", rr.Cylinders3D(lengths=np.full(k, 0.7), radii=np.full(k, 0.15),
                                             centers=np.column_stack([track.cones, np.full(k, 0.35)]),
                                             colors=[(242, 142, 43)]), static=True)


def log_race(laps: dict, track: T.Track, car: Car = None):
    car = car or Car()
    _log_track(track)
    lf = car.cg_to_front
    lr = car.wheelbase - lf
    r = car.tire_radius
    vmax = max(d.speed.max() for d in laps.values())
    ref = min(laps.values(), key=lambda d: d.t_lap.iloc[-1] if d.attrs.get("finished", True) else np.inf)
    for i, (name, d) in enumerate(laps.items()):
        base = f"cars/{i + 1}_{_path(name)}"
        color = COLORS[i % len(COLORS)]
        # static: the whole line, coloured by speed, and the car's parts
        pts = np.column_stack([d.X, d.Y, np.full(len(d), 0.02 + 0.005 * i)])
        rr.log(f"lines/{i + 1}_{_path(name)}", rr.LineStrips3D([pts], colors=[color], radii=0.03), static=True)
        rr.log(f"lines/{i + 1}_{_path(name)}/speed", rr.Points3D(pts[::5], colors=_speed_colors(d.speed.to_numpy()[::5], vmax),
                                                                radii=0.06), static=True)
        rr.log(f"{base}/body", rr.Boxes3D(centers=[((lf - lr) / 2, 0, r + 0.12)],
                                         half_sizes=[(car.wheelbase / 2 + 0.45, 0.42, 0.14)],
                                         colors=[color], labels=[f"{i + 1}"]), static=True)
        rr.log(f"{base}/nose", rr.Boxes3D(centers=[(lf + 0.3, 0, r + 0.08)], half_sizes=[(0.2, 0.25, 0.08)],
                                         colors=[color]), static=True)
        for w, (x, y) in {"fl": (lf, car.track_front / 2), "fr": (lf, -car.track_front / 2),
                          "rl": (-lr, car.track_rear / 2), "rr": (-lr, -car.track_rear / 2)}.items():
            rr.log(f"{base}/wheel_{w}", rr.Transform3D(translation=(x, y, r)), static=True)
            rr.log(f"{base}/wheel_{w}/steer/tire",
                   rr.Cylinders3D(lengths=[0.2], radii=[r], colors=[(25, 25, 25)],
                                  rotation_axis_angles=[rr.RotationAxisAngle(axis=(1, 0, 0), degrees=90)]),
                   static=True)

        t = d.t_lap.to_numpy()
        times = [rr.TimeColumn("lap time", duration=t)]
        rr.send_columns(base, indexes=times, columns=rr.Transform3D.columns(
            translation=np.column_stack([d.X, d.Y, np.zeros(len(d))]), quaternion=_quat_z(d.psi.to_numpy())))
        steer = d.steering_wheel.to_numpy() / car.steer_ratio
        for w in ("fl", "fr"):
            rr.send_columns(f"{base}/wheel_{w}/steer", indexes=times,
                            columns=rr.Transform3D.columns(quaternion=_quat_z(steer)))
        gap = np.interp(d.dist, ref.dist, ref.t_lap, right=np.nan)
        series = {"speed (m/s)": d.speed.to_numpy(), "body slip (deg)": np.degrees(d.beta.to_numpy()),
                  "steering wheel (deg)": np.degrees(d.steering_wheel.to_numpy()),
                  "gap to fastest (s)": t - gap}
        for key, vals in series.items():
            path = f"plots/{_path(key)}/{i + 1}_{_path(name)}"
            rr.send_columns(path, indexes=times, columns=rr.Scalars.columns(scalars=vals))
            rr.log(path, rr.SeriesLines(colors=[color], names=[f"{i + 1} {name}"]), static=True)
    rr.send_blueprint(rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/", contents=["/track/**", "/cars/**", "/lines/**"], name="race"),
            rrb.Vertical(*[rrb.TimeSeriesView(origin=f"/plots/{_path(k)}", name=k)
                           for k in ("speed (m/s)", "gap to fastest (s)", "body slip (deg)", "steering wheel (deg)")]),
            column_shares=[3, 2]),
        rrb.TimePanel(state="expanded"),
        collapse_panels=True))


def show(laps: dict, track: T.Track, save: str | None = None) -> str:
    """Write the race to an .rrd and open the Rerun viewer on it (only write, if `save`)."""
    path = save or tempfile.NamedTemporaryFile(suffix=".rrd", delete=False).name
    rr.init("kmslap_race")
    rr.save(path)
    log_race(laps, track)
    rr.disconnect()
    if save is None:
        subprocess.Popen(["rerun", path])
    return path
