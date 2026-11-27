"""Saved laps (reporting/runs) racing in 3D in Rerun: the track (surface, edges, cones),
each car the FSAE car of the Foxglove log (car_model) in its own colour, front wheels steered, its whole line coloured
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
from kmslap.reporting import car_model as cm
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


def _log_car(base: str, car: Car, color, label: str):
    """The FSAE car (car_model) at `base`, its livery in `color`; wheels at
    base/wheel_<w>/steer (the front ones turned by the steer transform)."""
    rgb = lambda c: tuple(int(255 * v) for v in c)
    parts = cm.body(car)
    for kind in ("cube", "cylinder", "sphere"):
        ps = [p for p in parts if p.kind == kind]
        cols = [color if p.paint else rgb(p.color) for p in ps]
        if kind == "cube":
            arch = rr.Boxes3D(centers=[p.center for p in ps], half_sizes=[np.divide(p.size, 2) for p in ps],
                              quaternions=[p.quat for p in ps], colors=cols, fill_mode="solid")
        elif kind == "cylinder":                         # Rerun's have one radius: a cone as its mean
            arch = rr.Cylinders3D(centers=[p.center for p in ps], radii=[p.size[0] * (1 + p.top) / 4 for p in ps],
                                  lengths=[p.size[2] for p in ps], quaternions=[p.quat for p in ps], colors=cols,
                                  fill_mode="solid")
        else:
            arch = rr.Ellipsoids3D(centers=[p.center for p in ps], half_sizes=[np.divide(p.size, 2) for p in ps],
                                   colors=cols, fill_mode="solid")
        rr.log(f"{base}/body/{kind}s", arch, static=True)
    rr.log(f"{base}/body/label", rr.Points3D([(0, 0, 1.5)], radii=0.01, labels=[label], colors=[color]), static=True)
    for w in cm.wheels(car):
        rr.log(f"{base}/wheel_{w.name}", rr.Transform3D(translation=(w.x, w.y, car.tire_radius)), static=True)
        ws = cm.wheel_parts(car)
        rr.log(f"{base}/wheel_{w.name}/steer/wheel",
               rr.Cylinders3D(radii=[p.size[0] / 2 for p in ws], lengths=[p.size[2] for p in ws],
                              quaternions=[p.quat for p in ws], colors=[rgb(p.color) for p in ws],
                              fill_mode="solid"), static=True)


def log_race(laps: dict, track: T.Track, car: Car = None):
    car = car or Car()
    _log_track(track)
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
        _log_car(base, car, color, f"{i + 1}")

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
        if "motor_torque" in d:                          # - : regen
            series["motor torque (N m)"] = d.motor_torque.to_numpy()
        if "brake" in d:
            series["brake pedal"] = d.brake.to_numpy()
        for key, vals in series.items():
            path = f"plots/{_path(key)}/{i + 1}_{_path(name)}"
            rr.send_columns(path, indexes=times, columns=rr.Scalars.columns(scalars=vals))
            rr.log(path, rr.SeriesLines(colors=[color], names=[f"{i + 1} {name}"]), static=True)
    rr.send_blueprint(rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/", contents=["/track/**", "/cars/**", "/lines/**"], name="race"),
            rrb.Vertical(*[rrb.TimeSeriesView(origin=f"/plots/{_path(k)}", name=k)
                           for k in ("speed (m/s)", "gap to fastest (s)", "body slip (deg)", "steering wheel (deg)",
                                     "motor torque (N m)", "brake pedal")]),
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
