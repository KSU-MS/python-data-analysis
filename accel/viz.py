"""Replay accel runs in Rerun (https://rerun.io): a 3D drag race, one lane per run,
with plots synced to the same timeline.

    python -m accel.viz                    # open the viewer with the standard comparison
    python -m accel.viz --save runs.rrd    # or save, then `rerun runs.rrd`

Or from Python/the notebook:

    log_runs({"no TC": simulate(car), "VCU mu_adaptive": simulate(car, controller=...)}, car)

Rear wheels are drawn with a spoke so wheelspin shows as the rear spokes outrunning the
fronts; arrows show the rear tire force (forward) and rear axle load (down).
"""
import argparse
import subprocess
import tempfile

import numpy as np
import rerun as rr
import rerun.blueprint as rrb

from .params import G, Car
from .sim import LENGTH, STAGE, Result

LANE_SPACING = 3.0     # m between lanes
PITCH_PER_G = 0.04     # rad of body pitch per g, exaggerated so it's visible
FORCE_SCALE = 1 / 2000  # m of arrow per N
COLORS = [(230, 80, 70), (70, 140, 230), (90, 190, 110), (230, 170, 50), (170, 100, 220)]


def _quat_y(angle):
    """xyzw quaternions for rotations about +y (positive = nose down)."""
    half = np.asarray(angle) / 2
    z = np.zeros_like(half)
    return np.column_stack([z, np.sin(half), z, np.cos(half)])


def _path(name: str) -> str:
    """Entity path segment for a run name (Rerun paths can't contain spaces)."""
    return "".join(c if c.isalnum() or c in "_-" else "_" for c in name)


def _log_static(car: Car, names):
    r, wb = car.tire_radius, car.wheelbase
    length = STAGE + LENGTH + 5
    width = LANE_SPACING * len(names)
    y_mid = -LANE_SPACING * (len(names) - 1) / 2
    rr.log("track/surface", rr.Boxes3D(centers=[(length / 2 - 3, y_mid, -0.01)],
                                       half_sizes=[(length / 2, width / 2, 0.01)],
                                       colors=[(60, 60, 60)]), static=True)
    lines = []
    for x in (STAGE, STAGE + LENGTH):
        lines.append([(x, y_mid - width / 2, 0.005), (x, y_mid + width / 2, 0.005)])
    rr.log("track/lines", rr.LineStrips3D(lines, colors=[(255, 255, 255)], radii=0.05,
                                          labels=["start", "finish"]), static=True)
    marks = [[(x, y_mid - width / 2, 0.005), (x, y_mid - width / 2 + 0.4, 0.005)]
             for x in range(10, int(STAGE + LENGTH), 10)]
    rr.log("track/10m_marks", rr.LineStrips3D(marks, colors=[(200, 200, 200)], radii=0.03),
           static=True)

    for i, name in enumerate(names):
        base = f"cars/{_path(name)}"
        color = COLORS[i % len(COLORS)]
        rr.log(f"{base}/body", rr.Boxes3D(centers=[(wb / 2, 0, r + 0.15)],
                                         half_sizes=[(wb / 2 + 0.45, 0.45, 0.15)],
                                         colors=[color], labels=[name]), static=True)
        for w, (x, y) in {"fl": (wb, 0.6), "fr": (wb, -0.6), "rl": (0, 0.6), "rr": (0, -0.6)}.items():
            rr.log(f"{base}/wheel_{w}", rr.Transform3D(translation=(x, y, r)), static=True)
            rr.log(f"{base}/wheel_{w}/spin/tire",
                   rr.Cylinders3D(lengths=[0.2], radii=[r], colors=[(30, 30, 30)],
                                  rotation_axis_angles=[rr.RotationAxisAngle(axis=(1, 0, 0), degrees=90)]),
                   static=True)
            # spoke offset to the outside face so it's visible
            side = 0.11 if y > 0 else -0.11
            rr.log(f"{base}/wheel_{w}/spin/spoke",
                   rr.Boxes3D(centers=[(0, side, 0)], half_sizes=[(r * 0.95, 0.005, 0.03)],
                              colors=[(240, 240, 240)]), static=True)


def log_runs(runs: dict, car: Car = None, rate_hz: float = 200.0, plots=None):
    """Log runs (name -> Result) to the current Rerun recording. Call rr.init/spawn/save
    first (or use show_runs)."""
    car = car or Car()
    names = list(runs)
    _log_static(car, names)
    r = car.tire_radius
    static_rear = car.rear_load(0.0, 0.0)

    for i, (name, res) in enumerate(runs.items()):
        d = res.df
        dt = float(d["t"].iloc[1] - d["t"].iloc[0])
        # wheel angles integrate at full rate, then downsample for display
        front_angle = np.cumsum(d["v"].to_numpy() / r * dt)
        rear_angle = np.cumsum(d["wheel_speed"].to_numpy() / r * dt) if "wheel_speed" in d \
            else front_angle
        step = max(int(round(1 / (rate_hz * dt))), 1)
        idx = np.arange(0, len(d), step)
        s = d.iloc[idx]
        t = s["t"].to_numpy()
        times = [rr.TimeColumn("time", duration=t)]
        n = len(t)
        base = f"cars/{_path(name)}"
        y = -LANE_SPACING * i

        pitch = -PITCH_PER_G * s["a"].to_numpy() / G       # nose up under acceleration
        rr.send_columns(base, indexes=times, columns=rr.Transform3D.columns(
            translation=np.column_stack([s["x"].to_numpy(), np.full(n, y), np.zeros(n)]),
            quaternion=_quat_y(pitch)))
        # wheel rotation: rolling forward is negative about +y
        for w, ang in (("fl", front_angle), ("fr", front_angle), ("rl", rear_angle), ("rr", rear_angle)):
            rr.send_columns(f"{base}/wheel_{w}/spin", indexes=times,
                            columns=rr.Transform3D.columns(quaternion=_quat_y(-ang[idx])))

        if "fx" in s:
            fx = s["fx"].to_numpy()
            rr.send_columns(f"{base}/rear_force", indexes=times, columns=rr.Arrows3D.columns(
                origins=np.tile((0.0, 0.0, 0.02), (n, 1)),
                vectors=np.column_stack([fx * FORCE_SCALE, np.zeros(n), np.zeros(n)])))
        if "fz_rear" in s:
            fz = s["fz_rear"].to_numpy()
            rr.send_columns(f"{base}/rear_load", indexes=times, columns=rr.Arrows3D.columns(
                origins=np.tile((0.0, 0.0, 1.2), (n, 1)),
                vectors=np.column_stack([np.zeros(n), np.zeros(n), -(fz / static_rear) * 0.6])))

        series = {
            "speed_kmh": s["v"].to_numpy() * 3.6,
            "accel_g": s["a"].to_numpy() / G,
            "motor_torque_nm": s["motor_torque"].to_numpy(),
            "slip": s["slip"].clip(-0.2, 1.0).to_numpy() if "slip" in s else None,
            "tc_scale": s["tc_scale"].to_numpy() if "tc_scale" in s else None,
        }
        log = res.controller_log
        if log is not None:
            log = log[log["t"] >= 0]
            lt = [rr.TimeColumn("time", duration=log["t"].to_numpy())]
            for key in ("mu", "v_est"):
                if key in log:
                    rr.send_columns(f"plots/{key}/{_path(name)}", indexes=lt,
                                    columns=rr.Scalars.columns(scalars=log[key].to_numpy()))
        for key, vals in series.items():
            if vals is not None:
                rr.send_columns(f"plots/{key}/{_path(name)}", indexes=times,
                                columns=rr.Scalars.columns(scalars=vals))
        color = COLORS[i % len(COLORS)]
        for key in list(series) + ["mu", "v_est"]:
            rr.log(f"plots/{key}/{_path(name)}", rr.SeriesLines(colors=[color], names=[name]),
                   static=True)

    plots = plots or ["speed_kmh", "accel_g", "slip", "motor_torque_nm", "tc_scale", "mu"]
    rr.send_blueprint(rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/", contents=["/track/**", "/cars/**"], name="drag race"),
            rrb.Grid(*[rrb.TimeSeriesView(origin=f"/plots/{p}", name=p) for p in plots]),
            column_shares=[3, 2]),
        rrb.TimePanel(state="expanded"),
        collapse_panels=True))


def show_runs(runs: dict, car: Car = None, save: str | None = None, **kw):
    """Save the runs to an .rrd and open the Rerun viewer on it (or only save, if
    `save` is given). The viewer is the `rerun` binary from the dev shell; nixpkgs'
    Python SDK doesn't bundle it, so rr.spawn() doesn't work there."""
    path = save or tempfile.NamedTemporaryFile(suffix=".rrd", delete=False).name
    rr.init("accel_sim")
    rr.save(path)
    log_runs(runs, car, **kw)
    rr.disconnect()
    if save is None:
        subprocess.Popen(["rerun", path])
    return path


def standard_runs(car: Car = None) -> dict:
    """No TC vs the firmware controllers on the current car."""
    from .firmware import FirmwareController
    from .powertrain import MotorMap
    from .sim import simulate

    car = car or Car().open_aero()
    motor = MotorMap("228HV")
    return {
        "no TC": simulate(car, motor),
        "VCU slip_pi": simulate(car, motor, controller=FirmwareController(mode="slip_pi", belief=car)),
        "VCU mu_adaptive": simulate(car, motor, controller=FirmwareController(mode="mu_adaptive",
                                                                              belief=car)),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--save", metavar="RRD", help="write an .rrd instead of opening the viewer")
    p.add_argument("--fd", type=float, default=Car.final_drive)
    args = p.parse_args()
    car = Car(final_drive=args.fd).open_aero()
    runs = standard_runs(car)
    for name, res in runs.items():
        print(f"{name:18s} {res.accel_time:.3f} s")
    show_runs(runs, car, save=args.save)


if __name__ == "__main__":
    main()
