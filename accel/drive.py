"""Drive the double-track car: a driver's pedal / brake / steering inputs, optionally
through the VCU controller, into vehicle.py's model.

    from accel.drive import drive
    from accel.driver import PathFollower, skidpad
    res = drive(PathFollower(skidpad(), speed=11.0))
    res.df          # logged every log_period: positions, speeds, per-wheel slip and
                    # forces, motor, pack SOC / temperature, driver inputs

    python -m accel.drive --out drive.png     # skidpad + a lap, plotted
"""
import argparse
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import vehicle
from .params import Car
from .powertrain import PACKS, MotorMap, Pack
from .tire import default_tire


@dataclass
class DriveResult:
    df: pd.DataFrame
    controller_log: pd.DataFrame | None = None


def drive(driver, car: Car = None, motor: MotorMap | str = "228HV",
          pack: Pack | str = "cosmx600v", tire=None, controller=None, *, dt: float = 2e-4,
          t_max: float = 60.0, driver_period: float = 0.01, log_period: float = 0.01,
          vx0: float = 0.0, soc0: float = 1.0, pack_temp0: float | None = None,
          relaxation_length: float = 0.15, relaxation_length_lat: float = 0.3) -> DriveResult:
    """Run until the driver is done or t_max.

    The driver's pedal maps linearly to a torque request (0..motor peak), as the VCU's
    pedal map would; controller (e.g. accel.firmware.FirmwareController) then turns
    that into the inverter command, from per-wheel sensor edges, motor rpm and
    body-frame velocity / acceleration. Without one, the request goes straight to
    the inverter. Brake pedal maps to car.max_brake_torque, split by brake_bias_front.
    """
    car = car or Car()
    motor = MotorMap(motor) if isinstance(motor, str) else motor
    pack = Pack(PACKS[pack]) if isinstance(pack, str) else pack
    tire = tire or default_tire()
    if controller is not None:
        controller.reset(car, motor)
    plant = vehicle.car_step(
        motor, tire, pack,
        vehicle.params(car, motor, pack, tire, relaxation_length=relaxation_length,
                       relaxation_length_lat=relaxation_length_lat, dt=dt),
        vehicle.initial_state(car, pack, vx=vx0, soc=soc0, pack_temp=pack_temp0))

    S, O = vehicle.STATES, vehicle.OUTPUTS
    iX, iY, ipsi, ivx, ivy, ir = (S.index(k) for k in ("X", "Y", "psi", "vx", "vy", "r"))
    iw = [S.index(f"omega_{w}") for w in vehicle.WHEELS]
    iax, iay, itq = O.index("ax"), O.index("ay"), O.index("motor_torque")
    rpm_per_rads = 60 / (2 * math.pi)
    gr = car.final_drive

    state, out = plant.state, [0.0] * len(O)
    pedal = brake = sw = 0.0
    t, next_driver, next_log = 0.0, 0.0, 0.0
    rows = []
    for _ in range(int(t_max / dt)):
        if t >= next_driver - 1e-12:
            next_driver += driver_period
            obs = {"X": state[iX], "Y": state[iY], "psi": state[ipsi], "vx": state[ivx],
                   "vy": state[ivy], "r": state[ir], "ax": out[iax], "ay": out[iay]}
            pedal, brake, sw = driver(t, obs)
            if driver.done:
                break
        t_req = pedal * motor.peak_torque
        if controller is not None:
            omegas = [state[i] for i in iw]
            motor_rpm = 0.5 * (omegas[2] + omegas[3]) * gr * rpm_per_rads
            t_cmd = controller.step(t, dt, omegas, motor_rpm, state[ivx], out[iax],
                                    out[itq], t_req)
        else:
            t_cmd = t_req
        plant(t_cmd, sw / car.steer_ratio, brake * car.max_brake_torque)
        if t >= next_log - 1e-12:
            next_log += log_period
            rows.append([t, pedal, brake, sw, t_req, t_cmd, *state, *plant.out])
        state, out = plant.state, plant.out
        t += dt

    df = pd.DataFrame(rows, columns=["t", "pedal", "brake", "steering_wheel", "torque_request",
                                     "torque_cmd", *S, *O])
    df["speed"] = np.hypot(df.vx, df.vy)
    df["beta"] = np.arctan2(df.vy, np.maximum(df.vx, 0.1))       # body slip angle
    df["rear_speed_diff"] = (df.omega_rl - df.omega_rr) * car.tire_radius   # m/s, + left faster
    return DriveResult(df, controller.log_frame() if controller is not None else None)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .driver import Path, PathFollower, skidpad
    from .envelope import gg_envelope, speed_profile

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="drive.png")
    args = ap.parse_args()

    car = Car()
    pad = skidpad()
    lap = Path.from_segments([("straight", 40), ("arc", 12, 180), ("straight", 25), ("arc", 6, -90),
                              ("arc", 8, 180), ("straight", 15), ("arc", 15, 90)])
    env = gg_envelope(car)
    runs = {
        "skidpad 10 m/s": drive(PathFollower(pad, speed=10.0, car=car), car),
        "course, 90% of envelope": drive(PathFollower(lap, speed=speed_profile(lap, env), car=car), car),
    }
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for (name, res), c in zip(runs.items(), ["#468ce6", "#e6504a"]):
        d = res.df
        axes[0, 0].plot(d.X, d.Y, color=c, label=name)
        axes[0, 1].plot(d.t, d.speed, color=c)
        axes[0, 2].plot(d.t, d.ay / 9.81, color=c)
        axes[1, 0].plot(d.t, np.degrees(d.beta), color=c)
        axes[1, 1].plot(d.t, d.pack_temp, color=c)
        axes[1, 2].plot(d.t, 100 * d.soc, color=c)
    axes[0, 0].plot(pad.x, pad.y, "k:", lw=0.8)
    axes[0, 0].plot(lap.x, lap.y, "k:", lw=0.8)
    axes[0, 0].set_aspect("equal")
    axes[0, 0].legend()
    for ax, (title, y) in zip(axes.flat, [("path", "y (m)"), ("speed", "m/s"), ("lateral accel", "g"),
                                          ("body slip angle", "deg"), ("pack temperature", "C"),
                                          ("state of charge", "%")]):
        ax.set(title=title, ylabel=y)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(args.out)
    for name, res in runs.items():
        d = res.df
        print(f"{name}: {d.t.iloc[-1]:.1f} s, peak ay {d.ay.abs().max() / 9.81:.2f} g, "
              f"pack {d.pack_temp.iloc[0]:.1f} -> {d.pack_temp.iloc[-1]:.2f} C, "
              f"SOC {100 * d.soc.iloc[0]:.1f} -> {100 * d.soc.iloc[-1]:.2f}%")


if __name__ == "__main__":
    main()
