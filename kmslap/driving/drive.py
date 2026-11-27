"""Drive the double-track car: a driver's pedal / brake / steering inputs, optionally
through the VCU controller, into vehicle.py's model.

    from kmslap.driving.drive import drive
    from kmslap.driving.follower import PathFollower
    from kmslap.driving.paths import skidpad
    res = drive(PathFollower(skidpad(), speed=11.0))
    res.df          # logged every log_period: positions, speeds, per-wheel slip and
                    # forces, motor, pack SOC / temperature, driver inputs

    accel drive --out drive.png     # skidpad + a lap, plotted
"""
import contextlib
import math
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from kmslap import dirs
from kmslap.models import twotrack as vehicle
from kmslap.car.params import Car
from kmslap.car.powertrain import PACKS, MotorMap, Pack
from kmslap.car.tire import default_tire


@dataclass
class DriveResult:
    df: pd.DataFrame
    controller_log: pd.DataFrame | None = None


def drive(driver, car: Car = None, motor: MotorMap | str = "228HV",
          pack: Pack | str = "cosmx600v", tire=None, controller=None, *, dt: float = 2e-4,
          t_max: float = 60.0, driver_period: float = 0.01, log_period: float = 0.01,
          vx0: float = 0.0, soc0: float = 1.0, pack_temp0: float | None = None,
          relaxation_length: float = 0.15, relaxation_length_lat: float = 0.3,
          drs=None, regen=None, X0: float = 0.0, Y0: float = 0.0, psi0: float = 0.0,
          pack_backend=None, realtime: bool = False, on_tick=None, torque_mode: str = "load",
          torque_controller=None, inverters=None, progress: str | None = None) -> DriveResult:
    """Run until the driver is done or t_max.

    The driver's pedal maps linearly to a torque request (0..motor peak), as the VCU's
    pedal map would; controller (e.g. kmslap.vcu.firmware.FirmwareController) then turns
    that into the inverter command, from per-wheel sensor edges, motor rpm and
    body-frame velocity / acceleration. Without one, the request goes straight to
    the inverter. Brake pedal maps to car.max_brake_torque, split by brake_bias_front.

    drs: a DRS policy (e.g. driver.DrsPolicy), called with the driver's view and
    inputs at the driver rate, returning the wing position it wants (0 closed, 1
    open); the wing moves there at its actuator rate. None: wing closed.

    regen: a driver.RegenPolicy turns regen on: off throttle, its negative torque
    command goes to the inverter (straight from the policy; the C++ controller only
    handles drive torque so far). None: no regen, the model is motoring-only.

    pack_backend: the pack in the loop (kmslap.hil): every pack_backend.period s of sim
    time it gets the current the car draws and returns the pack's open-circuit voltage,
    resistance, state of charge and temperature, which replace the model's own. None:
    the model's pack.
    realtime: run no faster than the wall clock (a real cell in the loop).
    torque_mode / torque_controller: a car with hub motors (car/drivetrain.py) commands
    every motor through vcu/torque_control.TorqueController in this mode ("fixed",
    "load", "yaw", "qp", "qp_lc"), or through the controller given (e.g. a ModeManager).
    It sees the IMU's (filtered) ax / ay, and the model's wheel loads as load cells.
    inverters: the hub motors' commands through their inverters (vcu/inverter.py):
    "native" (DTI current mode, AMK speed mode with torque limits, as the car's hub
    motors) or an Inverters. None: the torque asked for goes straight to the motors.
    progress: a description for a progress line in the terminal (kmslap.progress): by
    laps for a driver that counts them, else by time. None: none.
    on_tick(t, state, out, inputs): called at the driver rate (state / out by
    twotrack.STATES / OUTPUTS, inputs a dict of the driver's and the command); a true
    return stops the run.
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
                       relaxation_length_lat=relaxation_length_lat, dt=dt, regen=regen),
        vehicle.initial_state(car, pack, vx=vx0, soc=soc0, pack_temp=pack_temp0, X=X0, Y=Y0,
                              psi=psi0), hub_kind=car.hub_motor)

    S, O, P = vehicle.STATES, vehicle.OUTPUTS, vehicle.PARAMS
    if pack_backend is not None:
        ext = pack_backend.start(pack, soc0, pack.ambient_c if pack_temp0 is None else pack_temp0)
        plant.set_param(P.index("pack_ext"), 1.0)
        plant.set_param(P.index("pack_ext_v_oc"), ext["v_oc"])
        plant.set_param(P.index("pack_ext_r"), ext["r"])
    i_ext_v, i_ext_r = P.index("pack_ext_v_oc"), P.index("pack_ext_r")
    isoc, itemp, ibus = S.index("soc"), S.index("pack_temp"), S.index("bus_v")
    icur = O.index("bus_i")
    next_pack = next_pace = 0.0
    # in real time, keep pace at the pack's rate (a rig expects its setpoints evenly)
    pace_period = pack_backend.period if pack_backend is not None else driver_period
    wall0 = time.perf_counter()
    lateness = 0.0
    iX, iY, ipsi, ivx, ivy, ir = (S.index(k) for k in ("X", "Y", "psi", "vx", "vy", "r"))
    iw = [S.index(f"omega_{w}") for w in vehicle.WHEELS]
    iax, iay, itq = O.index("ax"), O.index("ay"), O.index("motor_torque")
    rpm_per_rads = 60 / (2 * math.pi)
    gr = car.final_drive

    state, out = plant.state, [0.0] * len(O)
    hub_cmds = vcu_hub_cmds = (0.0, 0.0, 0.0, 0.0)   # to the motors; what the VCU asked
    # hub motors: the torque split gives every motor its command, at the driver rate
    split = torque_controller
    if split is None and (car.hub_front or car.hub_rear):
        from kmslap.vcu.torque_control import TorqueController
        split = TorqueController(car, motor, torque_mode, regen, period=driver_period,
                                 slip_control=controller is None)
    if split is not None:
        from kmslap.vcu.torque_control import VehicleState
        central_cmd = 0.0
        iaxf, iayf, ibusv = S.index("ax_filt"), S.index("ay_filt"), S.index("bus_v")
        ifz = [O.index(f"fz_{w}") for w in vehicle.WHEELS]
    if inverters == "native":
        from kmslap.vcu.inverter import Inverters
        inverters = Inverters(car)
    if inverters is not None:
        hub_rpm = car.hub_ratio * rpm_per_rads                  # motor rpm per wheel rad/s
    pedal = brake = sw = 0.0
    wing = wing_target = 0.0
    wing_rate = dt / drs.travel_time if drs is not None else 0.0
    t, next_driver, next_log = 0.0, 0.0, 0.0
    rows = []
    next_tick = 0.0
    with _progress(progress, driver, t_max) as tick:
        for _ in range(int(t_max / dt)):
            if t >= next_tick:
                next_tick += 0.5
                tick(t)
            if realtime and t >= next_pace - 1e-12:
                next_pace += pace_period
                ahead = wall0 + t - time.perf_counter()
                if ahead > 0.002:
                    time.sleep(ahead - 0.001)
                while time.perf_counter() < wall0 + t:
                    pass
                lateness = max(-ahead, 0.0)
            if pack_backend is not None and t >= next_pack - 1e-12:
                next_pack += pack_backend.period
                ext = pack_backend.exchange(t, out[icur], state[ibus])
                plant.set_param(i_ext_v, ext["v_oc"])
                plant.set_param(i_ext_r, ext["r"])
                plant.set_state(isoc, ext["soc"])
                plant.set_state(itemp, ext["temp"])
                state = plant.state
            if t >= next_driver - 1e-12:
                next_driver += driver_period
                obs = {"X": state[iX], "Y": state[iY], "psi": state[ipsi], "vx": state[ivx],
                       "vy": state[ivy], "r": state[ir], "ax": out[iax], "ay": out[iay]}
                pedal, brake, sw = driver(t, obs)
                if driver.done:
                    break
                if on_tick is not None and on_tick(t, state, out, {"pedal": pedal, "brake": brake,
                                                                    "steering_wheel": sw, "drs": wing,
                                                                    "lateness": lateness}):
                    break
                if drs is not None:
                    wing_target = drs(obs, pedal, brake, sw)
                if split is not None:
                    central_cmd, hub_cmds = split(VehicleState(
                        pedal, brake, sw / car.steer_ratio, tuple(state[i] for i in iw), state[ivx], state[ir],
                        state[iaxf], state[ibusv], ay=state[iayf], fz=tuple(out[i] for i in ifz),
                        plan=getattr(driver, "plan_ff", None)))
                    vcu_hub_cmds = hub_cmds
                    if inverters is not None:
                        # each wheel's ground speed: the car's, +- the yaw rate times its half track
                        half = (car.track_front / 2, -car.track_front / 2, car.track_rear / 2, -car.track_rear / 2)
                        inverters.command(t, hub_cmds, [(state[ivx] - state[ir] * h) / car.tire_radius * hub_rpm
                                                        for h in half])
            if regen is not None:
                # pedal map with regen at its bottom; the drive part goes through TC, the
                # regen part (and brake blending) straight to the inverter
                t_map = regen.pedal_torque(pedal, motor.peak_torque)
                t_req = max(t_map, 0.0)
            else:
                t_req = pedal * motor.peak_torque
            if split is not None:
                # the split's share for the EMRAX (if there is one); TC on its drive torque
                t_req = max(central_cmd, 0.0)
            if controller is not None and (split is None or t_req > 0):
                omegas = [state[i] for i in iw]
                motor_rpm = 0.5 * (omegas[2] + omegas[3]) * gr * rpm_per_rads
                t_cmd = controller.step(t, dt, omegas, motor_rpm, state[ivx], out[iax],
                                        out[itq], t_req)
            else:
                t_cmd = t_req
            if regen is not None and t_map <= 0.0 and split is None:
                t_cmd = max(t_map - regen.blend_torque(brake, car), -regen.max_torque)
                if regen.slip_control:
                    v = max(state[ivx], 1.0)
                    slip = min((state[i] * car.tire_radius - state[ivx]) / v for i in iw[2:])
                    t_cmd *= regen.slip_scale(slip, dt)
            if split is not None and central_cmd <= 0.0:
                t_cmd = max(central_cmd, -regen.max_torque) if regen is not None else 0.0
            wing += max(-wing_rate, min(wing_rate, wing_target - wing))
            if inverters is not None:
                hub_cmds = inverters(t, dt, [state[i] * hub_rpm for i in iw])
            hydraulic = regen.hydraulic(brake) if regen is not None else brake
            plant(t_cmd, sw / car.steer_ratio, hydraulic * car.max_brake_torque, wing, *hub_cmds)
            if t >= next_log - 1e-12:
                next_log += log_period
                rows.append([t, pedal, brake, sw, wing, t_req, t_cmd, *vcu_hub_cmds, *state, *plant.out])
            state, out = plant.state, plant.out
            t += dt

    df = pd.DataFrame(rows, columns=["t", "pedal", "brake", "steering_wheel", "drs", "torque_request",
                                     "torque_cmd", *[f"hub_cmd_{w}" for w in vehicle.WHEELS], *S, *O])
    df["speed"] = np.hypot(df.vx, df.vy)
    df["beta"] = np.arctan2(df.vy, np.maximum(df.vx, 0.1))       # body slip angle
    df["rear_speed_diff"] = (df.omega_rl - df.omega_rr) * car.tire_radius   # m/s, + left faster
    return DriveResult(df, controller.log_frame() if controller is not None else None)


@contextlib.contextmanager
def _progress(description, driver, t_max):
    """tick(t): the progress line's update (a no-op without a description)."""
    if description is None:
        yield lambda t: None
        return
    from kmslap import progress
    laps = getattr(driver, "laps", None)
    n = len(driver.path.s) if laps and hasattr(driver, "path") else 0
    with progress.task(description, total=laps if n else t_max) as task:
        def tick(t):
            if n:
                task.update(completed=min(driver.laps_done + driver._i / n, laps))
                task.note(f"lap {min(driver.laps_done + 1, laps)}/{laps}, {t:.0f} s")
            else:
                task.update(completed=t)
        yield tick


def demo(out: str | None = None):
    """Skidpad and a short course, plotted into `out`."""
    out = dirs.out("drive", out, "drive.png")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from kmslap.driving.paths import Path, skidpad
    from kmslap.driving.follower import PathFollower
    from kmslap.models.envelope import gg_envelope, speed_profile

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
    fig.savefig(out)
    for name, res in runs.items():
        d = res.df
        print(f"{name}: {d.t.iloc[-1]:.1f} s, peak ay {d.ay.abs().max() / 9.81:.2f} g, "
              f"pack {d.pack_temp.iloc[0]:.1f} -> {d.pack_temp.iloc[-1]:.2f} C, "
              f"SOC {100 * d.soc.iloc[0]:.1f} -> {100 * d.soc.iloc[-1]:.2f}%")

