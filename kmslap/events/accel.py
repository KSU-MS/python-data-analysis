"""FSAE acceleration event simulation.

States: distance x, speed v, rear wheel speed omega, rear tire transient slip kappa,
filtered acceleration (for load transfer), bus voltage.

    J_eff * domega/dt = T_motor * gr * eta - Fx * r
    m_eff * dv/dt     = Fx - drag - rolling
    sigma * dkappa/dt = (omega*r - v) - |v| * kappa      (tire relaxation length)
    Fx                = 2 * tire.fx(kappa, Fz_rear / 2)   (locked rear axle)

J_eff includes the motor rotor reflected through the final drive (J_rotor * gr^2),
which is the same order as the wheels themselves. The relaxation-length slip state
keeps the tire force well defined at v = 0, where slip = (omega*r - v)/v is not.

The plant step is defined once in CasADi (accel/model.py) and compiled to C; this
module runs the loop around it: driver request, traction control or the VCU
controller, logging.

`quasi_static` is the joe-accel-style model: no wheel dynamics, force is
min(motor, peak grip). It's the ideal-traction-control upper bound.
"""
import math
from dataclasses import dataclass

import pandas as pd

from . import model
from .control import SlipTC
from .params import G, Car
from .powertrain import MotorMap, Pack
from .tire import default_tire

RPM_PER_RADS = 60 / (2 * math.pi)
STAGE = 0.3      # m, car is staged this far behind the start line
LENGTH = 75.0    # m, start line to finish line


@dataclass
class Result:
    df: pd.DataFrame
    label: str = ""
    controller_log: pd.DataFrame | None = None

    @property
    def accel_time(self) -> float:
        """Timed run: start line (x = 0.3 m) to finish (x = 75.3 m). NaN if not reached."""
        t = self.df["t"].to_numpy()
        x = self.df["x"].to_numpy()
        if x[-1] < STAGE + LENGTH:
            return float("nan")
        return _cross(t, x, STAGE + LENGTH) - _cross(t, x, STAGE)

    def summary(self) -> dict:
        df = self.df
        return {
            "label": self.label,
            "accel_time_s": round(float(self.accel_time), 3),
            "trap_kmh": round(float(df["v"].iloc[-1]) * 3.6, 1),
            "peak_accel_g": round(float(df["a"].max()) / G, 2),
            "max_slip": round(float(df["slip"].max()), 3),
            "min_bus_v": round(float(df["bus_v"].min()), 1),
        }


def _cross(t, x, target):
    for i in range(1, len(x)):
        if x[i] >= target:
            f = (target - x[i - 1]) / (x[i] - x[i - 1]) if x[i] > x[i - 1] else 0.0
            return t[i - 1] + f * (t[i] - t[i - 1])
    return float("nan")


def _powertrain_torque(car, motor, pack, t_cmd, omega_m, bus_v):
    """Motor torque after the motor map, 80 kW rule and pack limits.

    Returns (torque, bus_v, current)."""
    rpm = omega_m * RPM_PER_RADS
    torque = min(t_cmd, motor.max_torque(rpm, bus_v))
    if omega_m > 1e-6:
        torque = min(torque, car.power_limit * car.motor_eff / omega_m)
    p_req = torque * omega_m / car.motor_eff
    v, current, p_del = pack.solve(p_req)
    if p_del < p_req - 1e-9:
        torque = p_del * car.motor_eff / omega_m
    return max(torque, 0.0), v, current


def simulate(car: Car = None, motor: MotorMap | str = "228HV", pack: Pack | str = "cosmx600v",
             tire=None, tc: SlipTC | None = None, *, dt: float = 2e-4, t_max: float = 8.0,
             ramp_time: float = 0.1, torque_request=None, controller=None,
             relaxation_length: float = 0.15, slip_speed_floor: float = 1.0,
             label: str = "") -> Result:
    """Run one accel event.

    ramp_time: driver torque request ramps 0 -> peak over this long (KS5e drivers
        reached full pedal in 0.06-0.1 s).
    torque_request: optional f(t, v) -> Nm replacing the ramp, e.g. a launch map
        from accel.launch. Still capped by the motor, pack, and TC.
    controller: e.g. accel.firmware.FirmwareController; replaces `tc`. Runs the VCU
        code on modeled sensors, its log ends up in Result.controller_log.
    slip_speed_floor: below this speed (m/s) the measured slip fed to TC uses this
        as the denominator, like a real TC does to avoid dividing by ~0.
    """
    car = car or Car()
    motor = MotorMap(motor) if isinstance(motor, str) else motor
    pack = Pack(_pack_by_name(pack)) if isinstance(pack, str) else pack
    tire = tire or default_tire()
    request = torque_request or _ramp(motor.peak_torque, ramp_time)
    if tc is not None:
        tc.reset()
    if controller is not None:
        controller.reset(car, motor)

    r = car.tire_radius
    stop_x = STAGE + LENGTH + 0.5
    plant = model.accel_step(motor, tire, model.params(car, motor, pack, tire, relaxation_length, dt),
                               model.initial_state(pack))

    t = accel = torque = 0.0
    state = plant.state
    rows = []
    append = rows.append
    n = int(t_max / dt)
    for _ in range(n):
        x, v, omega, kappa = state[0], state[1], state[2], state[3]
        slip = (omega * r - v) / max(v, slip_speed_floor)
        t_req = min(request(t, v), motor.peak_torque)
        if controller is not None:
            t_cmd = controller(t, dt, v, omega, accel, torque, t_req)
            scale = t_cmd / t_req if t_req > 0 else 1.0
        else:
            scale = tc.update(t, slip) if tc is not None else 1.0
            t_cmd = t_req * scale

        plant(t_cmd)
        accel, fx, fz_r, fz_f, torque, current, drag, rpm = plant.out
        state = plant.state
        append((t, x, v, accel, omega * r, slip, kappa, fx, fz_r, fz_f,
                torque, t_req, scale, rpm, state[6], current, drag))
        t += dt
        if state[0] > stop_x:
            break

    df = pd.DataFrame(rows, columns=[
        "t", "x", "v", "a", "wheel_speed", "slip", "kappa", "fx", "fz_rear", "fz_front",
        "motor_torque", "torque_request", "tc_scale", "motor_rpm", "bus_v", "bus_i", "drag"])
    res = Result(df, label or ("TC" if tc else f"VCU {controller.mode}" if controller else "no TC"))
    if controller is not None:
        res.controller_log = controller.log_frame()
    return res


def quasi_static(car: Car = None, motor: MotorMap | str = "228HV", pack: Pack | str = "cosmx600v",
                 tire=None, *, dt: float = 2e-4, t_max: float = 8.0, ramp_time: float = 0.1,
                 grip_margin: float = 1.0, label: str = "ideal TC (quasi-static)") -> Result:
    """No wheel slip dynamics: drive force = min(motor force, peak tire grip).

    Equivalent to a perfect traction controller; the upper bound for `simulate`.
    motor_torque is the torque actually used: when grip-limited, what it takes to
    hold the tires at peak (times grip_margin) and spin up the driveline with the car.
    """
    car = car or Car()
    motor = MotorMap(motor) if isinstance(motor, str) else motor
    pack = Pack(_pack_by_name(pack)) if isinstance(pack, str) else pack
    tire = tire or default_tire()

    r, gr, m = car.tire_radius, car.final_drive, car.mass
    eta = car.driveline_eff
    j_rear = 2 * car.wheel_inertia + car.rear_driveline_inertia + motor.rotor_inertia * gr * gr
    m_front = m + 2 * car.wheel_inertia / (r * r)
    m_all = m_front + j_rear / (r * r)
    stop_x = STAGE + LENGTH + 0.5

    t = x = v = a = 0.0
    bus_v = pack.v_oc
    rows = []
    for _ in range(int(t_max / dt)):
        omega_m = v / r * gr
        t_req = motor.peak_torque * (min(t / ramp_time, 1.0) if ramp_time > 0 else 1.0)
        torque, bus_v, current = _powertrain_torque(car, motor, pack, t_req, omega_m, bus_v)
        drag = car.drag(v)
        rolling = car.crr * car.total_load(v) if v > 0 or torque > 0 else 0.0
        a_motor = (torque * gr * eta / r - drag - rolling) / m_all
        a_grip, fx_grip = grip_limited_accel(car, tire, v, j_rear, a, grip_margin)
        a = min(a_motor, a_grip)
        fz_r = car.rear_load(a, v)
        if a_grip <= a_motor:
            # J * domega/dt = T * gr * eta - Fx * r, with domega/dt = a / r
            torque = (fx_grip * r + j_rear * a / r) / (gr * eta)
            torque, bus_v, current = _powertrain_torque(car, motor, pack, torque, omega_m, bus_v)
        rows.append((t, x, v, a, torque, t_req, omega_m * RPM_PER_RADS, bus_v, current,
                     fz_r, a_grip <= a_motor))
        v += a * dt
        x += v * dt
        t += dt
        if x > stop_x:
            break

    df = pd.DataFrame(rows, columns=["t", "x", "v", "a", "motor_torque", "torque_request",
                                     "motor_rpm", "bus_v", "bus_i", "fz_rear", "grip_limited"])
    df["slip"] = float("nan")
    return Result(df, label)


def grip_limited_accel(car, tire, v, j_rear, a_guess=0.0, grip_margin=1.0):
    """Acceleration (and rear axle Fx) with the rear tires at peak grip * grip_margin.

    Rear load depends on acceleration (load transfer), so fixed-point iterate.
    """
    m_front = car.mass + 2 * car.wheel_inertia / (car.tire_radius ** 2)
    drag = car.drag(v)
    rolling = car.crr * car.total_load(v)
    a = a_guess
    for _ in range(6):
        fz_r = car.rear_load(a, v)
        fx = 2.0 * grip_margin * tire.peak(0.5 * fz_r)[1] * 0.5 * fz_r
        a = (fx - drag - rolling) / m_front
    return a, fx


def _ramp(peak, ramp_time):
    if ramp_time <= 0:
        return lambda t, v: peak
    return lambda t, v: peak * min(t / ramp_time, 1.0)


def _pack_by_name(name):
    from .powertrain import PACKS
    return PACKS[name]
