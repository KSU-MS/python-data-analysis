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

`quasi_static` is the joe-accel-style model: no wheel dynamics, force is
min(motor, peak grip). It's the ideal-traction-control upper bound.
"""
import math
from dataclasses import dataclass

import pandas as pd

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
             ramp_time: float = 0.1, torque_request=None, relaxation_length: float = 0.15,
             slip_speed_floor: float = 1.0, label: str = "") -> Result:
    """Run one accel event.

    ramp_time: driver torque request ramps 0 -> peak over this long (KS5e drivers
        reached full pedal in 0.06-0.1 s).
    torque_request: optional f(t, v) -> Nm replacing the ramp, e.g. a launch map
        from accel.launch. Still capped by the motor, pack, and TC.
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

    r, gr, m = car.tire_radius, car.final_drive, car.mass
    eta = car.driveline_eff
    j_rear = 2 * car.wheel_inertia + car.rear_driveline_inertia + motor.rotor_inertia * gr * gr
    m_eff = m + 2 * car.wheel_inertia / (r * r)   # front wheels spin up with the car
    sigma = relaxation_length
    stop_x = STAGE + LENGTH + 0.5

    t = x = v = omega = kappa = a_filt = t_lagged = 0.0
    bus_v = pack.v_oc
    rows = []
    append = rows.append
    n = int(t_max / dt)
    for _ in range(n):
        fz_r = car.rear_load(a_filt, v)
        fz_tot = car.total_load(v)
        fx = 2.0 * tire.fx(kappa, 0.5 * fz_r)

        slip = (omega * r - v) / max(v, slip_speed_floor)
        t_req = min(request(t, v), motor.peak_torque)
        scale = tc.update(t, slip) if tc is not None else 1.0
        if car.torque_tau > 0:   # inverter torque response
            t_lagged += dt * (t_req * scale - t_lagged) / car.torque_tau
        else:
            t_lagged = t_req * scale
        omega_m = omega * gr
        torque, bus_v, current = _powertrain_torque(car, motor, pack, t_lagged, omega_m, bus_v)

        drag = car.drag(v)
        rolling = car.crr * fz_tot if v > 0 or fx > 0 else 0.0
        accel = (fx - drag - rolling) / m_eff
        omega_dot = (torque * gr * eta - fx * r) / j_rear

        append((t, x, v, accel, omega * r, slip, kappa, fx, fz_r, fz_tot - fz_r,
                torque, t_req, scale, omega_m * RPM_PER_RADS, bus_v, current, drag))

        # semi-implicit Euler
        v = max(v + accel * dt, 0.0)
        omega = max(omega + omega_dot * dt, 0.0)
        kappa += dt * ((omega * r - v) - abs(v) * kappa) / sigma
        x += v * dt
        if car.pitch_tau > 0:
            a_filt += dt * (accel - a_filt) / car.pitch_tau
        else:
            a_filt = accel
        t += dt
        if x > stop_x:
            break

    df = pd.DataFrame(rows, columns=[
        "t", "x", "v", "a", "wheel_speed", "slip", "kappa", "fx", "fz_rear", "fz_front",
        "motor_torque", "torque_request", "tc_scale", "motor_rpm", "bus_v", "bus_i", "drag"])
    return Result(df, label or ("TC" if tc else "no TC"))


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
