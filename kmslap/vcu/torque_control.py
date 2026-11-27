"""Torque control for cars with more than one motor (car/drivetrain.py): a command for
every motor, in modes, prototyped here before the C++ controller (controller/) gets them,
as the single-motor traction control was.

    ctl = TorqueController(car, motor, mode="yaw")      # or "fixed", "load"
    central, (fl, fr, rl, rr) = ctl(VehicleState(...))  # N m at each motor, - = regen

Modes (the longitudinal split, then what's added on top):
  fixed   a constant front / rear split (front_bias), equal left / right
  load    shared by axle load (weight, load transfer from the acceleration, downforce)
  yaw     load, plus torque vectoring: a yaw rate target from the steering and speed
          (steady-state bicycle model with understeer gradient `k_us`, capped at what
          the grip gives, mu_yaw g / v), a PI on its error gives a yaw moment, made by
          left / right torque differences on the axles that can (each hub-driven axle;
          not one behind the diff)

Every mode then goes through: each motor's limits at its speed; a per-wheel slip limit
(slip against the wheel's own ground speed; both ways); the power limit, all motors'
electrical power (their efficiency maps) within car.power_limit, scaled together.

ModeManager switches modes the way HyTech's drivebrain does: only when the car is slow
and off the pedal, and the new mode's output is within limits.
"""
import math
from dataclasses import dataclass

import numpy as np

from kmslap.car.drivetrain import HubMotor
from kmslap.car.params import G, Car

WHEELS = ("fl", "fr", "rl", "rr")
MODES = ("fixed", "load", "yaw")


@dataclass
class VehicleState:
    """What the controller sees (the VCU's sensors: wheel speeds, IMU, steering, bus)."""
    pedal: float               # 0..1
    brake: float               # 0..1
    steer: float               # road wheel angle, rad, + left
    omegas: tuple              # wheel speeds fl, fr, rl, rr, rad/s
    vx: float                  # m/s (the VectorNav's)
    yaw_rate: float            # rad/s, + left
    ax: float                  # m/s^2, body frame (filtered)
    bus_v: float               # V


class TorqueController:
    def __init__(self, car: Car, motor, mode: str = "load", regen=None, *, front_bias: float = 0.45,
                 slip_target: float = 0.10, slip_band: float = 0.10, slip_tau: float = 0.02,
                 k_us: float = 0.0, mu_yaw: float = 1.3, yaw_kp: float = 800.0, yaw_ki: float = 2000.0,
                 period: float = 0.01, slip_control: bool = True):
        """k_us: rad/(m/s^2), the handling to aim for (0: neutral). yaw_kp / ki: N m of
        yaw moment per rad/s of yaw rate error (and its integral)."""
        if mode not in MODES:
            raise ValueError(f"mode {mode!r}: one of {MODES}")
        self.car, self.motor, self.mode, self.regen = car, motor, mode, regen
        self.front_bias, self.k_us, self.mu_yaw = front_bias, k_us, mu_yaw
        self.yaw_kp, self.yaw_ki, self.period = yaw_kp, yaw_ki, period
        self.slip_control, self.slip_target, self.slip_band = slip_control, slip_target, slip_band
        self.alpha = min(1.0, period / slip_tau)
        self.k = dict.fromkeys(WHEELS, 1.0)
        self.hub = HubMotor(car.hub_motor)
        self.hubs = {"fl": car.hub_front, "fr": car.hub_front, "rl": car.hub_rear, "rr": car.hub_rear}
        self.yaw_integral = 0.0
        self.log = {"yaw_ref": 0.0, "yaw_moment": 0.0}

    # --- what each motor can give, at the wheel
    def _hub_cap(self, omega):
        c = self.car
        return float(self.hub.max_torque(abs(omega) * c.hub_ratio * 60 / (2 * math.pi))) * c.hub_ratio * c.hub_gear_eff

    def _central_cap(self, om, bus_v):
        c = self.car
        if not c.central_drive:
            return 0.0
        rpm = 0.5 * (om["rl"] + om["rr"]) * c.final_drive * 60 / (2 * math.pi)
        return self.motor.max_torque(rpm, bus_v) * c.final_drive * c.driveline_eff

    def front_share(self, vx, ax):
        c = self.car
        if self.mode == "fixed":
            return self.front_bias
        down = 0.5 * c.rho * vx * vx * c.cla
        front = (c.mass * G * c.front_weight - c.mass * ax * c.cg_height / c.wheelbase + down * c.aero_front)
        return min(max(front / (c.mass * G + down), 0.0), 1.0)

    def _ask(self, s: VehicleState, cap_total):
        """Wheel torque the driver asks for: + drive, - regen."""
        c = self.car
        if self.regen is None:
            return s.pedal * cap_total
        t_map = self.regen.pedal_torque(s.pedal, self.motor.peak_torque)
        if t_map > 0:
            return t_map / self.motor.peak_torque * cap_total
        lift = t_map * c.final_drive / c.driveline_eff              # EMRAX-equivalent at the wheels
        return lift - self.regen.brake_fraction * s.brake * c.max_brake_torque

    def _yaw_moment(self, s: VehicleState):
        c = self.car
        v = max(s.vx, 1.0)
        ref = v * s.steer / (c.wheelbase + self.k_us * v * v)
        ref = min(max(ref, -self.mu_yaw * G / v), self.mu_yaw * G / v)
        err = ref - s.yaw_rate
        self.yaw_integral = min(max(self.yaw_integral + err * self.period, -0.5), 0.5)
        self.log["yaw_ref"] = ref
        return self.yaw_kp * err + self.yaw_ki * self.yaw_integral if s.vx > 3.0 else 0.0

    def _slip(self, om, s):
        c = self.car
        side = {"fl": c.track_front / 2, "fr": -c.track_front / 2, "rl": c.track_rear / 2, "rr": -c.track_rear / 2}
        for w in WHEELS:
            v = s.vx - s.yaw_rate * side[w]
            slip = (om[w] * c.tire_radius - v) / max(abs(v), 1.0)
            target = min(max(1 - (abs(slip) - self.slip_target) / self.slip_band, 0.0), 1.0)
            self.k[w] += self.alpha * (target - self.k[w])

    def __call__(self, s: VehicleState):
        """(central motor command, (hub commands fl, fr, rl, rr)), N m at the motors."""
        c = self.car
        om = dict(zip(WHEELS, s.omegas))
        if self.slip_control:
            self._slip(om, s)
        cap = {w: self._hub_cap(om[w]) if self.hubs[w] else 0.0 for w in WHEELS}
        central_cap = self._central_cap(om, s.bus_v)
        cap_f, cap_r = cap["fl"] + cap["fr"], cap["rl"] + cap["rr"] + central_cap
        ask = self._ask(s, cap_f + cap_r)
        f = self.front_share(s.vx, s.ax)
        t_f = math.copysign(min(abs(f * ask), cap_f), ask)
        t_r = math.copysign(min(abs(ask - t_f), cap_r), ask)
        t_f = math.copysign(min(abs(ask - t_r), cap_f), ask)
        # wheel torques: equal left / right, then the yaw moment on the axles that can
        wheel = {"fl": t_f / 2, "fr": t_f / 2, "rl": t_r / 2, "rr": t_r / 2}
        if c.central_drive and cap_r > 0:                   # the rear axle is the EMRAX + any rear hubs
            wheel["rl"] = wheel["rr"] = 0.5 * t_r * (cap["rl"] + cap["rr"]) / cap_r
        if self.mode == "yaw":
            mz = self._yaw_moment(s)
            self.log["yaw_moment"] = mz
            axles = [("fl", "fr", c.track_front)] if c.hub_front else []
            if c.hub_rear and not c.central_drive:
                axles.append(("rl", "rr", c.track_rear))
            for left, right, track in axles:
                # a torque difference dT between the wheels gives dT / r * track / 2 * 2
                dt = mz / len(axles) * c.tire_radius / track
                wheel[left] -= dt
                wheel[right] += dt
        cmds = {}
        for w in WHEELS:
            if not self.hubs[w]:
                cmds[w] = 0.0
                continue
            t = min(max(wheel[w], -cap[w]), cap[w])
            gear = c.hub_gear_eff if t >= 0 else 1 / c.hub_gear_eff
            cmds[w] = t / (c.hub_ratio * gear) * (self.k[w] if self.slip_control else 1.0)
        central = 0.0
        if c.central_drive and cap_r > 0:
            t = t_r * central_cap / cap_r
            central = t / (c.final_drive * (c.driveline_eff if t >= 0 else 1 / c.driveline_eff))
            if self.slip_control:
                central *= min(self.k["rl"], self.k["rr"])
        return central, (cmds["fl"], cmds["fr"], cmds["rl"], cmds["rr"])


class ModeManager:
    """Switches the active mode only when it's safe (as HyTech's drivebrain): every
    wheel below max_switch_speed, the pedal below max_switch_pedal, and the new mode's
    commands within the motors' limits. A request that can't happen now waits."""

    def __init__(self, controllers: dict, active: str, *, max_switch_speed: float = 5.0,
                 max_switch_pedal: float = 0.1):
        self.controllers, self.active = controllers, active
        self.max_switch_speed, self.max_switch_pedal = max_switch_speed, max_switch_pedal
        self.requested = active

    def request(self, mode: str):
        if mode not in self.controllers:
            raise ValueError(f"no mode {mode!r}")
        self.requested = mode

    def __call__(self, s: VehicleState):
        if self.requested != self.active:
            car = self.controllers[self.active].car
            slow = max(abs(w) for w in s.omegas) * car.tire_radius < self.max_switch_speed
            if slow and s.pedal < self.max_switch_pedal:
                self.active = self.requested
        return self.controllers[self.active](s)
