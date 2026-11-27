"""Torque split for cars with hub motors (car/drivetrain.py): the driver's pedals to a
torque command for each motor.

Drive: the pedal asks for a fraction of all the wheel torque the motors could give at
their speeds; it's shared between the axles by their load (the weight on each, with
the load transfer of the car's acceleration and the downforce), and what one axle
can't take goes to the other. Regen (with a RegenPolicy: off the pedal and blended with
the brakes, as wheel torque) is shared the same way, so the front hubs take most of it
under braking. Within an axle, left and right get the same (no torque vectoring yet).
The car model holds the total to the 80 kW and the pack's charge limit.

Slip limiting (tc, on by default): each wheel's slip against its own ground speed (the
body's, plus the yaw rate times where the wheel is); past `slip_target` its motor's
command is cut, smoothly over `slip_band`, both ways (wheelspin, and lock-up under
regen). The EMRAX goes by the worse rear wheel. A stand-in for a per-wheel traction
controller: the VCU's handles the one rear motor (vcu/firmware.py).
"""
import math

from kmslap.car.drivetrain import HubMotor
from kmslap.car.params import G, Car

WHEELS = ("fl", "fr", "rl", "rr")


class TorqueSplit:
    def __init__(self, car: Car, motor, regen=None, *, tc: bool = True, slip_target: float = 0.10,
                 slip_band: float = 0.10, tc_tau: float = 0.02, period: float = 0.01):
        self.car, self.motor, self.regen = car, motor, regen
        self.tc, self.slip_target, self.slip_band = tc, slip_target, slip_band
        self.alpha = min(1.0, period / tc_tau)
        self.k = dict.fromkeys(WHEELS, 1.0)
        self.hub = HubMotor(car.hub_motor)
        self.hubs = {"fl": car.hub_front, "fr": car.hub_front, "rl": car.hub_rear, "rr": car.hub_rear}

    def _hub_wheel_cap(self, omega):
        c = self.car
        return float(self.hub.max_torque(abs(omega) * c.hub_ratio * 60 / (2 * math.pi))) * c.hub_ratio * c.hub_gear_eff

    def front_share(self, vx: float, ax: float) -> float:
        c = self.car
        down = 0.5 * c.rho * vx * vx * c.cla
        lr = c.front_weight * c.wheelbase
        front = c.mass * G * lr / c.wheelbase - c.mass * ax * c.cg_height / c.wheelbase + down * c.aero_front
        return min(max(front / (c.mass * G + down), 0.0), 1.0)

    def _slip_factors(self, om, vx, yaw):
        c = self.car
        side = {"fl": c.track_front / 2, "fr": -c.track_front / 2, "rl": c.track_rear / 2, "rr": -c.track_rear / 2}
        for w in WHEELS:
            v = vx - yaw * side[w]
            slip = (om[w] * c.tire_radius - v) / max(abs(v), 1.0)
            target = min(max(1 - (abs(slip) - self.slip_target) / self.slip_band, 0.0), 1.0)
            self.k[w] += self.alpha * (target - self.k[w])

    def __call__(self, pedal: float, brake: float, omegas, vx: float, ax: float, bus_v: float, yaw: float = 0.0):
        """(central motor command, (hub commands fl, fr, rl, rr)), N m at the motors;
        negative = regen."""
        c = self.car
        om = dict(zip(WHEELS, omegas))
        if self.tc:
            self._slip_factors(om, vx, yaw)
        hub_cap = {w: self._hub_wheel_cap(om[w]) if self.hubs[w] else 0.0 for w in WHEELS}
        rear_rpm = 0.5 * (om["rl"] + om["rr"]) * c.final_drive * 60 / (2 * math.pi)
        central_cap = (self.motor.max_torque(rear_rpm, bus_v) * c.final_drive * c.driveline_eff
                       if c.central_drive else 0.0)
        cap_f = hub_cap["fl"] + hub_cap["fr"]
        cap_r = hub_cap["rl"] + hub_cap["rr"] + central_cap
        f = self.front_share(vx, ax)
        # wheel torque asked for: + drive, - regen
        if self.regen is not None:
            t_map = self.regen.pedal_torque(pedal, self.motor.peak_torque)
            drive = max(t_map, 0.0) / self.motor.peak_torque * (cap_f + cap_r)
            lift = min(t_map, 0.0) * c.final_drive / c.driveline_eff           # EMRAX-equivalent at the wheels
            regen = lift - self.regen.brake_fraction * brake * c.max_brake_torque
            regen = max(regen, -self.regen.max_torque * c.final_drive / c.driveline_eff) if not cap_f else regen
            ask = drive if drive > 0 else regen
        else:
            ask = pedal * (cap_f + cap_r)
        t_f = math.copysign(min(abs(f * ask), cap_f), ask)
        t_r = math.copysign(min(abs(ask - t_f), cap_r), ask)
        t_f = math.copysign(min(abs(ask - t_r), cap_f), ask)              # what the rear couldn't take
        cmds = {}
        for w, axle_t, axle_cap in (("fl", t_f, cap_f), ("fr", t_f, cap_f), ("rl", t_r, cap_r), ("rr", t_r, cap_r)):
            if not self.hubs[w]:
                cmds[w] = 0.0
                continue
            share = hub_cap[w] / axle_cap if axle_cap > 0 else 0.0
            wheel = axle_t * share
            gear = c.hub_gear_eff if wheel >= 0 else 1 / c.hub_gear_eff
            cmds[w] = wheel / (c.hub_ratio * gear) * (self.k[w] if self.tc else 1.0)
        central = 0.0
        if c.central_drive and cap_r > 0:
            wheel = t_r * central_cap / cap_r
            central = wheel / (c.final_drive * (c.driveline_eff if wheel >= 0 else 1 / c.driveline_eff))
            if self.tc:
                central *= min(self.k["rl"], self.k["rr"])
        return central, (cmds["fl"], cmds["fr"], cmds["rl"], cmds["rr"])
