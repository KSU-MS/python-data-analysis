"""Simulated drivers. A driver is called at its own rate with what it can see of the
car and returns (throttle pedal 0..1, brake pedal 0..1, steering wheel angle rad, + left):

    driver(t, obs) -> (pedal, brake, steering_wheel)

obs: X, Y, psi, vx, vy, r, ax, ay (world position / heading, body velocities).
"""
import math
from dataclasses import dataclass, field

import numpy as np

from kmslap.car.params import Car
from kmslap.driving.paths import Path

@dataclass
class ScriptedDriver:
    """Open loop: pedal(t), brake(t), steering_wheel(t)."""
    pedal: object = lambda t: 1.0
    brake: object = lambda t: 0.0
    steering_wheel: object = lambda t: 0.0
    done: bool = False

    def __call__(self, t, obs):
        return self.pedal(t), self.brake(t), self.steering_wheel(t)


@dataclass
class PathFollower:
    """Path tracking steering; speed control to a target speed.

    Steering: road wheel angle = wheelbase * path curvature (feedforward, previewed)
    minus feedback on lateral offset (with a slow integral for the steady offset
    understeer leaves), heading error, and yaw rate vs the path's. The yaw rate term
    damps the car near the limit; pure pursuit, without it, set up yaw oscillations
    that spun the car well inside its grip.

    Speed: the target profile's own acceleration (v dv/ds, so braking zones start on
    time) plus a PI on speed error give an acceleration (within accel_max /
    brake_max), turned into pedal / brake by feedforward from the car's mass, drag
    and gearing. Like a driver, it lifts when the rear slides (rear axle slip angle
    past slide_lift; not body slip, which is large in a tight slow corner without any
    sliding), which is the only traction control it has.

    speed: target speed per path point, or a constant (m/s).
    laps: on a closed path, finish (`done`) after this many laps from the start
        point; an open path finishes at its end. `laps_done` counts them.
    coast_time: lift-and-coast. When the speed target drops within this many
        seconds ahead (a braking zone coming), lift off instead of holding speed;
        brakes still come on when needed. 0 = no coasting.
    """
    path: Path
    speed: object
    car: Car = field(default_factory=Car)
    steer_preview: float = 0.15      # s of travel for the curvature feedforward
    k_lateral: float = 0.08          # rad of road wheel per m off the path
    k_lateral_i: float = 0.05        # rad per (m * s)
    k_heading: float = 0.6           # rad per rad
    k_yaw: float = 0.06              # rad per rad/s of yaw rate error
    speed_preview: float = 0.3       # s: aim for the lowest target speed this far ahead
    max_steering_wheel: float = math.radians(120)
    speed_kp: float = 2.0            # m/s^2 asked per m/s of speed error
    # integral, only where the target is steady (|profile accel| < 0.5 m/s^2): it
    # takes up cornering drag the feedforward doesn't model, without winding up
    # through braking zones (which left the car slow into every corner)
    speed_ki: float = 1.0            # m/s^2 per (m/s * s)
    accel_max: float = 6.0           # m/s^2
    brake_max: float = 16.0          # m/s^2
    peak_torque: float = 230.0       # Nm at full pedal (the VCU pedal map)
    slide_lift: tuple = (math.radians(7), math.radians(12))   # rear slip angle: start lifting, fully off
    laps: int | None = None
    coast_time: float = 0.0
    # brake feel with regen blending: braking torque per unit pedal is this much more
    # than the friction brakes' (a driver adapts to the pedal). 0 = no blending.
    brake_regen_fraction: float = 0.0
    # pedal map with regen at its bottom (RegenPolicy.lift_torque, Nm): the driver's
    # feedforward inverts it. 0 = the plain map.
    pedal_regen_torque: float = 0.0
    done: bool = False
    laps_done: int = 0
    lap_times: list = field(default_factory=list)   # t at each start/finish crossing
    _i: int = 0
    _integral: float = 0.0
    _lat_integral: float = 0.0
    _t_last: float | None = None

    def __post_init__(self):
        n = len(self.path.s)
        self._v_target = (np.full(n, float(self.speed)) if np.isscalar(self.speed)
                          else np.asarray(self.speed, float))
        v = self._v_target
        nxt = np.roll(v, -1) if self.path.closed else np.append(v[1:], v[-1])
        ds = np.diff(self.path.s, append=self.path.s[0] + self.path.length if self.path.closed
                     else self.path.s[-1] + 1.0)
        self._a_target = (nxt ** 2 - v ** 2) / (2 * np.maximum(ds, 1e-6))

    def __call__(self, t, obs):
        p = self.path
        X, Y, psi, vx = obs["X"], obs["Y"], obs["psi"], obs["vx"]
        # nearest point, searching forward from the last one
        n = len(p.s)
        idx = (self._i + np.arange(200)) % n if p.closed else np.arange(self._i, min(self._i + 200, n))
        new = int(idx[np.argmin(np.hypot(p.x[idx] - X, p.y[idx] - Y))])
        if p.closed and new < self._i - n // 2:
            self.laps_done += 1
            self.lap_times.append(t)
            if self.laps is not None and self.laps_done >= self.laps:
                self.done = True
        self._i = new
        if not p.closed and self._i >= n - 2:
            self.done = True
        dt = 0.0 if self._t_last is None else t - self._t_last
        self._t_last = t
        i = self._i
        h = p.heading[i]
        e_lat = -math.sin(h) * (X - p.x[i]) + math.cos(h) * (Y - p.y[i])    # + left of the line
        e_head = (psi - h + math.pi) % (2 * math.pi) - math.pi
        j = p.ahead(i, self.steer_preview * vx)
        delta_ff = self.car.wheelbase * p.k[j]
        fb = (self.k_lateral * e_lat + self.k_heading * e_head
              + self.k_yaw * (obs["r"] - vx * p.k[i]) + self._lat_integral)
        delta = delta_ff - fb
        sw = delta * self.car.steer_ratio
        if abs(sw) < self.max_steering_wheel:              # anti-windup
            self._lat_integral += self.k_lateral_i * e_lat * dt
        sw = max(-self.max_steering_wheel, min(self.max_steering_wheel, sw))
        window = p.span(i, p.ahead(i, self.speed_preview * vx))
        target = float(self._v_target[window].min())
        a_ff = float(self._a_target[window[-1]])
        err = target - vx
        a_des = a_ff + self.speed_kp * err + self._integral
        if abs(a_ff) < 0.5 and -self.brake_max < a_des < self.accel_max:
            self._integral = min(max(self._integral + self.speed_ki * err * dt, -2.0), 4.0)
        a_des = min(max(a_ff + self.speed_kp * err + self._integral, -self.brake_max), self.accel_max)

        c = self.car
        resist = c.drag(vx) + c.crr * c.total_load(vx)
        force = c.mass * a_des + resist
        motor = force * c.tire_radius / (c.final_drive * c.driveline_eff)
        lift = self.pedal_regen_torque
        pedal = max(0.0, (motor + lift) / (self.peak_torque + lift))
        brake = max(0.0, -force * c.tire_radius / (c.max_brake_torque * (1 + self.brake_regen_fraction)))
        lr = c.front_weight * c.wheelbase
        rear_slip = abs(math.atan2(obs["vy"] - obs["r"] * lr, max(vx, 0.5)))
        lo, hi = self.slide_lift
        pedal *= min(1.0, max(0.0, (hi - rear_slip) / (hi - lo)))
        if self.coast_time > 0:
            ahead = self._v_target[p.span(i, p.ahead(i, self.coast_time * vx))]
            if ahead.min() < min(target, vx) - 0.5:
                pedal = 0.0
        return min(pedal, 1.0), min(brake, 1.0), sw
