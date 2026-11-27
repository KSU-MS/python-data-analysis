"""Simulated drivers and the paths they follow.

A driver is called at its own rate with what it can see of the car and returns
(throttle pedal 0..1, brake pedal 0..1, steering wheel angle rad, + left):

    driver(t, obs) -> (pedal, brake, steering_wheel)

obs: X, Y, psi, vx, vy, r, ax, ay (world position / heading, body velocities).
"""
import math
from dataclasses import dataclass, field

import numpy as np

from .params import Car

SKIDPAD_RADIUS = 9.125   # m, FSAE skidpad driven line (15.25 m inner circle + 3 m lane)


@dataclass
class Path:
    """A driven line: points (x, y) with arc length s, heading (rad) and curvature k
    (1/m, + left)."""
    x: np.ndarray
    y: np.ndarray
    closed: bool = False
    s: np.ndarray = field(init=False)
    heading: np.ndarray = field(init=False)
    k: np.ndarray = field(init=False)

    def __post_init__(self):
        self.x, self.y = np.asarray(self.x, float), np.asarray(self.y, float)
        ds = np.hypot(np.diff(self.x), np.diff(self.y))
        self.s = np.concatenate([[0.0], np.cumsum(ds)])
        self.heading = np.unwrap(np.arctan2(np.gradient(self.y), np.gradient(self.x)))
        self.k = np.gradient(self.heading) / np.maximum(np.gradient(self.s), 1e-9)

    @property
    def length(self) -> float:
        return float(self.s[-1])

    @staticmethod
    def from_segments(segments, start=(0.0, 0.0), heading=0.0, step=0.25, closed=False):
        """segments: ("straight", length) or ("arc", radius, degrees; + left)."""
        x, y, h = [start[0]], [start[1]], heading
        for seg in segments:
            if seg[0] == "straight":
                n = max(int(seg[1] / step), 1)
                for _ in range(n):
                    x.append(x[-1] + seg[1] / n * math.cos(h))
                    y.append(y[-1] + seg[1] / n * math.sin(h))
            else:
                radius, sweep = seg[1], math.radians(seg[2])
                n = max(int(abs(sweep) * radius / step), 1)
                dh = sweep / n
                for _ in range(n):
                    h += dh
                    chord = 2 * radius * math.sin(abs(dh) / 2)
                    x.append(x[-1] + chord * math.cos(h - dh / 2))
                    y.append(y[-1] + chord * math.sin(h - dh / 2))
        return Path(np.array(x), np.array(y), closed=closed)


def skidpad(radius: float = SKIDPAD_RADIUS, laps: int = 2, direction: int = 1) -> Path:
    """Straight entry then `laps` circles (left for direction=1), starting tangent at
    the origin heading +x, like the skidpad's timed laps without the figure-eight."""
    return Path.from_segments([("straight", 15.0), ("arc", radius, 360.0 * laps * direction),
                               ("straight", 15.0)])


def speed_profile(path: Path, a_lat: float, a_accel: float, a_brake: float,
                  v_max: float = 35.0, v_start: float | None = None) -> np.ndarray:
    """Fastest speed along the path within lateral / longitudinal acceleration
    limits (forward-backward pass). A quasi-steady target for the driver, not a
    prediction: the car may not manage it."""
    v = np.minimum(v_max, np.sqrt(a_lat / np.maximum(np.abs(path.k), 1e-6)))
    ds = np.diff(path.s)
    if v_start is not None:
        v[0] = min(v[0], v_start)
    for i in range(1, len(v)):
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * a_accel * ds[i - 1]))
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * a_brake * ds[i]))
    return v


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

    Speed: a PI on speed error asks for an acceleration (within accel_max /
    brake_max), turned into pedal / brake by feedforward from the car's mass, drag
    and gearing. Like a driver, it lifts when the car slides (body slip angle past
    slide_lift), which is the only traction control it has.

    speed: target speed per path point, or a constant (m/s).
    Finishes (`done`) at the end of an open path.
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
    speed_ki: float = 0.5            # m/s^2 per (m/s * s)
    accel_max: float = 6.0           # m/s^2
    brake_max: float = 12.0          # m/s^2
    peak_torque: float = 230.0       # Nm at full pedal (the VCU pedal map)
    slide_lift: tuple = (math.radians(4), math.radians(10))   # body slip: start lifting, fully off
    done: bool = False
    _i: int = 0
    _integral: float = 0.0
    _lat_integral: float = 0.0
    _t_last: float | None = None

    def __post_init__(self):
        n = len(self.path.s)
        self._v_target = (np.full(n, float(self.speed)) if np.isscalar(self.speed)
                          else np.asarray(self.speed, float))

    def __call__(self, t, obs):
        p = self.path
        X, Y, psi, vx = obs["X"], obs["Y"], obs["psi"], obs["vx"]
        # nearest point, searching forward from the last one
        lo, hi = self._i, min(self._i + 200, len(p.s))
        d = np.hypot(p.x[lo:hi] - X, p.y[lo:hi] - Y)
        self._i = lo + int(np.argmin(d))
        if not p.closed and self._i >= len(p.s) - 2:
            self.done = True
        dt = 0.0 if self._t_last is None else t - self._t_last
        self._t_last = t
        i = self._i
        h = p.heading[i]
        e_lat = -math.sin(h) * (X - p.x[i]) + math.cos(h) * (Y - p.y[i])    # + left of the line
        e_head = (psi - h + math.pi) % (2 * math.pi) - math.pi
        j = int(np.searchsorted(p.s, p.s[i] + self.steer_preview * vx))
        j = j % len(p.s) if p.closed else min(j, len(p.s) - 1)
        delta_ff = self.car.wheelbase * p.k[j]
        fb = (self.k_lateral * e_lat + self.k_heading * e_head
              + self.k_yaw * (obs["r"] - vx * p.k[i]) + self._lat_integral)
        delta = delta_ff - fb
        sw = delta * self.car.steer_ratio
        if abs(sw) < self.max_steering_wheel:              # anti-windup
            self._lat_integral += self.k_lateral_i * e_lat * dt
        sw = max(-self.max_steering_wheel, min(self.max_steering_wheel, sw))
        k = int(np.searchsorted(p.s, p.s[self._i] + self.speed_preview * vx)) + 1
        err = float(self._v_target[self._i:max(k, self._i + 1)].min()) - vx
        a_des = self.speed_kp * err + self._integral
        if -self.brake_max < a_des < self.accel_max:     # anti-windup
            self._integral += self.speed_ki * err * dt
        a_des = min(max(self.speed_kp * err + self._integral, -self.brake_max), self.accel_max)

        c = self.car
        resist = c.drag(vx) + c.crr * c.total_load(vx)
        force = c.mass * a_des + resist
        pedal = max(0.0, force * c.tire_radius / (c.final_drive * c.driveline_eff * self.peak_torque))
        brake = max(0.0, -force * c.tire_radius / c.max_brake_torque)
        beta = abs(math.atan2(obs["vy"], max(vx, 0.5)))
        lo, hi = self.slide_lift
        pedal *= min(1.0, max(0.0, (hi - beta) / (hi - lo)))
        return min(pedal, 1.0), min(brake, 1.0), sw
