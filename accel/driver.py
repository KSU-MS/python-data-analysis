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
        if self.closed:
            # a loop: the last point connects back to the first (don't repeat it)
            if np.hypot(self.x[-1] - self.x[0], self.y[-1] - self.y[0]) < 1e-6:
                self.x, self.y = self.x[:-1], self.y[:-1]
            dx = 0.5 * (np.roll(self.x, -1) - np.roll(self.x, 1))
            dy = 0.5 * (np.roll(self.y, -1) - np.roll(self.y, 1))
            ds = np.hypot(np.diff(self.x, append=self.x[0]), np.diff(self.y, append=self.y[0]))
            self._length = float(ds.sum())
            self.s = np.concatenate([[0.0], np.cumsum(ds[:-1])])
            self.heading = np.unwrap(np.arctan2(dy, dx))
            dh = (np.roll(self.heading, -1) - np.roll(self.heading, 1) + np.pi) % (2 * np.pi) - np.pi
            self.k = 0.5 * dh / np.maximum(np.hypot(dx, dy), 1e-9)
        else:
            ds = np.hypot(np.diff(self.x), np.diff(self.y))
            self.s = np.concatenate([[0.0], np.cumsum(ds)])
            self._length = float(self.s[-1])
            self.heading = np.unwrap(np.arctan2(np.gradient(self.y), np.gradient(self.x)))
            self.k = np.gradient(self.heading) / np.maximum(np.gradient(self.s), 1e-9)

    @property
    def length(self) -> float:
        return self._length

    def ahead(self, i: int, dist: float) -> int:
        """Index of the point `dist` metres past point i (wrapping on a loop)."""
        if self.closed:
            rem = (self.s[i] + dist) % self._length
            return int(np.searchsorted(self.s, rem)) % len(self.s)
        return min(int(np.searchsorted(self.s, self.s[i] + dist)), len(self.s) - 1)

    def span(self, i: int, j: int) -> np.ndarray:
        """Indices from i to j inclusive, going forward (wrapping on a loop)."""
        if j >= i:
            return np.arange(i, j + 1)
        return np.concatenate([np.arange(i, len(self.s)), np.arange(0, j + 1)])

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


@dataclass
class DrsPolicy:
    """When to open the wing, from signals the VCU has: speed, lateral g, steering
    wheel angle, pedal, brake. Open when fast, nearly straight and on the gas; close
    on the brake, when the driver turns in (steering leads lateral g, so this closes
    before the corner loads up), or when lateral g builds. Called by drive() at the
    driver rate.

    min_speed: m/s (the old calc_drag notebook opened above 20 m/s).
    max_lateral: m/s^2; above this the car needs the downforce.
    max_steering_wheel: rad; close when the driver steers more than this.
    min_pedal: open only while accelerating or holding speed.
    travel_time: s for the actuator to go fully open or closed.
    """
    min_speed: float = 12.0
    max_lateral: float = 0.3 * 9.81
    max_steering_wheel: float = math.radians(180)
    min_pedal: float = 0.05
    travel_time: float = 0.2

    def __call__(self, obs, pedal, brake, steering_wheel) -> float:
        open_ = (obs["vx"] > self.min_speed and abs(obs["ay"]) < self.max_lateral
                 and abs(steering_wheel) < self.max_steering_wheel
                 and pedal >= self.min_pedal and brake == 0)
        return 1.0 if open_ else 0.0


@dataclass
class RegenPolicy:
    """Regen strategy in the VCU (rear wheels only: it's the motor). Passing one to
    drive() turns regen on; without one the car never regens.

    The VCU commands negative motor torque:
      lift_torque           Nm at the motor with the pedal up: the bottom of the pedal
                            map (one-pedal style), torque rising linearly from
                            -lift_torque at pedal 0 to +peak at full pedal, so there's no
                            switch to chatter on. 0 = coast, today's map.
      brake_fraction        brake blending: regen at the rear wheels as this fraction of
                            the friction brake torque (front + rear pressure sensors
                            give it), on top of it (the hydraulics aren't by-wire)
      max_rear_brake_share  blending is capped so rear friction + regen stays at most
                            this share of all braking: regen is rear-only and shifts the
                            effective bias rearward (rear lock-up, spins)
      max_torque            Nm, the inverter's regen limit (it also respects current
                            limits; the model applies the pack's charge limits)
    Below Car.regen_min_speed there's no regen, fading in up to regen_fade_speed (in
    the vehicle model).
    """
    lift_torque: float = 0.0
    brake_fraction: float = 0.0
    max_rear_brake_share: float = 0.55
    max_torque: float = 230.0

    def pedal_torque(self, pedal: float, peak: float) -> float:
        """The pedal map with regen at its bottom (Nm at the motor, + drive)."""
        return pedal * (peak + self.lift_torque) - self.lift_torque

    def blend_torque(self, brake: float, car: Car) -> float:
        """Brake blending regen magnitude (Nm, motor side) for brake pedal 0..1."""
        friction = brake * car.max_brake_torque
        rear = friction * (1 - car.brake_bias_front)
        blend = self.brake_fraction * friction
        if friction > 0:
            s = self.max_rear_brake_share
            blend = min(blend, max(0.0, (s * friction - rear) / (1 - s)))
        # wheel torque -> motor torque through the final drive (regen side of the chain)
        return blend * car.driveline_eff / car.final_drive
