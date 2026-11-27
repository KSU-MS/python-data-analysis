"""VCU-side policies the drive loop applies on top of the driver: when to open the
DRS wing, and how to regen."""
import math
from dataclasses import dataclass, field

from kmslap.car.params import Car

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
    # defaults from kmslap.strategy on the Michigan 2026 endurance line (placeholder
    # downforce): open from 8 m/s, close past 20 deg of steering; a lateral g limit
    # never helped once steering closes the wing
    min_speed: float = 8.0
    max_lateral: float = 0.9 * 9.81
    max_steering_wheel: float = math.radians(20)
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
      regen_travel          the brake pedal's first travel that's regen only, without
                            the hydraulics (FSAE T.3.2.4: up to 0.9): regen rising to
                            max_torque (hub cars: every motor's torque) over it, the
                            hydraulics over the rest, regen staying on. 0 = none: the
                            pedal is all hydraulic. Rear-only regen (the EMRAX alone)
                            brakes the rear axle only over that travel, so it's an
                            option there
      slip_control          regen traction control (the EMRAX: hub cars' controller has
                            its own): regen backs off as a rear wheel's slip goes past
                            slip_target, to none slip_band beyond it (wheel speeds
                            against the VectorNav's), filtered by slip_tau
    Below Car.regen_min_speed there's no regen, fading in up to regen_fade_speed (in
    the vehicle model).
    """
    lift_torque: float = 0.0
    brake_fraction: float = 0.0
    max_rear_brake_share: float = 0.55
    max_torque: float = 230.0
    regen_travel: float = 0.0
    slip_control: bool = False
    slip_target: float = 0.08
    slip_band: float = 0.06
    slip_tau: float = 0.02
    _k: float = field(default=1.0, init=False, repr=False)

    def slip_scale(self, slip: float, dt: float) -> float:
        """The share of the regen torque kept at this wheel slip (the rear's most
        negative), filtered; 1 without slip control."""
        if not self.slip_control:
            return 1.0
        target = min(max(1 - (-slip - self.slip_target) / self.slip_band, 0.0), 1.0)
        self._k += min(dt / self.slip_tau, 1.0) * (target - self._k)
        return self._k

    def hydraulic(self, brake: float) -> float:
        """The hydraulic brakes' share of their full pressure at brake pedal travel 0..1."""
        if self.regen_travel <= 0:
            return brake
        return max(brake - self.regen_travel, 0.0) / (1 - self.regen_travel)

    def travel_fraction(self, brake: float) -> float:
        """The regen asked for by the pedal's regen travel, as a fraction of the most."""
        return min(brake / self.regen_travel, 1.0) if self.regen_travel > 0 else 0.0

    def pedal_torque(self, pedal: float, peak: float) -> float:
        """The pedal map with regen at its bottom (Nm at the motor, + drive)."""
        return pedal * (peak + self.lift_torque) - self.lift_torque

    def blend_torque(self, brake: float, car: Car) -> float:
        """Brake blending regen magnitude (Nm, motor side) for brake pedal 0..1."""
        friction = self.hydraulic(brake) * car.max_brake_torque
        rear = friction * (1 - car.brake_bias_front)
        blend = self.brake_fraction * friction
        if friction > 0:
            s = self.max_rear_brake_share
            blend = min(blend, max(0.0, (s * friction - rear) / (1 - s)))
        # wheel torque -> motor torque through the final drive (regen side of the chain)
        return blend * car.driveline_eff / car.final_drive + self.travel_fraction(brake) * self.max_torque
