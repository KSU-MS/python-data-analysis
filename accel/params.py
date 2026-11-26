"""Vehicle parameters for the accel sim. Defaults are KS6e-ish, carried over from
accel_weight_transfer.ipynb; anything marked ESTIMATE has not been measured."""
from dataclasses import dataclass

G = 9.81


@dataclass
class Car:
    mass: float = 285.0             # kg, car + driver
    cg_height: float = 0.2751       # m
    wheelbase: float = 1.61544      # m
    front_weight: float = 0.5       # static fraction of weight on the front axle
    tire_radius: float = 0.229      # m, loaded radius
    final_drive: float = 3.0        # motor rev / wheel rev
    driveline_eff: float = 0.95     # chain + bearings
    motor_eff: float = 0.90         # motor * inverter, battery -> shaft

    cda: float = 0.7                # drag coefficient * frontal area, m^2
    cla: float = 0.8                # lift coefficient * area, m^2 (positive = downforce)
    aero_front: float = 0.5         # fraction of downforce on the front axle
    rho: float = 1.226              # air density, kg/m^3
    crr: float = 0.015              # rolling resistance coefficient, ESTIMATE

    wheel_inertia: float = 0.10     # kg m^2 per wheel+tire, ESTIMATE
    rear_driveline_inertia: float = 0.01  # sprocket/diff/axle on the wheel side, ESTIMATE

    # first-order lag on the acceleration used for longitudinal load transfer,
    # standing in for pitch/suspension dynamics. 0 = instantaneous.
    pitch_tau: float = 0.05         # s

    # inverter torque response (first-order lag); KS5e logs show 30-60 ms from
    # command to feedback, part of which is logging latency. ESTIMATE
    torque_tau: float = 0.03        # s

    power_limit: float = 80e3       # W at the accumulator (FSAE EV rule)

    @property
    def cg_to_front(self) -> float:
        """Distance from the front axle to the CG ("b")."""
        return (1 - self.front_weight) * self.wheelbase

    def rear_load(self, accel: float, v: float) -> float:
        """Rear axle normal force (N) at a given longitudinal acceleration and speed."""
        static = self.mass * G * self.cg_to_front / self.wheelbase
        transfer = self.mass * accel * self.cg_height / self.wheelbase
        downforce = 0.5 * self.rho * self.cla * v * v * (1 - self.aero_front)
        return static + transfer + downforce

    def total_load(self, v: float) -> float:
        return self.mass * G + 0.5 * self.rho * self.cla * v * v

    def drag(self, v: float) -> float:
        return 0.5 * self.rho * self.cda * v * v
