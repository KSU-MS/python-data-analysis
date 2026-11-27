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

    # --- double-track (vehicle.py) only ---
    track_front: float = 1.22       # m, ESTIMATE
    track_rear: float = 1.20        # m, ESTIMATE
    yaw_inertia: float = 110.0      # kg m^2 about the CG, with driver, ESTIMATE
    roll_stiffness_front: float = 0.5   # front share of lateral load transfer, ESTIMATE
    roll_tau: float = 0.05          # s, lag on lateral load transfer (roll), ESTIMATE
    steer_ratio: float = 4.5        # steering wheel angle / road wheel angle, ESTIMATE

    # Drexler FSAE limited slip (clutch pack, ramp angles + preload, adjustable).
    # Locking torque = preload + lock * |input torque| in drive (lock_coast off
    # throttle); a bias ratio TBR is lock = (TBR - 1) / (TBR + 1). ESTIMATE all.
    diff_preload: float = 15.0      # Nm across the clutches
    diff_lock_drive: float = 0.4
    diff_lock_coast: float = 0.3
    # numerical: the clutches stick below this wheel speed difference. Too small
    # makes the step stiff at dt = 0.2 ms; 2 rad/s is ~0.5 m/s at the tire.
    diff_slip_eps: float = 2.0      # rad/s

    brake_bias_front: float = 0.65  # front share of brake torque, ESTIMATE
    max_brake_torque: float = 1500.0  # Nm summed over the wheels at full pedal, ESTIMATE

    # drag reduction system: aero with the wing open. None = same as closed (no DRS).
    # cda / cla / aero_front above are the closed (normal) wing.
    drs_cda_open: float | None = None
    drs_cla_open: float | None = None
    drs_aero_front_open: float | None = None

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
