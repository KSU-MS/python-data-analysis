"""Vehicle parameters. Defaults are the current car where known (KS6e-ish, carried
over from accel_weight_transfer.ipynb); anything marked ESTIMATE or PLACEHOLDER has
not been measured. README "Car data to collect" lists everything still to fetch.

TODO(car data): replace the ESTIMATE / PLACEHOLDER values below with the current
car's (see the README list).
"""
import dataclasses
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

    # Aero, wing CLOSED (full downforce). The wing can be set open by hand
    # (open_aero()) or switched while driving with DRS (drs_* below, vehicle.py).
    # Drag: python-endurance-drag-energy-calculator calc_drag.ipynb, Cd 1.815 on
    # 1.146 m^2. Downforce: PLACEHOLDER, not measured on our car: that drag times the
    # L/D (2.4) of a published full-aero FSAE car (ClA 5.47 / CdA 2.28,
    # pidduuu.github.io "Justification for Aerodynamics", 2023). TODO(car data)
    cda: float = 1.815 * 1.146118   # drag coefficient * frontal area, m^2 (2.08)
    cla: float = 5.0                # lift coefficient * area, m^2 (positive = downforce)
    aero_front: float = 0.47        # fraction of downforce on the front axle, PLACEHOLDER
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

    # Aero with the wing OPEN (DRS, or set open by hand). None = same as closed.
    # Drag: calc_drag.ipynb, Cd 0.682 on 0.961 m^2. Downforce: PLACEHOLDER from
    # Monash's FSAE DRS study (front wing -37%, rear wing -67% downforce) applied to
    # a typical split of the closed ClA (front wing 35%, rear wing 40%, undertray
    # 25% unchanged): 5.0 * 0.60 = 3.0, balance moving forward. TODO(car data)
    drs_cda_open: float | None = 0.68167 * 0.9606706   # 0.655
    drs_cla_open: float | None = 3.0
    drs_aero_front_open: float | None = 0.57

    def open_aero(self) -> "Car":
        """The car with its wing set open by hand: the open aero as its only config
        (e.g. for the accel event, or a static low-drag endurance setup)."""
        return dataclasses.replace(
            self, cda=_or(self.drs_cda_open, self.cda), cla=_or(self.drs_cla_open, self.cla),
            aero_front=_or(self.drs_aero_front_open, self.aero_front),
            drs_cda_open=None, drs_cla_open=None, drs_aero_front_open=None)

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


def _or(value, default):
    return default if value is None else value
