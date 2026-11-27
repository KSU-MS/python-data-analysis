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
    final_drive: float = 4.18       # motor rev / wheel rev: the VCU's GEAR_RATIO (kms-vcu
                                    # include/parameters.hpp, since June 2025). kmslap assumed
                                    # 3.0 before; neither is measured yet. The report sweeps both
    driveline_eff: float = 0.95     # chain + bearings
    motor_eff: float = 0.90         # motor * inverter, battery -> shaft

    # Aero, wing CLOSED (full downforce). The wing can be set open by hand
    # (open_aero()) or switched while driving with DRS (drs_* below, vehicle model).
    # Measured on the KS7e aero tests of 6 July 2024 (same package as the current
    # car; `kmslap aero`, validation/aero.py): downforce from shock pots calibrated by
    # load transfer, ClA 1.69 with 45% on the front; drag from coast-downs, CdA
    # 1.2-1.45 (the absolute value is soft over the 7-13 m/s coasts; 1.30 is the
    # middle). Speeds only reached 19 m/s.
    cda: float = 1.30               # drag coefficient * frontal area, m^2
    cla: float = 1.69               # lift coefficient * area, m^2 (positive = downforce)
    aero_front: float = 0.45        # fraction of downforce on the front axle
    rho: float = 1.226              # air density, kg/m^3
    crr: float = 0.015              # rolling resistance coefficient, ESTIMATE
    # the plan-view footprint past the axles (cone hits). FSAE 2027 T.7.5 lets wings reach
    # 700 mm ahead of the front tires and 250 mm behind the rear; ours come within about
    # an inch of both (the user's recollection, not measured: ESTIMATE). Nothing is wider
    # than the tires (T.7.6), so the footprint is CAR_WIDTH wide
    overhang_front: float = 0.229 + 0.700 - 0.025   # m ahead of the front axle
    overhang_rear: float = 0.229 + 0.250 - 0.025    # m behind the rear axle

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

    # Regen hardware (used only when drive() is given a RegenPolicy). Motor
    # efficiency comes from the EMRAX map (powertrain.motor_efficiency_map).
    regen_inverter_eff: float = 0.97     # AC -> DC when generating, ESTIMATE
    regen_coulombic_eff: float = 0.99    # charge stored / charge pushed in, ESTIMATE
    regen_min_speed: float = 5 / 3.6     # m/s, no regen below (FSAE-style 5 km/h)
    regen_fade_speed: float = 8 / 3.6    # m/s, full regen above, linear fade between

    # Aero with the wing OPEN (DRS, or set open by hand). None = same as closed.
    # ESTIMATE (no DRS run in the tests): the measured package (ClA ~1.74, CdA ~0.41
    # over the bare car) split as typical, rear wing ~40% of its downforce and ~60% of
    # its drag, and opening removes ~2/3 of the rear wing's both (Monash / Princeton
    # FSAE DRS figures); the lost downforce comes off the rear axle. TODO(car data)
    drs_cda_open: float | None = 1.14
    drs_cla_open: float | None = 1.22
    drs_aero_front_open: float | None = 0.62

    # --- drivetrain (car/drivetrain.py: variants with hub motors, and their masses) ---
    central_drive: bool = True      # the EMRAX driving the rear wheels: chain + LSD
    hub_front: bool = False         # a hub motor at each front wheel
    hub_rear: bool = False          # ... at each rear wheel (with central_drive off: no diff)
    hub_motor: str = "dti"          # "dti" (F-SiC kit F-MOT) or "amk" (DD5)
    hub_ratio: float = 13.0         # hub gearbox reduction, motor rev / wheel rev
    hub_gear_eff: float = 0.97      # hub gearbox efficiency, ESTIMATE

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
