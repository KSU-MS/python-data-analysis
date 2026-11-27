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
  qp      every motor's torque at once from a small QP (after HyTech's drivebrain
          qp_torq_allocator), each control period: track the driver's torque and the
          yaw PI's moment, keep each wheel's slip off its peak (a linear tire, the
          slip term on only near the peak), change torques smoothly; within each
          wheel's grip, the motors' limits and the electrical power limit. Wheel loads
          from the IMU and speed (static weight, load transfer, downforce): the
          sensors the car has now. The yaw target is capped at the lateral g the
          tire tables give at those loads; when the car yaws past it (power
          oversteer), the EMRAX (no vectoring behind the diff) gives up drive
  qp_lc   qp with the wheel loads measured (load cells: VehicleState.fz)

fixed / load / yaw then go through: each motor's limits at its speed; a per-wheel slip
limit (slip against the wheel's own ground speed; both ways). The qp modes do their
own traction control (the slip term, and a slow torque recovery after it acted). All:
regen fades out below car.regen_fade_speed, none below car.regen_min_speed. The car
model holds the electrical power within car.power_limit.

The yaw PI's gains grow with speed above yaw_v_ref. Tuned with vcu/torque_bench.py
(`kmslap torque-bench`: accel, skidpad, a power-on sine steer).

ModeManager switches modes the way HyTech's drivebrain does: only when the car is slow
and off the pedal, and the new mode's output is within limits.
"""
import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from kmslap.car.drivetrain import HubMotor
from kmslap.car.params import G, Car

WHEELS = ("fl", "fr", "rl", "rr")
MODES = ("fixed", "load", "yaw", "qp", "qp_lc")
CENTRAL_EFF = 0.93          # EMRAX x its inverter, for the QP's power row: ESTIMATE


@lru_cache(maxsize=None)
def _qp_solver():
    """The QP's solver (qpOASES): 5 torques, 7 rows; built once, shared."""
    import casadi as ca
    return ca.conic("tq", "qpoases", {"h": ca.Sparsity.dense(5, 5), "a": ca.Sparsity.dense(7, 5)},
                    {"printLevel": "none", "error_on_fail": False})


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
    ay: float | None = None    # m/s^2, + left (filtered); None: vx * yaw_rate
    fz: tuple | None = None    # wheel loads fl, fr, rl, rr, N: load cells, if fitted


class TorqueController:
    def __init__(self, car: Car, motor, mode: str = "load", regen=None, *, front_bias: float = 0.45,
                 slip_target: float = 0.10, slip_band: float = 0.10, slip_tau: float = 0.02,
                 k_us: float = 0.0, mu_yaw: float = 1.3, yaw_kp: float = 4000.0, yaw_ki: float = 4000.0,
                 period: float = 0.01, slip_control: bool = True, tire=None, yaw_v_ref: float | None = 10.0,
                 qp_weights=(1.0, 4.0, 0.05, 0.2), qp_slip_weight: float = 20.0, kappa_frac: float = 1.0,
                 mu_frac: float = 1.0, inside_regen: float = 5.0, tc_rise_rate: float | None = None,
                 tc_hold: float = 0.1, ay_margin: float = 1.05,
                 friction_circle: bool = False, circle_floor: float = 0.2, oversteer_cut: float = 0.8,
                 launch_speed: float = 3.0):
        """k_us: rad/(m/s^2), the handling to aim for (0: neutral). yaw_kp / ki: N m of
        yaw moment per rad/s of yaw rate error (and its integral); yaw_v_ref: m/s
        above which they grow in proportion to speed (None: constant).

        qp modes. tire: the tire the controller's calibration tables come from (each
        wheel's peak grip and the slip at it, against load; default_tire()).
        qp_weights: (driver torque, yaw moment, torque change, split) weights, on
        errors normalised by the ask / 500 N m; split pulls each wheel to its share
        of the last total by grip (the load mode's split), which otherwise drifts when
        only the total matters (power limited). qp_slip_weight: on (predicted slip -
        kappa_frac x the peak slip)^2 / 0.01. mu_frac: the share of the table's peak
        grip a wheel may use. inside_regen: N m at a motor a wheel may regen while the
        car drives, to turn it. tc_rise_rate: N m/s at a wheel, how fast torque comes
        back for tc_hold s after the slip term acted (HyTech: 800; None: off, which
        is faster in the sim's accel). launch_speed: m/s, the qp modes' slip is
        against at least this ground speed (at launch, a slip speed, not a ratio).
        ay_margin: the qp modes cap the yaw rate target at ay_margin x the lateral
        acceleration the tables give at the wheels' loads (not mu_yaw).
        friction_circle: a wheel's drive / brake grip is what's left of its peak
        after its share of the sideways force (m ay, by load), at least circle_floor
        of the peak. oversteer_cut: the share of the EMRAX's drive torque taken away
        when the car yaws faster than the target (from 0.05 rad/s over, all of it at
        0.2): the lever on the rear an AWD car's front-only vectoring lacks"""
        if mode not in MODES:
            raise ValueError(f"mode {mode!r}: one of {MODES}")
        self.car, self.motor, self.mode, self.regen = car, motor, mode, regen
        self.front_bias, self.k_us, self.mu_yaw = front_bias, k_us, mu_yaw
        self.yaw_kp, self.yaw_ki, self.period = yaw_kp, yaw_ki, period
        self.yaw_v_ref = yaw_v_ref
        self.slip_control, self.slip_target, self.slip_band = slip_control, slip_target, slip_band
        self.alpha = min(1.0, period / slip_tau)
        self.k = dict.fromkeys(WHEELS, 1.0)
        self.hub = HubMotor(car.hub_motor)
        self.hubs = {"fl": car.hub_front, "fr": car.hub_front, "rl": car.hub_rear, "rr": car.hub_rear}
        self.yaw_integral = 0.0
        self.log = {"yaw_ref": 0.0, "yaw_moment": 0.0}
        self.kappa = dict.fromkeys(WHEELS, 0.0)
        if mode in ("qp", "qp_lc"):
            self.qp_weights, self.qp_slip_weight = qp_weights, qp_slip_weight
            self.kappa_frac, self.mu_frac, self.inside_regen = kappa_frac, mu_frac, inside_regen
            self.tc_rise_rate, self.tc_hold, self.ay_margin = tc_rise_rate, tc_hold, ay_margin
            self.friction_circle, self.circle_floor = friction_circle, circle_floor
            self.oversteer_cut, self.launch_speed = oversteer_cut, launch_speed
            self.t_prev = np.zeros(4)                   # wheel torques the QP gave last
            self.hold = np.zeros(4)                     # s left of slow torque recovery
            self._qp_tables(tire)
            self._qp_solver()

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

    def _yaw_moment(self, s: VehicleState, ay_cap: float | None = None):
        """ay_cap: m/s^2 the yaw rate target is capped at (ay = v r); None: mu_yaw g."""
        c = self.car
        v = max(s.vx, 1.0)
        ref = v * s.steer / (c.wheelbase + self.k_us * v * v)
        cap = (ay_cap if ay_cap is not None else self.mu_yaw * G) / v
        ref = min(max(ref, -cap), cap)
        err = ref - s.yaw_rate
        self.yaw_integral = min(max(self.yaw_integral + err * self.period, -0.5), 0.5)
        self.log["yaw_ref"] = ref
        # the gains grow with speed above yaw_v_ref (the car's own yaw damping falls as 1/v)
        k = max(1.0, s.vx / self.yaw_v_ref) if self.yaw_v_ref else 1.0
        return k * (self.yaw_kp * err + self.yaw_ki * self.yaw_integral) if s.vx > 3.0 else 0.0

    def _slip(self, om, s):
        c = self.car
        side = {"fl": c.track_front / 2, "fr": -c.track_front / 2, "rl": c.track_rear / 2, "rr": -c.track_rear / 2}
        for w in WHEELS:
            v = s.vx - s.yaw_rate * side[w]
            slip = (om[w] * c.tire_radius - v) / max(abs(v), 1.0)
            self.kappa[w] = slip
            target = min(max(1 - (abs(slip) - self.slip_target) / self.slip_band, 0.0), 1.0)
            self.k[w] += self.alpha * (target - self.k[w])

    def __call__(self, s: VehicleState):
        """(central motor command, (hub commands fl, fr, rl, rr)), N m at the motors."""
        c = self.car
        om = dict(zip(WHEELS, s.omegas))
        qp = self.mode in ("qp", "qp_lc")
        if self.slip_control or qp:
            self._slip(om, s)
        cap = {w: self._hub_cap(om[w]) if self.hubs[w] else 0.0 for w in WHEELS}
        central_cap = self._central_cap(om, s.bus_v)
        cap_f, cap_r = cap["fl"] + cap["fr"], cap["rl"] + cap["rr"] + central_cap
        ask = self._ask(s, cap_f + cap_r)
        if ask < 0:                                         # regen fades out when slow
            ask *= min(max((s.vx - c.regen_min_speed) / max(c.regen_fade_speed - c.regen_min_speed, 1e-6),
                           0.0), 1.0)
        if qp:
            return self._qp(s, om, cap, central_cap, ask)
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


    # --- qp modes
    def _qp_tables(self, tire):
        """Calibration, against wheel load: peak grip, the slip to aim for, and the
        tire's longitudinal stiffness up to it (the QP's linear tire)."""
        if tire is None:
            from kmslap.car.tire import default_tire
            tire = default_tire()
        self.fz_grid = np.arange(100.0, 3001.0, 100.0)
        peaks = [tire.peak(fz) for fz in self.fz_grid]
        self.kappa_opt = np.array([self.kappa_frac * k for k, _ in peaks])
        self.mu_peak = np.array([mu for _, mu in peaks])
        self.c_kappa = np.array([tire.fx(k, fz) / k for k, fz in zip(self.kappa_opt, self.fz_grid)])

    def _qp_solver(self):
        self.solver = _qp_solver()
        self.log.update(qp_fail=0)

    def estimate_fz(self, s: VehicleState):
        """Wheel loads fl, fr, rl, rr (N) from the IMU and speed: static weight, load
        transfer (longitudinal; lateral split by roll stiffness) and downforce."""
        c = self.car
        ay = s.ay if s.ay is not None else s.vx * s.yaw_rate
        down = 0.5 * c.rho * s.vx * s.vx * c.cla
        long = c.mass * s.ax * c.cg_height / c.wheelbase
        front = c.mass * G * c.front_weight - long + down * c.aero_front
        rear = c.mass * G * (1 - c.front_weight) + long + down * (1 - c.aero_front)
        lat = c.mass * ay * c.cg_height
        d_f = c.roll_stiffness_front * lat / c.track_front
        d_r = (1 - c.roll_stiffness_front) * lat / c.track_rear
        fz = np.array([front / 2 - d_f, front / 2 + d_f, rear / 2 - d_r, rear / 2 + d_r])
        return np.maximum(fz, 50.0)

    def _qp(self, s: VehicleState, om, cap, central_cap, ask):
        c = self.car
        r = c.tire_radius
        fz = np.asarray(s.fz, float) if self.mode == "qp_lc" and s.fz is not None else self.estimate_fz(s)
        fz = np.maximum(fz, 50.0)
        sgn = 1.0 if ask >= 0 else -1.0
        kappa_opt = sgn * np.interp(fz, self.fz_grid, self.kappa_opt)
        rck = r * np.interp(fz, self.fz_grid, self.c_kappa)
        grip = self.mu_frac * np.interp(fz, self.fz_grid, self.mu_peak) * fz * r
        # the yaw rate target, capped at the lateral acceleration the tires give here
        mz_des = self._yaw_moment(s, ay_cap=self.ay_margin * grip.sum() / r / (self.mu_frac * c.mass))
        # power oversteer: yawing more than the target, the EMRAX (which can't
        # vectorize: it drives both rears through the diff) gives up drive
        ref = self.log["yaw_ref"]
        excess = math.copysign(1.0, s.yaw_rate) * (s.yaw_rate - ref) if s.vx > 3.0 else 0.0
        oversteer = min(max((excess - 0.05) / 0.15, 0.0), 1.0)
        self.log["oversteer"] = oversteer
        if self.friction_circle:
            # what's left for drive / brake after the sideways force: each wheel carries
            # its share of m ay by its load (the IMU's ay; no load cells needed)
            ay = s.ay if s.ay is not None else s.vx * s.yaw_rate
            fy = fz * abs(ay) / G * r
            grip = np.sqrt(np.maximum(grip ** 2 - fy ** 2, (self.circle_floor * grip) ** 2))
        self.log["yaw_moment"] = mz_des
        # slip against at least launch_speed: near standstill, slip speed (m/s) / that
        kappa = np.array([self.kappa[w] for w in WHEELS]) * max(abs(s.vx), 1.0) / max(abs(s.vx), self.launch_speed)
        # the slip term, on near the peak: from 0.01 under the target to 0.02 over
        on = np.clip((sgn * kappa - (np.abs(kappa_opt) - 0.01)) / 0.03, 0.0, 1.0)
        beta = self.qp_slip_weight * on / 0.01
        # x = hub torques at the wheels (fl, fr, rl, rr), then the EMRAX's at the rear axle
        m = np.zeros((4, 5))
        m[:, :4] = np.eye(4)
        m[2:, 4] = 0.5                                   # the diff splits it evenly
        hf, hr = c.track_front / 2 / r, c.track_rear / 2 / r
        z = np.array([-hf, hf, -hr, hr, 0.0])            # yaw moment per N m at each wheel
        w_a, w_z, w_s, w_d = self.qp_weights
        a_n, z_n = max(abs(ask), 50.0), 500.0
        tp = self.t_prev
        one = m.sum(axis=0)                              # sum of wheel torques
        h = 2 * (w_a / a_n ** 2 * np.outer(one, one) + w_z / z_n ** 2 * np.outer(z, z))
        g = 2 * (-w_a * ask / a_n ** 2 * one - w_z * mz_des / z_n ** 2 * z)
        # the split: each wheel's share of the total by grip, among the wheels driven
        driven = np.array([self.hubs[w] or (c.central_drive and w in ("rl", "rr")) for w in WHEELS])
        share = grip * driven / max((grip * driven).sum(), 1e-9)
        for i in range(4):
            q = w_s / a_n ** 2 + beta[i] / rck[i] ** 2
            h += 2 * q * np.outer(m[i], m[i])
            g += 2 * (-w_s / a_n ** 2 * tp[i] + beta[i] * (kappa[i] - kappa_opt[i] - tp[i] / rck[i]) / rck[i]) * m[i]
            d = m[i] - share[i] * one                      # wheel torque - its share of the total
            h += 2 * w_d / a_n ** 2 * np.outer(d, d)
        h += 1e-6 * w_s / a_n ** 2 * np.eye(5)          # the diff's null direction (AWD)
        # bounds: each motor's limits; while driving and turning, a little regen inside
        turning = abs(mz_des) > 50.0 and s.vx > 2.0
        lbx, ubx = np.zeros(5), np.zeros(5)
        for i, w in enumerate(WHEELS):
            if not self.hubs[w]:
                continue
            inside = min(self.inside_regen * c.hub_ratio * c.hub_gear_eff, cap[w]) if turning else 0.0
            lbx[i], ubx[i] = (-inside, cap[w]) if ask >= 0 else (-cap[w], 0.0)
        lbx[4], ubx[4] = ((0.0, central_cap * (1.0 - self.oversteer_cut * oversteer)) if ask >= 0
                          else (-central_cap, 0.0))
        # rows: the driver's torque (= 0 coasting), electrical power, each wheel's grip,
        # the yaw moment between 0 and what was asked
        wheel_om = np.array([om[w] for w in WHEELS])
        power = np.zeros(5)
        for i, w in enumerate(WHEELS):
            if self.hubs[w]:
                rpm = abs(wheel_om[i]) * c.hub_ratio * 60 / (2 * math.pi)
                guess = min(abs(share[i] * ask), cap[w]) / c.hub_ratio   # at the split's torque
                eta = float(self.hub.efficiency(rpm, max(guess, 1.0)))
                power[i] = wheel_om[i] / (eta * c.hub_gear_eff)
        power[4] = 0.5 * (wheel_om[2] + wheel_om[3]) / (CENTRAL_EFF * c.driveline_eff)
        a = np.vstack([one, power, m, z])
        lba = np.concatenate([[min(ask, 0.0), -np.inf], -grip, [min(mz_des, 0.0) - 20.0]])
        uba = np.concatenate([[max(ask, 0.0), c.power_limit], grip, [max(mz_des, 0.0) + 20.0]])
        x0 = np.clip(np.append(tp, 0.0), lbx, ubx)
        sol = self.solver(h=h, g=g, a=a, lba=lba, uba=uba, lbx=lbx, ubx=ubx, x0=x0)
        if not self.solver.stats()["success"]:
            self.log["qp_fail"] += 1
            x = np.zeros(5)
        else:
            x = np.array(sol["x"]).ravel()
        # slow torque recovery after the slip term acted (HyTech's TC hold)
        self.hold = np.where(on > 0, self.tc_hold, np.maximum(self.hold - self.period, 0.0))
        tw = m @ x
        step = (self.tc_rise_rate or 0.0) * self.period
        for i, w in enumerate(WHEELS):
            if self.tc_rise_rate is None:
                break
            excess = sgn * (tw[i] - tp[i]) - step
            if self.hold[i] > 0 and excess > 0:
                if self.hubs[w]:
                    x[i] -= sgn * excess
                else:
                    x[4] -= sgn * 2 * excess
                tw = m @ x
        self.t_prev = m @ x
        cmds = []
        for i, w in enumerate(WHEELS):
            gear = c.hub_gear_eff if x[i] >= 0 else 1 / c.hub_gear_eff
            cmds.append(x[i] / (c.hub_ratio * gear) if self.hubs[w] else 0.0)
        central = 0.0
        if c.central_drive:
            central = x[4] / (c.final_drive * (c.driveline_eff if x[4] >= 0 else 1 / c.driveline_eff))
        return central, tuple(cmds)


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
