"""Hub motor inverters as the VCU commands them over CAN, between the torque controller
(vcu/torque_control.py) and the car model's motor torque lag (models/twotrack.py).

    inv = Inverters(car)                  # the kind the car's hub motors come with
    inv.command(t, torques, ground_rpm)   # at the controller rate: N m at each motor
    torques = inv(t, dt, motor_rpm)       # every model step: what the motors are asked for

DTI HV500 / F-SiC (datasheets/DTI F-SiC Documents, CAN manual): a native current
mode. SetAcCurrent (0.1 A peak steps; sending it switches the inverter to current
control), SetBrakeCurrent for regen, the AC current limits on top. The inverter stops
controlling (free running) when no control message comes within its timeout.

AMK racing kit (PDK_205481 KW26-S5-FSE-4Q, sec. 3.2 / 6.4.4): speed control with
torque limits only (ID32800 = 0x480003): AMK_TargetVelocity (rpm) and
AMK_TorqueLimitPositiv / Negativ (0.1 % of the nominal torque Mn). A torque comes from
saturating the speed loop: a setpoint far above the motor's speed with the positive
limit set to the torque (HyTech's drivebrain: 45 m/s), or 0 rpm with the negative
limit for regen. encoder "slip" puts the setpoint at the ground speed plus a slip
target instead, so the inverter's own speed loop holds the wheel there (traction
control at the inverter's loop rate). With the QP torque allocator that costs
stability (`kmslap torque-bench --inverters slip`: the speed loop caps each front
wheel's torque a few percent of slip in, which is the vectoring's authority in a
corner), so "max" is the default.
"""
import math

from kmslap.car.drivetrain import motor_sheet

RPM_PER_RADS = 60 / (2 * math.pi)


class DtiCurrentMode:
    """One DTI inverter in current control: torque = kt * I_ac, in 0.1 A steps."""

    def __init__(self, kind: str = "dti", timeout: float = 1.0):
        s = motor_sheet(kind)
        self.kt = s.get("torque_constant_nm_per_arms", motor_sheet("dti")["torque_constant_nm_per_arms"])
        self.i_max = s["peak_torque_nm"] / self.kt       # SetMaxAcCurrent: the motor's peak
        self.timeout = timeout
        self.amps, self.last = 0.0, None

    def command(self, t, torque, ground_rpm=None):
        amps = min(max(torque / self.kt, -self.i_max), self.i_max)
        self.amps = round(amps * 10) / 10                # SetAcCurrent / SetBrakeCurrent: 0.1 A
        self.last = t

    def __call__(self, t, dt, rpm):
        if self.last is None or t - self.last > self.timeout:
            return 0.0                                   # timed out: free running
        return self.amps * self.kt


class AmkSpeedMode:
    """One AMK inverter: a PI speed loop whose output is clamped to the torque limits.
    kp: N m per rpm of speed error; tn: the integral's reset time (AMK ID100 / ID101)."""

    def __init__(self, kind: str = "amk", encoder: str = "max", kp: float = 0.05, tn: float = 0.02,
                 speed_set: float = 45.0, slip_target: float = 0.10, wheel_rpm_per_ms: float = 1.0,
                 timeout: float = 0.05):
        if encoder not in ("max", "slip"):
            raise ValueError(f"encoder {encoder!r}: 'max' or 'slip'")
        s = motor_sheet(kind)
        self.mn = s["nominal_torque_nm"]
        self.unit = self.mn / 1000                       # 0.1 % of Mn
        self.peak = s["peak_torque_nm"]
        self.encoder, self.kp, self.tn = encoder, kp, tn
        self.speed_set, self.slip_target = speed_set, slip_target
        self.rpm_per_ms = wheel_rpm_per_ms               # motor rpm per m/s at the ground
        self.timeout = timeout
        self.rpm_set, self.pos, self.neg = 0.0, 0.0, 0.0
        self.integral, self.last = 0.0, None

    def command(self, t, torque, ground_rpm):
        """The torque as a setpoint and limits (AMK Setpoints 1)."""
        q = lambda x: round(min(abs(x), self.peak) / self.unit) * self.unit
        if torque >= 0:
            if self.encoder == "max":
                self.rpm_set = self.speed_set * self.rpm_per_ms
            else:
                self.rpm_set = max(ground_rpm * (1 + self.slip_target), ground_rpm + 200.0)
            self.pos, self.neg = q(torque), 0.0
        else:
            self.rpm_set = 0.0 if self.encoder == "max" else ground_rpm * (1 - self.slip_target)
            self.pos, self.neg = 0.0, q(torque)
        self.rpm_set = round(self.rpm_set)               # AMK_TargetVelocity: rpm, integer
        self.last = t

    def __call__(self, t, dt, rpm):
        if self.last is None or t - self.last > self.timeout:
            self.integral = 0.0
            return 0.0
        err = self.rpm_set - rpm
        out = self.kp * err + self.integral
        if -self.neg < out < self.pos:                   # integrate only when not clamped
            self.integral += self.kp * err * dt / self.tn
        self.integral = min(max(self.integral, -self.neg), self.pos)
        return min(max(out, -self.neg), self.pos)


class Inverters:
    """The four hub motor inverters (fl, fr, rl, rr); a wheel without a hub motor gets 0."""

    def __init__(self, car, kind: str | None = None, **kw):
        kind = kind or car.hub_motor
        hubs = (car.hub_front, car.hub_front, car.hub_rear, car.hub_rear)
        rpm_per_ms = car.hub_ratio / car.tire_radius * RPM_PER_RADS
        if kind == "dti":
            make = lambda: DtiCurrentMode(**kw)
        elif kind == "amk":
            make = lambda: AmkSpeedMode(wheel_rpm_per_ms=rpm_per_ms, **kw)
        else:
            raise ValueError(f"inverter kind {kind!r}: 'dti' or 'amk'")
        self.kind = kind
        self.units = [make() if h else None for h in hubs]

    def command(self, t, torques, ground_rpm):
        for u, tq, g in zip(self.units, torques, ground_rpm):
            if u is not None:
                u.command(t, tq, g)

    def __call__(self, t, dt, motor_rpm):
        return tuple(u(t, dt, n) if u is not None else 0.0 for u, n in zip(self.units, motor_rpm))
