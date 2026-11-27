"""Traction control: a slip-target PI controller that scales the driver's torque request.

Replaces the Autopid-based attempt in accel_weight_transfer.ipynb. Same idea (output
in [-1, 0] added to 1 to scale torque), with these fixes:
  - error is normalized by the target so gains don't depend on the target slip
  - the integral can only ever take torque away (clamped to [-1, 0] of output), and
    stops winding down while the output is already at full cut (anti-windup)
  - derivative uses the correct time units (old code was off by 1e6)
  - runs at a fixed controller period with zero-order hold, like an ECU task
"""
from dataclasses import dataclass, field


@dataclass
class SlipTC:
    # gains tuned by sweep on the LC0 tire with inverter lag, FD 3.0-4.5: within
    # 0.7% of ideal TC, <0.035 slip overshoot. Much higher kp limit-cycles against
    # the wheel/tire shuffle mode. Default target is 0.85 x the LC0 tire's peak
    # slip of 0.12; use for_tire() for other tires.
    target_slip: float = 0.10
    kp: float = 0.3
    ki: float = 1.0            # 1/s
    kd: float = 0.0            # s
    # integral gain multiplier while slip is below target, so torque comes back
    # quickly once grip is available (e.g. as the car goes motor-limited)
    recover_gain: float = 3.0
    period: float = 0.003      # s, controller update rate

    _next_t: float = field(default=0.0, init=False, repr=False)
    _integral: float = field(default=0.0, init=False, repr=False)
    _prev_err: float = field(default=None, init=False, repr=False)
    _scale: float = field(default=1.0, init=False, repr=False)

    @classmethod
    def for_tire(cls, tire, fz: float = 700.0, margin: float = 0.85, **kw) -> "SlipTC":
        """Target `margin` x the tire's peak slip at wheel load `fz`. Staying a bit
        under the peak costs little grip and keeps overshoot off the falling side."""
        return cls(target_slip=margin * tire.peak(fz)[0], **kw)

    def reset(self):
        self._next_t = 0.0
        self._integral = 0.0
        self._prev_err = None
        self._scale = 1.0

    @property
    def scale(self) -> float:
        """Current torque multiplier in [0, 1]."""
        return self._scale

    def update(self, t: float, slip: float) -> float:
        """Feed the measured slip at time t; returns the torque multiplier in [0, 1]."""
        if t + 1e-12 < self._next_t:
            return self._scale
        dt = self.period
        self._next_t = t + dt

        err = (self.target_slip - slip) / self.target_slip
        d_err = 0.0 if self._prev_err is None else (err - self._prev_err) / dt
        self._prev_err = err

        ki = self.ki * (self.recover_gain if err > 0 else 1.0)
        trial = min(max(self._integral + ki * err * dt, -1.0), 0.0)
        # conditional integration: while the output is already pinned at full cut,
        # don't keep winding the integral down (it would delay recovery)
        if not (self.kp * err + trial + self.kd * d_err < -1.0 and err < 0):
            self._integral = trial
        u = self.kp * err + self._integral + self.kd * d_err
        self._scale = 1.0 + min(max(u, -1.0), 0.0)
        return self._scale
