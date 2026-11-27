"""A cell's two sides in the loop: CellModel is a (virtual) real cell, the truth behind
the rig emulator; CellEstimator is what the host makes of any rig's measurements, real
or emulated, to hand the car model a pack.

CellModel isn't the sim's pack model on purpose: it adds an RC polarization branch, its
own resistance and a single cell's thermals, so the loop has something the sim doesn't
already know to find (as a real cell will).
"""
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class CellModel:
    """One cell: OCV(SOC) behind R0(T) and an RC branch; lumped thermal mass.
    Current positive = discharge."""
    ocv_soc: np.ndarray
    ocv_cell: np.ndarray
    capacity_ah: float = 13.0
    soc: float = 1.0
    temp_c: float = 25.0
    ambient_c: float = 25.0
    r0: float = 1.3e-3          # ohm at 25 C (datasheet: DCIR <= 1.5 mOhm at high rate)
    r1: float = 0.8e-3          # ohm, polarization
    tau1: float = 8.0           # s
    r_temp_coeff: float = 0.02  # R(T) = R exp(-coeff (T - 25))
    heat_capacity: float = 269.0   # J/K: 0.269 kg x ~1000 J/(kg K)
    h_a: float = 0.15           # W/K to ambient: a cell in its fixture, ESTIMATE
    v_rc: float = 0.0

    def ocv(self) -> float:
        return float(np.interp(min(max(self.soc, 0.0), 1.0), self.ocv_soc, self.ocv_cell))

    def r0_now(self) -> float:
        return self.r0 * math.exp(-self.r_temp_coeff * (self.temp_c - 25.0))

    def voltage(self, current: float) -> float:
        return self.ocv() - current * self.r0_now() - self.v_rc

    def step(self, current: float, dt: float) -> float:
        """Advance dt with `current` flowing; the terminal voltage after."""
        self.soc -= current * dt / (self.capacity_ah * 3600)
        a = math.exp(-dt / self.tau1)
        self.v_rc = self.v_rc * a + current * self.r1 * (1 - a)
        heat = current * current * self.r0_now() + self.v_rc * current
        self.temp_c += dt * (heat - self.h_a * (self.temp_c - self.ambient_c)) / self.heat_capacity
        return self.voltage(current)


@dataclass
class CellEstimator:
    """The host's view of a cell from its measurements: state of charge by counting
    charge; resistance learned from current changes over `lag` s (dV / dI: sample to
    sample, currents change too little to see past the noise); the effective
    open-circuit voltage (with whatever polarization it carries) as V + I R, smoothed.
    The car model gets the pack built from it: series x OCV behind series / parallel x R."""
    ocv_soc: np.ndarray
    ocv_cell: np.ndarray
    capacity_ah: float
    soc: float
    r: float                      # ohm, prior (the sim's per-cell resistance)
    r_min: float = 0.3e-3
    r_max: float = 10e-3
    step_min: float = 5.0         # A: smaller current steps don't update R (noise)
    r_gain: float = 0.05
    tau_ocv: float = 0.05         # s, smoothing of the open-circuit voltage
    lag: float = 0.02             # s
    v_oc: float = field(default=float("nan"))
    _hist: deque = field(default_factory=deque)
    _t: float = 0.0

    def __post_init__(self):
        if math.isnan(self.v_oc):
            self.v_oc = float(np.interp(self.soc, self.ocv_soc, self.ocv_cell))

    def update(self, v: float, i: float, dt: float):
        self.soc -= i * dt / (self.capacity_ah * 3600)
        self._t += dt
        self._hist.append((self._t, v, i))
        while len(self._hist) > 1 and self._hist[1][0] <= self._t - self.lag:
            self._hist.popleft()
        t0, v0, i0 = self._hist[0]
        if self._t - t0 >= self.lag and abs(i - i0) >= self.step_min:
            r_obs = min(max(-(v - v0) / (i - i0), self.r_min), self.r_max)
            self.r += self.r_gain * (r_obs - self.r)
            self._hist.clear()                   # each window counts once
        a = dt / (self.tau_ocv + dt) if dt > 0 else 1.0
        self.v_oc += a * (v + i * self.r - self.v_oc)
