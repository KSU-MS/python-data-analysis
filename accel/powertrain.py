"""Motor torque map and accumulator model.

The dq-current solver in motor.py costs ~0.35 ms per call, so it's tabulated once
per motor into a max-torque(rpm, bus voltage) grid and interpolated during the sim.
"""
import math
import warnings
from functools import lru_cache

import numpy as np

import battery
from motor import EMRAX_MOTORS, Motor

MOTORS = {m.name: m for m in EMRAX_MOTORS}

# Torque requested from the dq solver when building the map; it caps output at this.
PEAK_TORQUE = {"228MV": 230.0, "228HV": 230.0, "208MV": 160.0, "208HV": 160.0}

# Rotor inertia from EMRAX datasheets (228: 383 kg cm^2, 208: 256 kg cm^2). VERIFY.
ROTOR_INERTIA = {"228MV": 0.0383, "228HV": 0.0383, "208MV": 0.0256, "208HV": 0.0256}


class MotorMap:
    def __init__(self, name: str, rpm_step: float = 100.0, v_min: float = 150.0,
                 v_max: float = 650.0, v_step: float = 20.0):
        motor: Motor = MOTORS[name]
        self.name = name
        self.peak_torque = PEAK_TORQUE[name]
        self.rotor_inertia = ROTOR_INERTIA[name]
        self.max_rpm = float(motor.max_rate)
        self.rpm_step, self.v_min, self.v_step = rpm_step, v_min, v_step
        self.table = _build_table(name, rpm_step, v_min, v_max, v_step)

    def max_torque(self, rpm: float, voltage: float) -> float:
        """Bilinear interpolation of the max-torque table; 0 above the rpm limit."""
        if rpm >= self.max_rpm:
            return 0.0
        tbl = self.table
        ni, nj = len(tbl), len(tbl[0])
        fi = min(max(rpm / self.rpm_step, 0.0), ni - 1.000001)
        fj = min(max((voltage - self.v_min) / self.v_step, 0.0), nj - 1.000001)
        i, j = int(fi), int(fj)
        di, dj = fi - i, fj - j
        row0 = tbl[i]
        row1 = tbl[i + 1]
        t0 = row0[j] + (row0[j + 1] - row0[j]) * dj
        t1 = row1[j] + (row1[j + 1] - row1[j]) * dj
        return t0 + (t1 - t0) * di


@lru_cache(maxsize=None)
def _build_table(name, rpm_step, v_min, v_max, v_step):
    motor = MOTORS[name]
    peak = PEAK_TORQUE[name]
    rpms = np.arange(0.0, motor.max_rate + rpm_step, rpm_step)
    volts = np.arange(v_min, v_max + v_step / 2, v_step)
    tbl = np.zeros((len(rpms), len(volts)))
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        for i, rpm in enumerate(rpms):
            for j, v in enumerate(volts):
                # the dq solver divides by electrical speed, so avoid rpm == 0
                tbl[i, j] = motor.get_qd_currents(max(rpm, 1.0), peak, v)[2]
    # store as nested lists: indexing python lists is much faster than numpy scalars
    return tuple(tuple(float(x) for x in row) for row in np.nan_to_num(tbl))


class Pack:
    """Accumulator as an open-circuit voltage behind a series resistance.

    State of charge is held constant: a 75 m run uses ~0.1 kWh of a ~5 kWh pack.
    """

    def __init__(self, pack: battery.batteryPack):
        self.name = pack.name
        self.v_oc = pack.getInstantaneousVoltage(0)
        self.resistance = pack.cell.resistance * pack.seriesCount / pack.parallelCount
        self.max_current = pack.maxdischarge

    def solve(self, power: float):
        """Bus (voltage, current) when drawing `power` W, honoring the current limit.

        Solves P = V*I with V = Voc - I*R in closed form (replaces the old
        fixed-point iteration). Returns the delivered power too, which is less than
        requested when the pack can't supply it.
        """
        if power <= 0:
            return self.v_oc, 0.0, 0.0
        r = self.resistance
        disc = self.v_oc * self.v_oc - 4 * r * power
        current = (self.v_oc - math.sqrt(disc)) / (2 * r) if disc >= 0 else self.v_oc / (2 * r)
        current = min(current, self.max_current)
        v = self.v_oc - current * r
        return v, current, v * current


PACKS = {name: getattr(battery.batt_models, name)
         for name in vars(battery.batt_models) if not name.startswith("_")}
