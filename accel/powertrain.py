"""Motor torque map and accumulator model.

The dq-current solver in motor.py costs ~0.35 ms per call, so it's tabulated once
per motor into a max-torque(rpm, bus voltage) grid and interpolated during the sim.
"""
import math
import tomllib
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np

import battery
from motor import EMRAX_MOTORS, Motor

MOTORS = {m.name: m for m in EMRAX_MOTORS}

# Torque requested from the dq solver when building the map; it caps output at this.
PEAK_TORQUE = {"228MV": 230.0, "228HV": 230.0, "208MV": 160.0, "208HV": 160.0}

# Rotor inertia: 228 from emrax_228_technical_data_4.5.pdf (421 kg cm^2); 208 from an
# older EMRAX datasheet (256 kg cm^2), VERIFY. PEAK_TORQUE above stays 230 Nm, under
# the 228's datasheet 240 Nm "for a few seconds": confirm what the inverter is set to.
ROTOR_INERTIA = {"228MV": 0.0421, "228HV": 0.0421, "208MV": 0.0256, "208HV": 0.0256}


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


DATASHEETS = Path(__file__).resolve().parent.parent / "datasheets"
# motor efficiency maps (shaft vs electrical), digitized from the datasheets by
# datasheets/extract_emrax_efficiency.py. EMRAX says its MV / LV 228s have the HV's maps.
EFFICIENCY_MAPS = {"228HV": "emrax_228hv_efficiency.csv", "228MV": "emrax_228hv_efficiency.csv"}


@lru_cache(maxsize=None)
def motor_efficiency_map(name: str):
    """(rpm grid, torque grid (Nm), efficiency [rpm, torque] as a fraction)."""
    import pandas as pd

    d = pd.read_csv(DATASHEETS / EFFICIENCY_MAPS[name], comment="#")
    rpms, torques = np.unique(d.rpm), np.unique(d.torque_nm)
    eff = d.pivot(index="rpm", columns="torque_nm", values="efficiency").loc[rpms, torques].to_numpy()
    return rpms.astype(float), torques.astype(float), eff
# manufacturer datasheets for the cells in battery.batt_models, by pack name prefix
CELL_DATASHEETS = {"COSMX": "cosmx_95b0d0hd.toml"}


@lru_cache(maxsize=None)
def cell_datasheet(pack_name: str) -> dict | None:
    for prefix, file in CELL_DATASHEETS.items():
        if pack_name.upper().startswith(prefix):
            return tomllib.loads((DATASHEETS / file).read_text())
    return None


class Pack:
    """Accumulator as an open-circuit voltage behind a series resistance.

    State of charge is held constant: a 75 m run uses ~0.1 kWh of a ~5 kWh pack.
    """

    def __init__(self, pack: battery.batteryPack, *, cell_mass: float | None = None,
                 cell_heat_capacity: float = 1000.0, h_a: float = 15.0,
                 ambient_c: float = 25.0, resistance_temp_coeff: float = 0.02,
                 v_cell_min: float | None = None, v_cell_max: float | None = None):
        """The thermal and state-of-charge terms are used by the double-track model
        (vehicle.py); the accel sim holds SOC and temperature constant.

        Cells with a datasheet in datasheets/ (CELL_DATASHEETS) take their limits
        from it: continuous discharge and charge current, voltage range, mass. The
        keyword arguments override.

        cell_mass: kg per cell (datasheet, else 0.27 ESTIMATE).
        cell_heat_capacity: J/(kg K), ESTIMATE (typical Li-ion pouch).
        h_a: W/K from the pack to ambient. No active cooling; ESTIMATE.
        resistance_temp_coeff: R(T) = R * exp(-coeff * (T - 25 C)), ESTIMATE.
        v_cell_min / v_cell_max: V, loaded cell voltage the BMS keeps the pack
            within (datasheet range, else 3.0 / 4.2): discharge current is limited to
            hold v_cell_min, so a low pack loses power and an empty one stops the car;
            charge (regen) current to hold v_cell_max.
        """
        sheet = cell_datasheet(pack.name or "")
        self.name = pack.name
        self.v_oc = pack.getInstantaneousVoltage(0)
        self.resistance = pack.cell.resistance * pack.seriesCount / pack.parallelCount
        # continuous ratings (the datasheet's peaks only last 2 s)
        self.max_current = (sheet["discharge"]["continuous_a"] * pack.parallelCount if sheet
                            else pack.maxdischarge)
        self.max_charge_current = (sheet["charge"]["continuous_a"] * pack.parallelCount if sheet
                                   else 0.5 * pack.maxdischarge)
        self.series, self.parallel = pack.seriesCount, pack.parallelCount
        self.capacity_ah = pack.cell.maxcapacity * pack.parallelCount / 1000
        # cell open-circuit voltage vs state of charge, from the cell's capacity curve
        # (battery_soc_curves/; the COSMX one is adjusted from the LG HE2 curve, not measured)
        self.ocv_soc = np.linspace(0.0, 1.0, 26)
        self.ocv_cell = np.asarray(pack.cell.voltageUpdateFunction(self.ocv_soc * pack.cell.maxcapacity))
        cell_mass = cell_mass if cell_mass is not None else (sheet["mass_kg"] if sheet else 0.27)
        self.thermal_mass = cell_mass * pack.seriesCount * pack.parallelCount * cell_heat_capacity
        self.h_a = h_a
        self.ambient_c = ambient_c
        self.resistance_temp_coeff = resistance_temp_coeff
        self.v_cell_min = v_cell_min if v_cell_min is not None else (sheet["voltage_min"] if sheet else 3.0)
        self.v_cell_max = v_cell_max if v_cell_max is not None else (sheet["voltage_max"] if sheet else 4.2)
        self.datasheet = sheet

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
