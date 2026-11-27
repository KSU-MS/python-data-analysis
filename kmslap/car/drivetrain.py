"""Drivetrains: which wheels are driven, by what, and what the car weighs for it.

The current car is rear-wheel drive: one EMRAX 228 through the chain (final_drive) and
the Drexler LSD. The alternatives drive wheels with hub motors, each through its own
reduction gearbox at the wheel: front hubs on top of the rear EMRAX (all-wheel drive),
or a motor at every wheel (no chain, no diff). Hub motors: DTI's F-SiC kit (F-MOT) or
AMK's racing kit (DD5).

    car = variant("awd_emrax_dti")      # a Car, with its mass, weight split and CG height
    mass_breakdown("4wd_dti")           # what was taken out and put in

VARIANTS:
  rwd_emrax       the car as it is
  awd_emrax_dti   + DTI hub motors at the front wheels
  awd_emrax_amk   + AMK hub motors at the front wheels
  4wd_dti         DTI hub motors at all four wheels (no EMRAX, chain or diff)
  4wd_amk         AMK hub motors at all four wheels

Masses of the parts the current car has, and of the hub corner kits, are ESTIMATES
(COMPONENTS, CORNER_KIT); motors from their datasheets (datasheets/*.toml).
"""
import dataclasses
import math
import tomllib
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from kmslap import dirs
from kmslap.car.params import Car

DATASHEETS = {"dti": "dti_fmot.toml", "amk": "amk_dd5.toml"}


@dataclass(frozen=True)
class Part:
    name: str
    mass: float      # kg
    x: float         # m behind the front axle
    z: float         # m above the ground


def _components(car: Car) -> list[Part]:
    """The current car's rear drivetrain: what a 4WD hub car leaves out. ESTIMATES but
    the EMRAX (datasheet)."""
    wb = car.wheelbase
    return [
        Part("EMRAX 228HV", 12.3, wb - 0.30, 0.26),
        Part("its inverter (Cascadia PM100DX)", 7.5, wb - 0.55, 0.35),
        Part("motor mount", 1.5, wb - 0.30, 0.20),
        Part("chain and sprockets", 1.2, wb - 0.10, 0.23),
        Part("chainguard", 0.8, wb - 0.10, 0.25),
        Part("Drexler LSD, carrier and mounts", 4.8, wb, car.tire_radius),
        Part("rear half shafts and CV joints", 2.0, wb, car.tire_radius),
    ]


# per driven corner, besides the motor and its share of an inverter: ESTIMATES
CORNER_KIT = {"gearbox (planetary, ~13:1)": 2.0, "heavier upright / hub": 0.6, "motor cooling": 0.4,
              "HV cabling": 0.5}


@lru_cache(maxsize=None)
def motor_sheet(kind: str) -> dict:
    return tomllib.loads((dirs.DATASHEETS / DATASHEETS[kind]).read_text())


class HubMotor:
    """A hub motor's limits and efficiency (motor x inverter), at the motor shaft."""

    def __init__(self, kind: str):
        s = motor_sheet(kind)
        self.kind, self.name = kind, s["model"]
        self.peak_torque = s["peak_torque_nm"]
        self.peak_power = s["peak_power_kw"] * 1e3
        self.max_rpm = float(s["max_rpm"])
        self.mass = s["mass_kg"]
        self.inverter_mass = s["inverter"]["mass_kg_per_motor"]
        self.inverter_eff = s["inverter"]["efficiency"]
        self.rotor_inertia = s["rotor_inertia_kgm2"]
        dti = motor_sheet("dti")
        self._iron = np.array(dti["iron_w_vs_rpm"], float)
        self._cu = np.array(dti["copper_w_vs_arms"], float)
        self._kt = dti["torque_constant_nm_per_arms"]
        # another motor without loss data: DTI's losses at the same fraction of peak
        # torque and speed, scaled to its power (an ESTIMATE)
        self._t_scale = dti["peak_torque_nm"] / self.peak_torque
        self._n_scale = dti["max_rpm"] / self.max_rpm
        self._p_scale = self.peak_power / (dti["peak_power_kw"] * 1e3)

    def max_torque(self, rpm):
        """N m: peak torque up to peak power, then constant power; 0 past max speed."""
        rpm = np.abs(np.asarray(rpm, float))
        w = np.maximum(rpm * 2 * math.pi / 60, 1e-6)
        return np.where(rpm >= self.max_rpm, 0.0, np.minimum(self.peak_torque, self.peak_power / w))

    def losses(self, rpm, torque):
        """W in the motor (iron + copper)."""
        rpm_d = np.abs(np.asarray(rpm, float)) * self._n_scale
        amps = np.abs(np.asarray(torque, float)) * self._t_scale / self._kt
        return self._p_scale * (np.interp(rpm_d, *self._iron.T) + np.interp(amps, self._cu[:, 0], self._cu[:, 1],
                                                                             right=None))

    def efficiency(self, rpm, torque):
        """Shaft power / electrical power (motor x inverter), either direction."""
        p = np.abs(np.asarray(torque, float) * np.asarray(rpm, float) * 2 * math.pi / 60)
        eta_m = p / np.maximum(p + self.losses(rpm, torque), 1e-9)
        return np.clip(eta_m * self.inverter_eff, 0.3, 1.0)

    def tables(self, n_rpm: int = 81, n_torque: int = 33):
        """(rpm grid, torque grid, max torque over rpm, efficiency [rpm, torque]) for the
        car model's interpolants."""
        rpms = np.linspace(0, self.max_rpm, n_rpm)
        torques = np.linspace(0, self.peak_torque, n_torque)
        eff = self.efficiency(rpms[:, None], torques[None, :])
        eff[0, :] = eff[1, :]            # standing still: the first speed's efficiency
        eff[:, 0] = eff[:, 1]
        return rpms, torques, self.max_torque(rpms), eff


VARIANTS = {
    "rwd_emrax": dict(central_drive=True, hub_front=False, hub_rear=False, hub_motor="dti"),
    "awd_emrax_dti": dict(central_drive=True, hub_front=True, hub_rear=False, hub_motor="dti"),
    "awd_emrax_amk": dict(central_drive=True, hub_front=True, hub_rear=False, hub_motor="amk"),
    "4wd_dti": dict(central_drive=False, hub_front=True, hub_rear=True, hub_motor="dti"),
    "4wd_amk": dict(central_drive=False, hub_front=True, hub_rear=True, hub_motor="amk"),
}


def name_of(car: Car) -> str:
    """The VARIANTS name of the car's drivetrain ("custom" if none matches)."""
    for name, v in VARIANTS.items():
        if all(getattr(car, k) == x for k, x in v.items()):
            return name
    return "custom"


def mass_breakdown(name: str, base: Car = None) -> dict:
    """The variant's mass, weight split and CG height from the current car's, part by
    part: {"removed": [...], "added": [...], "mass", "front_weight", "cg_height"}."""
    base = base or Car()
    v = VARIANTS[name]
    wb = base.wheelbase
    removed = [] if v["central_drive"] else _components(base)
    added = []
    hub = HubMotor(v["hub_motor"])
    corners = [("front", 0.0)] * (2 * v["hub_front"]) + [("rear", wb)] * (2 * v["hub_rear"])
    for axle, x in corners:
        added.append(Part(f"{hub.name} ({axle})", hub.mass, x, base.tire_radius))
        for part, m in CORNER_KIT.items():
            added.append(Part(f"{part} ({axle})", m, x, base.tire_radius))
        added.append(Part(f"inverter share ({axle})", hub.inverter_mass, 0.6 * wb, 0.30))
    m0 = base.mass
    x0 = (1 - base.front_weight) * wb               # CG behind the front axle
    z0 = base.cg_height
    m = m0 - sum(p.mass for p in removed) + sum(p.mass for p in added)
    mx = m0 * x0 - sum(p.mass * p.x for p in removed) + sum(p.mass * p.x for p in added)
    mz = m0 * z0 - sum(p.mass * p.z for p in removed) + sum(p.mass * p.z for p in added)
    return {"removed": removed, "added": added, "mass": m, "front_weight": 1 - (mx / m) / wb,
            "cg_height": mz / m}


def variant(name: str, base: Car = None, **overrides) -> Car:
    """The car with this drivetrain (and its mass, weight split and CG height)."""
    base = base or Car()
    b = mass_breakdown(name, base)
    return dataclasses.replace(base, **VARIANTS[name], mass=b["mass"], front_weight=b["front_weight"],
                               cg_height=b["cg_height"], **overrides)
