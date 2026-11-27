"""Monte Carlo over model uncertainty: the simulated car draws its true
coefficients from ranges, while controllers keep their nominal beliefs.

    samples = sample_plants(40)
    table = monte_carlo(strategies, samples)
"""
import dataclasses

import numpy as np
import pandas as pd

from .params import Car
from .sim import quasi_static, simulate
from .tire import PacejkaTire

# (low, high) multipliers on the nominal value, or absolute ranges where noted
RANGES = {
    "grip": (0.85, 1.15),
    "mass": (0.95, 1.05),
    "cg_height": (0.8, 1.2),
    "wheel_inertia": (0.7, 1.3),
    "rear_driveline_inertia": (0.7, 1.3),
    "torque_tau": (0.01, 0.06),          # s, absolute
    "pitch_tau": (0.02, 0.10),           # s, absolute
    "relaxation_length": (0.10, 0.25),   # m, absolute
    "crr": (0.5, 1.5),
    "cda": (0.8, 1.2),
    "cla": (0.8, 1.2),
    "driveline_eff": (0.90, 0.97),       # absolute
    "tire_radius": (0.98, 1.02),
    "vn_accel_bias": (-0.2, 0.2),        # m/s^2, absolute
}
ABSOLUTE = {"torque_tau", "pitch_tau", "relaxation_length", "driveline_eff", "vn_accel_bias"}


@dataclasses.dataclass
class Plant:
    car: Car
    tire: PacejkaTire
    relaxation_length: float
    vn_accel_bias: float
    draw: dict


def sample_plants(n: int, nominal: Car = None, seed: int = 0, ranges=RANGES) -> list[Plant]:
    nominal = nominal or Car()
    rng = np.random.default_rng(seed)
    plants = []
    for _ in range(n):
        d = {k: float(rng.uniform(*v)) for k, v in ranges.items()}
        car_fields = {k: (d[k] if k in ABSOLUTE else getattr(nominal, k) * d[k])
                      for k in d if hasattr(nominal, k)}
        plants.append(Plant(car=dataclasses.replace(nominal, **car_fields),
                            tire=PacejkaTire(scale=0.6 * d["grip"]),
                            relaxation_length=d["relaxation_length"],
                            vn_accel_bias=d["vn_accel_bias"], draw=d))
    return plants


def monte_carlo(strategies: dict, plants: list[Plant], motor="228HV") -> pd.DataFrame:
    """strategies: name -> f(plant) returning simulate() kwargs (fresh controller
    objects per call). Returns one row per plant: the draw, the ideal time, and each
    strategy's percent over ideal."""
    rows = []
    for i, p in enumerate(plants):
        ideal = quasi_static(p.car, motor, tire=p.tire).accel_time
        row = {"sample": i, **p.draw, "ideal_s": ideal}
        for name, make in strategies.items():
            r = simulate(p.car, motor, tire=p.tire, relaxation_length=p.relaxation_length,
                         **make(p))
            row[name] = 100 * (r.accel_time / ideal - 1)
        rows.append(row)
    return pd.DataFrame(rows)
