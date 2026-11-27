"""Endurance strategy tuning: the projection from 2 laps and the scoring."""
import math

import pytest

from kmslap.car.powertrain import PACKS, Pack
from kmslap.strategy import _soc_after, evaluate


def test_soc_after_energy():
    pack = Pack(PACKS["cosmx600v"])
    assert _soc_after(0.0) == pytest.approx(1.0)
    full = pack.capacity_ah * 3600 * pack.series * 3.6      # roughly the whole pack
    assert 0.0 <= _soc_after(full) < 0.1
    assert _soc_after(1e6) > _soc_after(2e6) > _soc_after(4e6)


def test_evaluate_projects_and_penalizes():
    drs = {"min_speed": 11.0, "max_lateral": 9.0, "max_steering_wheel": math.radians(20)}
    r = evaluate({"margin": 0.85, "coast_time": 1.0, "drs": drs, "soc_floor": 0.1})
    assert r["finished_2"] and r["drs"] and 0 < r["drs_open"] < 1
    assert r["event_time_s"] == pytest.approx(2 * r["lap_standing_s"] + 16 * r["lap_flying_s"])
    # an impossible reserve: same run, scored with a penalty
    strict = evaluate({"margin": 0.85, "coast_time": 1.0, "drs": drs, "soc_floor": 0.99})
    assert strict["objective"] > strict["event_time_s"] + 1000
