"""Track loading, DRS, the pack's undervoltage limit, and the endurance event."""
import math

import numpy as np
import pytest

from accel import Car
from accel.drive import drive
from accel.driver import DrsPolicy, ScriptedDriver
from accel.endurance import endurance
from accel.powertrain import PACKS, Pack
from accel.track import michigan_2026_endurance


@pytest.fixture(scope="module")
def track():
    return michigan_2026_endurance()


def test_michigan_line(track):
    assert track.closed
    assert track.length == pytest.approx(1218, rel=0.01)      # the trace's length
    assert np.isfinite(track.k).all()
    assert 2.5 < 1 / np.abs(track.k).max() < 5                # tightest hairpin radius
    turning = np.sum((np.diff(track.heading, append=track.heading[0]) + math.pi) % (2 * math.pi) - math.pi)
    assert abs(abs(turning) - 2 * math.pi) < 0.01             # one loop


def test_drs_cuts_drag_and_downforce():
    car = Car()
    cruise = lambda: ScriptedDriver(pedal=lambda t: 0.5)
    closed = drive(cruise(), car, vx0=20.0, t_max=1.0).df.iloc[-1]
    opened = drive(cruise(), car, vx0=20.0, t_max=1.0, drs=DrsPolicy(min_speed=5)).df.iloc[-1]
    assert opened.drs == 1.0
    q = 0.5 * car.rho * opened.vx ** 2
    assert opened.drag == pytest.approx(q * car.drs_cda_open, rel=1e-6)
    assert closed.drag == pytest.approx(0.5 * car.rho * closed.vx ** 2 * car.cda, rel=1e-6)
    assert opened.fz_fl + opened.fz_fr + opened.fz_rl + opened.fz_rr < \
        closed.fz_fl + closed.fz_fr + closed.fz_rl + closed.fz_rr


def test_undervoltage_limit_stops_an_empty_pack():
    pack = Pack(PACKS["cosmx600v"])
    floor = lambda: ScriptedDriver(pedal=lambda t: 1.0)
    # low: drives, but the BMS holds the loaded cells at v_cell_min
    low = drive(floor(), Car(), pack=pack, soc0=0.08, t_max=4.0).df
    loaded = low[low.bus_i > 1.0]
    assert len(loaded) and (loaded.bus_v / pack.series >= pack.v_cell_min - 1e-3).all()
    assert (loaded.bus_v / pack.series < pack.v_cell_min + 0.01).any()    # the limit binds
    # empty (open-circuit already under the limit): no current, the car doesn't go
    empty = drive(floor(), Car(), pack=pack, soc0=0.03, t_max=2.0).df
    assert empty.bus_i.max() == 0 and empty.X.max() < 0.1


def test_endurance_two_laps(track):
    res = endurance(Car(), laps=2, path=track, speed=np.full(len(track.s), 9.0))
    s = res.summary
    assert s["finished"] and s["laps"] == 2
    assert all(abs(t - track.length / 9.0) < 15 for t in res.lap_times)
    assert 0 < 1 - s["soc_end"] < 0.2
    # the pack cooled during the driver change: temperature dips between stints
    stint0_end = res.df[res.df.stint == 0].pack_temp.iloc[-1]
    stint1_start = res.df[res.df.stint == 1].pack_temp.iloc[0]
    assert stint1_start < stint0_end
