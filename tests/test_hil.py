"""Cell in the loop: the rig protocol (emulator <-> host backend), the pack backends in
the car model, the Foxglove layout."""
import json

import numpy as np
import pytest

from kmslap.car.params import Car
from kmslap.car.powertrain import PACKS, Pack
from kmslap.driving.drive import drive
from kmslap.driving.follower import PathFollower
from kmslap.driving.paths import Path
from kmslap.hil.backends import RigFault, SimPack, virtual_cell


@pytest.fixture(scope="module")
def pack():
    return Pack(PACKS["cosmx600v"])


def oval():
    return Path.from_segments([("straight", 40.0), ("arc", 12.0, 180.0), ("straight", 40.0), ("arc", 12.0, 180.0)],
                              step=0.5, closed=True)


def _drive(pack, backend, t_max=6.0):
    p = oval()
    return drive(PathFollower(p, speed=14.0, car=Car(), laps=1), Car(), pack=pack, pack_backend=backend,
                 psi0=float(p.heading[0]), t_max=t_max, log_period=0.01).df


def test_sim_pack_backend_matches_the_models_own_pack(pack):
    own, sim = _drive(pack, None), _drive(pack, SimPack())
    assert own.speed.iloc[-1] == pytest.approx(sim.speed.iloc[-1], rel=1e-3)
    assert own.soc.iloc[-1] == pytest.approx(sim.soc.iloc[-1], abs=1e-5)
    assert np.abs(own.bus_v - sim.bus_v).max() < 1.0          # V on a ~600 V pack, 10 ms ticks


def test_virtual_cell_in_the_loop(pack):
    be = virtual_cell(pack, lockstep=True)
    d = _drive(pack, be)
    assert be.status["state"] == "RUNNING" and not be.status["faults"]
    # the rig's own charge counter and the host's agree
    assert be.counters["charge"] == pytest.approx((1 - be.est.soc) * pack.capacity_ah, rel=0.02)
    # the car saw the cell: its pack voltage is the cell's x 144 (with the cell's own sag)
    assert d.bus_v.max() / pack.series == pytest.approx(be.rig.cell.ocv(), abs=0.05)
    be.stop()
    assert be.status["state"] == "IDLE"
    be.close()


def test_rig_watchdog_and_current_limits(pack):
    be = virtual_cell(pack, lockstep=True)
    be.start(pack, 1.0, 25.0)
    be.exchange(0.0, 10.0, 0.0)
    be.rig.step(0.05)                                          # 50 ms without a setpoint
    be._poll(0.051)
    assert be.status["state"] == "FAULT" and "watchdog" in be.status["faults"]
    be.close()
    be = virtual_cell(pack, lockstep=True)
    be.start(pack, 1.0, 25.0)
    with pytest.raises(RigFault, match="over_current"):
        for k in range(20):
            be.exchange(k * 1e-3, 320.0, 0.0)                  # twice the continuous limit
    be.close()


def test_hil_layout_has_the_controls():
    from kmslap.hil.live import hil_layout
    s = hil_layout()
    layout = json.loads(s)
    # the app's file format: panels by id, a split tree of their ids
    assert {"configById", "layout"} <= set(layout)
    kinds = {k.split("!")[0] for k in layout["configById"]}
    assert {"3D", "map", "Gauge", "Indicator", "CallService", "Plot", "Tab"} <= kinds
    calls = [c["serviceName"] for k, c in layout["configById"].items() if k.startswith("CallService")]
    assert sorted(calls) == ["abort", "reset", "start", "stop"]

    def ids(tree):
        return [tree] if isinstance(tree, str) else ids(tree["first"]) + ids(tree["second"])
    shown = set(ids(layout["layout"]))
    for c in layout["configById"].values():
        for tab in c.get("tabs", []):
            shown |= set(ids(tab["layout"]))
    assert shown == set(layout["configById"])                      # every panel placed, none dangling
    assert "/hil/pack.cell_current" in s
