"""Pack backends for drive(pack_backend=...): what stands behind the car's battery.

Each has `period` (s of sim time between exchanges), `start(pack, soc0, temp0)` and
`exchange(t, i_pack, v_bus)`, both returning the pack the car model should see for the
next period: {"v_oc", "r" (pack level), "soc", "temp"}; `stop()`; and `telemetry`
(the latest cell-level values, for the GUI).

  SimPack      the sim's own pack model, ticked from outside (no hardware)
  CanBackend   a rig on CAN (cardata/hil/cell_load.dbc): the real one, or
  virtual_cell(...)  the rig emulator (rig.py) on an in-process virtual CAN bus

The car's pack is series x parallel cells; a rig holds one: it gets the pack current /
parallel and its cell is scaled back up (OCV x series, R x series / parallel).
"""
import math
import time

import numpy as np

from kmslap.hil.cell import CellEstimator, CellModel
from kmslap.hil.rig import COMMANDS, STATES, checksum, load_dbc

STATE_NAMES = {v: k for k, v in STATES.items()}


class RigFault(RuntimeError):
    pass


class SimPack:
    """The model's pack (OCV(SOC) behind R(T), lumped thermal), outside the model."""
    period = 0.01

    def start(self, pack, soc0, temp0):
        self.pack, self.soc, self.temp = pack, soc0, temp0
        self.telemetry = {}
        return self._ext(0.0)

    def _ext(self, i):
        p = self.pack
        v_oc = p.series * float(np.interp(min(max(self.soc, 0), 1), p.ocv_soc, p.ocv_cell))
        r = p.resistance * math.exp(-p.resistance_temp_coeff * (self.temp - 25))
        cell_v = (v_oc - i * r) / p.series
        self.telemetry = dict(cell_voltage=cell_v, cell_current=i / p.parallel, cell_temp=self.temp, soc=self.soc,
                              cell_v_oc=v_oc / p.series, cell_r=r * p.parallel / p.series, rig_state="SIM",
                              faults="", latency_ms=0.0)
        return {"v_oc": v_oc, "r": r, "soc": self.soc, "temp": self.temp}

    def exchange(self, t, i_pack, v_bus):
        p = self.pack
        r = p.resistance * math.exp(-p.resistance_temp_coeff * (self.temp - 25))
        self.soc -= i_pack * self.period / (p.capacity_ah * 3600)
        self.temp += self.period * (i_pack * i_pack * r - p.h_a * (self.temp - p.ambient_c)) / p.thermal_mass
        return self._ext(i_pack)

    def stop(self):
        pass


class CanBackend:
    """The host side of the rig protocol."""
    period = 0.001

    def __init__(self, bus, *, lockstep=None, timeout: float = 1.0):
        """bus: a python-can Bus to the rig. lockstep: a RigEmulator stepped with the
        sim's clock after each message (tests, faster than real time); None: the rig
        runs on its own (hardware, or an emulator thread)."""
        import can
        self._can = can
        self.bus, self.lockstep, self.timeout = bus, lockstep, timeout
        self.db = load_dbc()
        self.counter = 0
        self.sent = {}
        self.meas = None
        self.status = {"state": "BOOT", "faults": ""}
        self.temps = {}
        self.counters = {}
        self.latency_ms = float("nan")
        self.telemetry = {}

    def _send(self, name, **values):
        m = self.db.get_message_by_name(name)
        data = bytearray(m.encode({s.name: values.get(s.name, 0) for s in m.signals}, strict=False))
        if name == "LOAD_SETPOINT":
            data[7] = checksum(data)
        self.bus.send(self._can.Message(arbitration_id=m.frame_id, data=bytes(data), is_extended_id=False))

    def _poll(self, now):
        if self.lockstep is not None:
            self.lockstep.step(now)
        new = False
        while (msg := self.bus.recv(timeout=0)) is not None:
            try:
                m = self.db.get_message_by_frame_id(msg.arbitration_id)
            except KeyError:
                continue
            d = m.decode(msg.data, decode_choices=False)
            if m.name == "LOAD_MEASUREMENT":
                self.meas, new = d, True
                sent = self.sent.get(int(d["setpoint_counter_echo"]))
                if sent is not None and self.lockstep is None:
                    self.latency_ms = (time.perf_counter() - sent) * 1e3
            elif m.name == "LOAD_STATUS":
                faults = [k[6:] for k, v in d.items() if k.startswith("fault_") and v]
                self.status = {"state": STATE_NAMES.get(int(d["state"]), "?"), "faults": ",".join(faults),
                               "bus_voltage": d["bus_voltage"]}
            elif m.name == "LOAD_TEMPERATURES":
                self.temps = d
            elif m.name == "LOAD_COUNTERS":
                self.counters = d
        return new

    def _clock(self):
        """Wall time, or in lockstep the handshake's own sim time, running up to 0."""
        if self.lockstep is None:
            return time.perf_counter()
        self._lock_t = min(self._lock_t + 1e-3, -1e-3)
        return self._lock_t

    def _wait(self, state):
        t_end = time.perf_counter() + self.timeout
        while self.status["state"] != state:
            if self.status["state"] == "FAULT":
                raise RigFault(f"rig fault: {self.status['faults']}")
            if time.perf_counter() > t_end:
                raise RigFault(f"rig didn't reach {state} (it's {self.status['state']})")
            self._poll(self._clock())
            if self.lockstep is None:
                time.sleep(0.001)

    def start(self, pack, soc0, temp0):
        self.pack = pack
        self._lock_t = -0.1
        self._send("LOAD_LIMITS", cell_voltage_min=pack.v_cell_min, cell_voltage_max=pack.v_cell_max,
                   discharge_current_max=pack.max_current / pack.parallel * 1.05,
                   charge_current_max=pack.max_charge_current / pack.parallel * 1.05, cell_temp_max=60)
        self._send("LOAD_COMMAND", command=COMMANDS["CLEAR_FAULTS"])      # harmless when there are none
        self._poll(self._clock())
        self._send("LOAD_COMMAND", command=COMMANDS["ARM"])
        self._wait("ARMED")
        self._send_setpoint(0.0)
        self._poll(self._clock())
        r_cell = pack.resistance * pack.parallel / pack.series
        self.est = CellEstimator(pack.ocv_soc, pack.ocv_cell, pack.capacity_ah / pack.parallel, soc0, r_cell)
        if self.meas is not None:
            self.est.v_oc = self.meas["cell_voltage"]
        self._t_last = None
        return self._ext()

    def _send_setpoint(self, i_cell, enable=1):
        # the field's range; the current limits are the rig's to enforce (it faults)
        i_cell = min(max(i_cell, -327.0), 327.0)
        self.counter = (self.counter + 1) % 16
        self.sent[self.counter] = time.perf_counter()
        self._send("LOAD_SETPOINT", current_setpoint=i_cell, enable=enable, counter=self.counter, slew_limit=0)

    def _ext(self):
        p = self.pack
        temp = self.temps.get("cell_temp_1", p.ambient_c)
        m = self.meas or {"cell_voltage": self.est.v_oc, "cell_current": 0.0}
        self.telemetry = dict(cell_voltage=m["cell_voltage"], cell_current=m["cell_current"], cell_temp=temp,
                              soc=self.est.soc, cell_v_oc=self.est.v_oc, cell_r=self.est.r,
                              rig_state=self.status["state"], faults=self.status["faults"], latency_ms=self.latency_ms,
                              rig_charge_ah=self.counters.get("charge", 0.0), rig_energy_wh=self.counters.get("energy", 0.0))
        return {"v_oc": p.series * self.est.v_oc, "r": self.est.r * p.series / p.parallel,
                "soc": self.est.soc, "temp": temp}

    def exchange(self, t, i_pack, v_bus):
        self._send_setpoint(i_pack / self.pack.parallel)
        if self._poll(t) and self.meas is not None:
            dt = self.period if self._t_last is None else t - self._t_last
            self._t_last = t
            self.est.update(self.meas["cell_voltage"], self.meas["cell_current"], dt)
        if self.status["state"] == "FAULT":
            raise RigFault(f"rig fault: {self.status['faults']}")
        return self._ext()

    def stop(self):
        for _ in range(3):
            try:
                self._send_setpoint(0.0, enable=0)
            except Exception:
                break
        self._send("LOAD_COMMAND", command=COMMANDS["DISARM"])
        self._poll(time.perf_counter() if self.lockstep is None else (self._t_last or 0.0) + 0.005)

    def close(self):
        rig = getattr(self, "rig", None)
        if rig is not None:
            rig.stop_thread()
            rig.bus.shutdown()
        self.bus.shutdown()


def virtual_cell(pack, soc0: float = 1.0, temp0: float = 25.0, *, lockstep: bool = False, **rig_kw):
    """A CanBackend to the rig emulator, with a CellModel cell, over a virtual CAN bus.
    lockstep: the emulator steps with sim time (faster than real time); otherwise it
    runs in its own thread against the wall clock (run drive(realtime=True))."""
    import can
    from kmslap.hil.rig import RigEmulator

    channel = f"kmslap-hil-{id(pack)}-{time.perf_counter_ns()}"
    host = can.Bus(interface="virtual", channel=channel, receive_own_messages=False)
    rig_bus = can.Bus(interface="virtual", channel=channel, receive_own_messages=False)
    cell = CellModel(pack.ocv_soc, pack.ocv_cell, capacity_ah=pack.capacity_ah / pack.parallel, soc=soc0,
                     temp_c=temp0, ambient_c=pack.ambient_c)
    rig = RigEmulator(rig_bus, cell, **rig_kw)
    backend = CanBackend(host, lockstep=rig if lockstep else None)
    if not lockstep:
        rig.start_thread()
    backend.rig = rig
    return backend
