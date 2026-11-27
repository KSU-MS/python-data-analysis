"""The rig's side of the protocol (cardata/hil/cell_load.dbc), emulated: a stand-in for
the hardware in ~/dev/kms-cell-load with a CellModel as the cell. The host's `can`
backend talks to it exactly as to the real rig, so `--pack virtual-cell` exercises the
whole path (and is a reference for the rig's firmware: states, watchdog, limits).

Emulated hardware: the setpoint takes effect after `latency`, slewed at the setpoint's
limit (or 10 kA/s); measurements carry noise and the DBC's resolution.

    rig = RigEmulator(can.Bus(interface="virtual", channel="hil"), cell)
    rig.start_thread()          # 1 kHz against the wall clock
    # or, in lockstep with a sim: rig.step(now) after each host message
"""
import threading
import time
from collections import deque

import cantools
import numpy as np

from kmslap import dirs

DBC = dirs.REPO / "cardata" / "hil" / "cell_load.dbc"
STATES = {"BOOT": 0, "IDLE": 1, "ARMED": 2, "RUNNING": 3, "FAULT": 4}
COMMANDS = {"NONE": 0, "ARM": 1, "DISARM": 2, "CLEAR_FAULTS": 3, "ZERO_CURRENT_OFFSET": 4}
WATCHDOG_S = 0.020


def load_dbc():
    return cantools.database.load_file(str(DBC))


def checksum(data: bytes) -> int:
    return sum(data[:7]) & 0xFF


class RigEmulator:
    def __init__(self, bus, cell, *, latency: float = 0.001, slew: float = 10_000.0,
                 noise_v: float = 0.3e-3, noise_i: float = 0.05, seed: int = 0):
        import can
        self._can = can
        self.bus, self.cell, self.latency, self.default_slew = bus, cell, latency, slew
        self.noise_v, self.noise_i = noise_v, noise_i
        self.db = load_dbc()
        self.rng = np.random.default_rng(seed)
        self.state = "IDLE"
        self.faults = set()
        self.limits = dict(cell_voltage_min=3.0, cell_voltage_max=4.28, discharge_current_max=330.0,
                           charge_current_max=240.0, cell_temp_max=60.0)
        self.setpoint, self.enable, self.slew, self.counter = 0.0, 0, slew, 0
        self.current = 0.0
        self.pending = deque()            # (time effective, setpoint)
        self.last_setpoint_t = None
        self.charge = self.energy = 0.0
        self.now = None
        self._meas_counter = 0
        self._next = {"status": 0.0, "temps": 0.0, "counters": 0.0}
        self._stop = threading.Event()
        self._thread = None

    # --- host messages
    def _receive(self, now):
        while (msg := self.bus.recv(timeout=0)) is not None:
            try:
                m = self.db.get_message_by_frame_id(msg.arbitration_id)
            except KeyError:
                continue
            if m.name == "LOAD_SETPOINT":
                if checksum(msg.data) != msg.data[7]:
                    continue
                d = m.decode(msg.data)
                self.last_setpoint_t = now
                self.counter = int(d["counter"])
                self.slew = d["slew_limit"] or self.default_slew
                self.enable = int(d["enable"])
                if self.state in ("ARMED", "RUNNING"):
                    self.state = "RUNNING" if self.enable else "ARMED"
                    self.pending.append((now + self.latency, d["current_setpoint"] if self.enable else 0.0))
            elif m.name == "LOAD_LIMITS":
                d = m.decode(msg.data)
                # the host can only tighten the built-in limits
                self.limits = dict(cell_voltage_min=max(3.0, d["cell_voltage_min"]),
                                   cell_voltage_max=min(4.28, d["cell_voltage_max"]),
                                   discharge_current_max=min(330.0, d["discharge_current_max"]),
                                   charge_current_max=min(240.0, d["charge_current_max"]),
                                   cell_temp_max=min(60.0, d["cell_temp_max"]))
            elif m.name == "LOAD_COMMAND":
                cmd = str(m.decode(msg.data)["command"])
                if cmd == "ARM" and self.state == "IDLE" and not self._check(0.0):
                    self.state, self.charge, self.energy = "ARMED", 0.0, 0.0
                    self.last_setpoint_t = now
                elif cmd == "DISARM" and self.state in ("ARMED", "RUNNING"):
                    self.state = "IDLE"
                elif cmd == "CLEAR_FAULTS" and self.state == "FAULT" and abs(self.current) < 1.0:
                    self.faults.clear()
                    self.state = "IDLE"
                self._send_status()

    def _check(self, current) -> set:
        v = self.cell.voltage(current)
        lim = self.limits
        f = set()
        if v > lim["cell_voltage_max"] + 0.02:
            f.add("over_voltage")
        if v < lim["cell_voltage_min"] - 0.02:
            f.add("under_voltage")
        if current > 1.05 * lim["discharge_current_max"] or -current > 1.05 * lim["charge_current_max"]:
            f.add("over_current")
        if self.cell.temp_c > lim["cell_temp_max"]:
            f.add("cell_over_temp")
        return f

    def _fault(self, causes):
        self.faults |= causes
        if self.state != "FAULT":
            self.state = "FAULT"
            self._send_status()

    # --- the rig's loop
    def step(self, now: float):
        dt = 0.0 if self.now is None else max(now - self.now, 0.0)
        self.now = now
        self._receive(now)
        if (self.state in ("ARMED", "RUNNING") and self.last_setpoint_t is not None
                and now - self.last_setpoint_t > WATCHDOG_S):
            self._fault({"watchdog"})
        target = self.setpoint
        while self.pending and self.pending[0][0] <= now:
            target = self.setpoint = self.pending.popleft()[1]
        if self.state != "RUNNING":
            target = 0.0
        step_max = self.slew * dt if dt > 0 else abs(target - self.current)
        self.current += float(np.clip(target - self.current, -step_max, step_max))
        if self.state not in ("ARMED", "RUNNING", "FAULT"):
            self.current = 0.0
        v = self.cell.step(self.current, dt) if dt > 0 else self.cell.voltage(self.current)
        self.charge += self.current * dt / 3600
        self.energy += self.current * v * dt / 3600
        if self.state in ("ARMED", "RUNNING") and (bad := self._check(self.current)):
            self._fault(bad)
        self._send("LOAD_MEASUREMENT", cell_voltage=v + self.rng.normal(0, self.noise_v),
                   cell_current=self.current + self.rng.normal(0, self.noise_i),
                   counter=self._meas_counter % 256, sample_age=200, setpoint_counter_echo=self.counter)
        self._meas_counter += 1
        for key, period, fn in (("status", 0.01, self._send_status), ("temps", 0.1, self._send_temps),
                                ("counters", 0.1, self._send_counters)):
            if now >= self._next[key]:
                self._next[key] = now + period
                fn()

    def _send(self, name, **values):
        m = self.db.get_message_by_name(name)
        data = bytearray(m.encode({s.name: values.get(s.name, 0) for s in m.signals}, strict=False))
        self.bus.send(self._can.Message(arbitration_id=m.frame_id, data=bytes(data), is_extended_id=False))

    def _send_status(self):
        f = {f"fault_{k}": 1 for k in self.faults}
        self._send("LOAD_STATUS", state=STATES[self.state], contactor_closed=int(self.state in ("ARMED", "RUNNING")),
                   bus_voltage=12.6, stage_temp=35, **f)

    def _send_temps(self):
        t = self.cell.temp_c
        self._send("LOAD_TEMPERATURES", cell_temp_1=t, cell_temp_2=t - 0.3, tab_temp_pos=t + 1.0, tab_temp_neg=t + 0.8)

    def _send_counters(self):
        self._send("LOAD_COUNTERS", charge=self.charge, energy=self.energy)

    # --- running on its own, against the wall clock
    def start_thread(self, period: float = 0.001):
        def loop():
            nxt = time.perf_counter()
            while not self._stop.is_set():
                self.step(time.perf_counter())
                nxt += period
                time.sleep(max(nxt - time.perf_counter(), 0))
        self._thread = threading.Thread(target=loop, name="rig-emulator", daemon=True)
        self._thread.start()

    def stop_thread(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
