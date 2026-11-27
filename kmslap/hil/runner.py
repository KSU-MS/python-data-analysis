"""kmslap hil: the lap sim with its battery in the loop, live in Foxglove.

    kmslap hil ac18                         # serve; press Start in Foxglove
    kmslap hil ac18 --pack virtual-cell     # the rig emulator on a virtual CAN bus
    kmslap hil ac18 --pack can --channel can0   # the real rig (~/dev/kms-cell-load)
    kmslap hil ac18 --autostart --once      # one run, no GUI needed

Idle, it serves Foxglove (ws://localhost:8765, layout written to <out>/hil.layout.json).
Start (a service; its request may set {"laps": n}) drives the track's minimum-time
plan with the plan's feedforward, regen on, in real time, the chosen backend behind the
car's pack; Stop ends the run (the rig to 0 A and disarmed), Abort too and says so,
Reset returns to idle (a rig fault is cleared when the next run arms it). Each
run ends with an MCAP in <out>/: the car's sensors in the logger's format
(reporting/mcap_log) plus the /hil topics, for Foxglove afterwards.
"""
import queue
import threading
import time
from pathlib import Path

import numpy as np

from kmslap.car.params import Car
from kmslap.car.powertrain import PACKS, Pack
from kmslap.hil import live as L
from kmslap.hil.backends import RigFault, SimPack, virtual_cell
from kmslap.models import twotrack
from kmslap.reporting import mcap_log

PUBLISH_HZ = 25


class HilRunner:
    def __init__(self, track_name: str = "ac18", pack: str = "sim", *, laps: int = 2, grip: float = 0.85,
                 margin: float = 0.35, regen: bool = True, realtime: bool = True, port: int = 8765,
                 out: str = "hil", can_interface: str = "socketcan", can_channel: str = "can0"):
        from kmslap.events import track as T
        self.track_name, self.pack_kind, self.laps, self.grip, self.margin = track_name, pack, laps, grip, margin
        self.regen, self.realtime = regen, realtime
        self.can_interface, self.can_channel = can_interface, can_channel
        self.car = Car()
        self.pack = Pack(PACKS["cosmx600v"])
        self.track = T.TRACK_NAMES[track_name]()
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "hil.layout.json").write_text(L.hil_layout())
        self.commands = queue.Queue()
        self.stop_requested = threading.Event()
        self.abort_requested = False
        self.run_state, self.lap = "idle", 0
        self.backend = None
        self.plan = None
        self.live = None
        self.port = port

    # --- services (called from the server's threads)
    def _start(self, req):
        if self.run_state == "running":
            return {"ok": False, "error": "already running"}
        self.commands.put(("start", req))
        return {"ok": True}

    def _stop(self, req, why="stopped"):
        self.abort_requested = why == "abort"
        self.stop_requested.set()
        self.commands.put((why, req))
        return {"ok": True}

    def _reset(self, req):
        self.stop_requested.set()
        self.commands.put(("reset", req))
        return {"ok": True}

    # --- publishing
    def _ns(self):
        return time.time_ns()

    def _publish_idle(self):
        ns = self._ns()
        self.live.log("/hil/run", L.hil_run()(state=L.RUN_STATES[self.run_state], lap=self.lap, laps=self.laps,
                                               rig_state=L.RIG_STATES.get(getattr(self.backend, "telemetry", {}).get(
                                                   "rig_state", "SIM" if self.pack_kind == "sim" else "IDLE"), 1)), ns)
        self.live.log("/sim/track", mcap_log.track_scene(self.track, self.car, getattr(self.plan, "line", None)), ns)

    def _hil_messages(self, t, out_vals, state_vals):
        tel = self.backend.telemetry
        cell_v, cell_i = tel.get("cell_voltage", 0.0), tel.get("cell_current", 0.0)
        pack_v, pack_i = state_vals["bus_v"], out_vals["bus_i"]
        pack = L.hil_pack()(
            cell_voltage=cell_v, cell_current=cell_i, cell_temp=tel.get("cell_temp", 0.0), soc=tel.get("soc", 0.0),
            cell_power_w=cell_v * cell_i, pack_voltage=pack_v, pack_current=pack_i, pack_power_kw=pack_v * pack_i / 1e3,
            cell_v_oc=tel.get("cell_v_oc", 0.0), cell_r_mohm=tel.get("cell_r", 0.0) * 1e3,
            latency_ms=float(np.nan_to_num(tel.get("latency_ms", 0.0))), lateness_ms=self._lateness * 1e3,
            rig_charge_ah=tel.get("rig_charge_ah", 0.0), rig_energy_wh=tel.get("rig_energy_wh", 0.0))
        run = L.hil_run()(state=L.RUN_STATES[self.run_state], lap=self.lap, laps=self.laps, elapsed_s=t,
                          rig_state=L.RIG_STATES.get(tel.get("rig_state", "SIM"), 1))
        return pack, run

    # --- a run
    def _backend(self):
        if self.pack_kind == "sim":
            return SimPack()
        if self.pack_kind == "virtual-cell":
            return virtual_cell(self.pack)
        if self.pack_kind == "can":
            import can
            from kmslap.hil.backends import CanBackend
            return CanBackend(can.Bus(interface=self.can_interface, channel=self.can_channel, bitrate=1_000_000))
        raise ValueError(f"no pack backend {self.pack_kind!r}")

    def run_once(self, laps: int | None = None) -> Path:
        from kmslap.driving.drive import drive
        from kmslap.driving.follower import PathFollower
        from kmslap.driving.policies import RegenPolicy
        from kmslap.events.laptime import plan

        laps = laps or self.laps
        self.laps = laps
        if self.plan is None:
            self.plan = plan(self.track, self.grip, self.margin, 2.0, self.car)
        p = self.plan
        follower = PathFollower.for_plan(p, car=self.car, laps=laps, plan_grip=self.grip)
        names = {k: i for i, k in enumerate(twotrack.STATES)}
        onames = {k: i for i, k in enumerate(twotrack.OUTPUTS)}
        self.backend = self._backend()
        self.stop_requested.clear()
        self.abort_requested = False
        self.run_state, self.lap, self._lateness = "running", 0, 0.0
        hil_events = []
        last = {"pub": -1.0, "loc": -1.0, "track": -10.0}
        lap_starts = []

        def on_tick(t, state, out, inputs):
            self._lateness = inputs.get("lateness", 0.0)
            if len(follower.lap_times) != len(lap_starts):
                lap_starts[:] = follower.lap_times
            self.lap = len(follower.lap_times)
            if t - last["pub"] >= 1 / PUBLISH_HZ:
                last["pub"] = t
                ns = self._ns()
                sv = {k: state[i] for k, i in names.items()}
                ov = {k: out[i] for k, i in onames.items()}
                row = {**sv, **ov, **inputs, "speed": float(np.hypot(sv["vx"], sv["vy"])),
                       "beta": float(np.arctan2(sv["vy"], max(sv["vx"], 0.1)))}
                lap_t = t - (lap_starts[-1] if lap_starts else 0.0)
                last_lap = lap_starts[-1] - (lap_starts[-2] if len(lap_starts) > 1 else 0.0) if lap_starts else 0.0
                pack, run = self._hil_messages(t, ov, sv)
                if self.live is not None:
                    self.live.log("/sim/tf", mcap_log.car_pose(sv["X"], sv["Y"], sv["psi"], ns), ns)
                    self.live.log("/sim/car", mcap_log.car_scene(self.car, inputs["steering_wheel"] / self.car.steer_ratio, ns), ns)
                    self.live.log("/sim/state", mcap_log.sim_state(row, self.car, self.lap, lap_t, last_lap), ns)
                    self.live.log("/hil/pack", pack, ns)
                    self.live.log("/hil/run", run, ns)
                    if self.track.origin is not None and t - last["loc"] >= 0.2:
                        from foxglove_schemas_protobuf.LocationFix_pb2 import LocationFix
                        last["loc"] = t
                        lon, lat = self.track.lonlat(sv["X"], sv["Y"])
                        fix = LocationFix(latitude=float(lat), longitude=float(lon), frame_id="car")
                        fix.timestamp.FromNanoseconds(ns)
                        self.live.log("/sim/location", fix, ns)
                    if t - last["track"] >= 2.0:
                        last["track"] = t
                        self.live.log("/sim/track", mcap_log.track_scene(self.track, self.car, p.line), ns)
                hil_events.append((t, "/hil/pack", pack))
                hil_events.append((t, "/hil/run", run))
            return self.stop_requested.is_set()

        res = None
        outcome = "finished"
        try:
            res = drive(follower, self.car, pack=self.pack, regen=RegenPolicy(lift_torque=40.0, brake_fraction=1.0)
                        if self.regen else None, pack_backend=self.backend, realtime=self.realtime, on_tick=on_tick,
                        psi0=float(p.line.heading[0] - p.beta_ff[0]), X0=float(p.line.x[0]), Y0=float(p.line.y[0]),
                        t_max=(laps + 1) * 4 * p.lap_time, log_period=0.005)
            if self.stop_requested.is_set():
                outcome = "stopped"
                if self.abort_requested:
                    print("[hil] aborted")
        except RigFault as e:
            outcome = "fault"
            print(f"[hil] {e}")
        finally:
            self.backend.stop()
            if hasattr(self.backend, "close"):
                self.backend.close()
        self.run_state = outcome
        laps_s = np.diff([0.0] + follower.lap_times)
        print(f"[hil] {outcome}: {len(laps_s)} laps {np.round(laps_s, 2).tolist()} ({self.pack_kind} pack)")
        if res is None or res.df.empty:
            return None
        path = self.out / f"{time.strftime('%Y%m%d_%H%M%S')}_{self.track_name}_{self.pack_kind}.mcap"
        mcap_log.write(res.df, self.track, path, self.car, plan_line=p.line, lap_times=follower.lap_times,
                       extra_events=hil_events, preroll_s=1.0)
        print(f"[hil] wrote {path}")
        return path

    def serve(self, autostart: bool = False, once: bool = False):
        self.live = L.Live({"start": self._start, "stop": self._stop, "reset": self._reset,
                            "abort": lambda req: self._stop(req, "abort")}, port=self.port)
        from foxglove_schemas_protobuf.FrameTransform_pb2 import FrameTransform
        from foxglove_schemas_protobuf.LocationFix_pb2 import LocationFix
        from foxglove_schemas_protobuf.SceneUpdate_pb2 import SceneUpdate
        for topic, cls in (("/sim/tf", FrameTransform), ("/sim/car", SceneUpdate), ("/sim/track", SceneUpdate),
                           ("/sim/location", LocationFix), ("/sim/state", mcap_log._sim_state_class()),
                           ("/hil/pack", L.hil_pack()), ("/hil/run", L.hil_run())):
            self.live.declare(topic, cls)
        print(f"[hil] Foxglove: connect to ws://localhost:{self.port}, layout {self.out / 'hil.layout.json'}")
        if autostart:
            self.commands.put(("start", {}))
        try:
            while True:
                try:
                    cmd, req = self.commands.get(timeout=0.5)
                except queue.Empty:
                    self._publish_idle()
                    continue
                if cmd == "start":
                    self.run_once(int(req.get("laps", self.laps)))
                    self._publish_idle()
                    if once:
                        return
                elif cmd == "reset":
                    self.run_state, self.lap = "idle", 0
                    self._publish_idle()
                elif cmd in ("stopped", "abort"):
                    pass          # handled inside the run (it checks the flag each tick)
        finally:
            self.live.close()
