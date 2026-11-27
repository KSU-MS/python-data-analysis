"""Run the VCU controller (controller/vcu_control.cpp) inside the sim.

The controller only sees what the car's VCU would: wheel speed sensor edges, motor
rpm, inverter torque feedback and VectorNav frames at their CAN rates (with latency
and noise), and the driver's torque request. It commands torque at its own control
period, held between commands. Its car model comes from a `Car` you give it, which
can differ from the car being simulated, to test robustness to wrong coefficients.

    ctrl = FirmwareController(mode="mu_adaptive")
    res = simulate(car, controller=ctrl)
    res.controller_log      # what the controller saw and decided, per tick
    ctrl.learned            # radii / accel bias / mu to carry into the next run
    write_log("run.vcu", ctrl.records())   # the same binary records the car writes
"""
import ctypes
import math
import subprocess
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .params import Car
from .powertrain import MotorMap

CONTROLLER_DIR = Path(__file__).resolve().parent.parent / "controller"
LIB_PATH = CONTROLLER_DIR / "build" / "libvcu_control.so"

MODES = {"passthrough": 0, "slip_pi": 1, "mu_adaptive": 2}
MODE_GAINS = {
    "slip_pi": dict(kp=0.05, ki=0.3, recover_gain=3.0),
    "mu_adaptive": dict(kp=0.1, ki=0.3, recover_gain=1.0),
}
LOG_VERSION = 2
FAULTS = {"gear_ratio": 1 << 0}  # vcu_fault bits
_f, _u8, _u16, _u32, _i32 = (ctypes.c_float, ctypes.c_uint8, ctypes.c_uint16,
                             ctypes.c_uint32, ctypes.c_int32)


class Config(ctypes.Structure):
    _fields_ = [
        ("mode", _i32),
        ("mass", _f), ("cg_height", _f), ("wheelbase", _f), ("cg_to_front", _f),
        ("tire_radius", _f), ("final_drive", _f), ("driveline_eff", _f), ("rear_inertia", _f),
        ("cla", _f), ("cda", _f), ("rho", _f), ("crr", _f), ("aero_front", _f),
        ("torque_peak", _f),
        ("ws_teeth", _i32), ("ws_min_window_us", _u32), ("ws_timeout_us", _u32),
        ("ws_valid_speed", _f), ("vn_vel_gain", _f), ("ws_vel_gain", _f),
        ("accel_filter_tau", _f),
        ("radius_cal_speed", _f), ("radius_cal_rate", _f), ("radius_tolerance", _f),
        ("coast_torque", _f), ("accel_bias_rate", _f),
        ("control_period_us", _u32),
        ("slip_target", _f), ("slip_speed_floor", _f), ("kp", _f), ("ki", _f),
        ("recover_gain", _f), ("scale_max", _f), ("mu_init", _f), ("mu_min", _f),
        ("mu_max", _f), ("mu_adapt_rate", _f), ("mu_transfer_rate", _f),
        ("torque_track_tol", _f),
        ("ratio_check_revs", _f), ("ratio_tolerance", _f), ("ratio_fail_count", _i32),
    ]


class Learned(ctypes.Structure):
    _fields_ = [("r_front", _f), ("r_rear", _f), ("accel_bias", _f), ("mu", _f)]


class Inputs(ctypes.Structure):
    _fields_ = [
        ("now_us", _u32),
        ("ws_edges", _u16 * 4),
        ("ws_last_edge_us", _u32 * 4),
        ("motor_rpm_new", _u8), ("motor_rpm", _f),
        ("motor_torque_new", _u8), ("motor_torque_fb", _f),
        ("vn_new", _u8), ("vn_vel_x", _f), ("vn_accel_x", _f),
        ("driver_torque_nm", _f),
    ]


class Outputs(ctypes.Structure):
    _fields_ = [
        ("torque_cmd_nm", _f), ("wheel_omega", _f * 4), ("v_est", _f), ("accel_est", _f),
        ("slip", _f), ("scale", _f), ("torque_ff_nm", _f), ("gear_ratio", _f), ("faults", _u32),
        ("learned", Learned),
    ]


# vcu_log_record_t: packed, little-endian
LOG_DTYPE = np.dtype([
    ("version", "<u2"), ("mode", "<u2"), ("now_us", "<u4"),
    ("ws_edges", "<u2", 4), ("ws_last_edge_us", "<u4", 4),
    ("motor_rpm_new", "u1"), ("motor_torque_new", "u1"), ("vn_new", "u1"), ("faults", "u1"),
    ("motor_rpm", "<f4"), ("motor_torque_fb", "<f4"), ("vn_vel_x", "<f4"), ("vn_accel_x", "<f4"),
    ("driver_torque_nm", "<f4"), ("torque_cmd_nm", "<f4"), ("v_est", "<f4"), ("slip", "<f4"),
    ("scale", "<f4"), ("mu", "<f4"),
])
LogRecord = ctypes.c_char * LOG_DTYPE.itemsize

_lib = None


def load_library():
    """Build controller/ if its sources are newer than the library, then load it.

    Checks every shared struct's size against the C side, so the ctypes mirrors
    here can't silently drift from vcu_control.h.
    """
    global _lib
    if _lib is None:
        subprocess.run(["make", "-C", str(CONTROLLER_DIR), "-s"], check=True)
        lib = ctypes.CDLL(str(LIB_PATH))
        lib.vcu_state_size.restype = ctypes.c_size_t
        lib.vcu_struct_size.restype = ctypes.c_size_t
        lib.vcu_struct_size.argtypes = [ctypes.c_int]
        lib.vcu_init.argtypes = [ctypes.c_void_p, ctypes.POINTER(Config),
                                 ctypes.POINTER(Learned), ctypes.c_uint32]
        lib.vcu_step.argtypes = [ctypes.c_void_p, ctypes.POINTER(Inputs), ctypes.POINTER(Outputs)]
        lib.vcu_make_log_record.argtypes = [ctypes.c_void_p, ctypes.POINTER(Inputs),
                                            ctypes.POINTER(Outputs), ctypes.c_void_p]
        for i, struct in enumerate([Config, Learned, Inputs, Outputs, LogRecord]):
            c_size = lib.vcu_struct_size(i)
            if c_size != ctypes.sizeof(struct):
                raise RuntimeError(f"{struct.__name__} is {ctypes.sizeof(struct)} bytes in "
                                   f"Python but {c_size} in vcu_control.h")
        _lib = lib
    return _lib


@dataclass
class Sensors:
    """What the VCU receives, and how imperfectly. Times in seconds."""
    ws_teeth: int = 19
    motor_rpm_period: float = 0.01     # inverter motor position frame
    motor_torque_period: float = 0.01  # inverter torque feedback frame
    vn_period: float = 0.01            # VectorNav frames
    can_latency: float = 0.002         # sample -> VCU receive
    vn_vel_noise: float = 0.05         # m/s 1-sigma
    vn_accel_noise: float = 0.15       # m/s^2 1-sigma
    vn_accel_bias: float = 0.0         # m/s^2
    tick: float = 0.001                # VCU loop period
    staged_time: float = 1.0           # s sitting on the line before launch
    seed: int = 0


WHEELS = ("fl", "fr", "rl", "rr")
LEARNED = ("r_front", "r_rear", "accel_bias", "mu")
INPUT_COLUMNS = ["now_us", *[f"ws_edges_{w}" for w in WHEELS],
                 *[f"ws_last_edge_us_{w}" for w in WHEELS],
                 "motor_rpm_new", "motor_rpm", "motor_torque_new", "motor_torque_fb",
                 "vn_new", "vn_vel_x", "vn_accel_x", "driver_torque_nm"]
OUTPUT_COLUMNS = ["t", "torque_cmd", "v_est", "accel_est", "slip", "scale", "torque_ff",
                  "gear_ratio", "faults", *[f"omega_{w}" for w in WHEELS], *LEARNED]


@dataclass
class FirmwareController:
    """The C++ controller plus sensor models, called by `simulate` every plant step.

    belief: the car model the controller is configured with.
    overrides: any vcu_config_t field (e.g. slip_target, kp, mu_init).
    learned: radii / accel bias / mu from a previous run (like EEPROM); after a run
        `self.learned` holds the updated values.
    """
    mode: str = "mu_adaptive"
    belief: Car = field(default_factory=Car)
    sensors: Sensors = field(default_factory=Sensors)
    overrides: dict = field(default_factory=dict)
    learned: dict | None = None

    def config(self, motor: MotorMap) -> Config:
        b = self.belief
        gr = b.final_drive
        c = dict(
            mode=MODES[self.mode],
            mass=b.mass, cg_height=b.cg_height, wheelbase=b.wheelbase,
            cg_to_front=b.cg_to_front, tire_radius=b.tire_radius, final_drive=gr,
            driveline_eff=b.driveline_eff,
            rear_inertia=2 * b.wheel_inertia + b.rear_driveline_inertia + motor.rotor_inertia * gr * gr,
            cla=b.cla, cda=b.cda, rho=b.rho, crr=b.crr, aero_front=b.aero_front,
            torque_peak=motor.peak_torque,
            ws_teeth=self.sensors.ws_teeth, ws_min_window_us=10_000, ws_timeout_us=500_000,
            ws_valid_speed=3.0, vn_vel_gain=0.2, ws_vel_gain=0.05, accel_filter_tau=0.02,
            radius_cal_speed=5.0, radius_cal_rate=1.0, radius_tolerance=0.05,
            coast_torque=2.0, accel_bias_rate=2.0,
            control_period_us=5_000,
            slip_target=0.10, slip_speed_floor=1.0,
            scale_max=1.5, mu_init=1.2, mu_min=0.6, mu_max=2.5, mu_adapt_rate=1.0,
            mu_transfer_rate=1.0, torque_track_tol=5.0,
            # final drive options differ by 10%+. Over 2 rear wheel revs the measured
            # ratio is within 5% on a correct car (worst of 120 Monte Carlo runs, at
            # launch, where rpm frames lag); a wrong ratio faults within ~6 m
            ratio_check_revs=2.0, ratio_tolerance=0.08, ratio_fail_count=2,
        )
        # gains swept per mode with these sensor models over grip 0.85-1.15 (the
        # Python SlipTC gains, kp 0.3 ki 1, limit-cycle once CAN rates, latency and
        # inverter lag are in the loop)
        c.update(MODE_GAINS.get(self.mode, MODE_GAINS["slip_pi"]))
        c.update(self.overrides)
        return Config(**c)

    def reset(self, car: Car, motor: MotorMap):
        """Called by simulate() with the true car, before the run."""
        lib = load_library()
        self._lib = lib
        self._state = ctypes.create_string_buffer(lib.vcu_state_size())
        self._cfg = self.config(motor)
        self._learned_in = _learned_struct(self.learned)
        lib.vcu_init(self._state, ctypes.byref(self._cfg), ctypes.byref(self._learned_in), 0)
        self._in, self._out, self._rec = Inputs(), Outputs(), LogRecord()
        self._rng = np.random.default_rng(self.sensors.seed)
        self._r = car.tire_radius        # true radius drives the true sensors
        self._gr = car.final_drive
        self._tooth = 2 * math.pi / self.sensors.ws_teeth
        self._t0 = self.sensors.staged_time
        self._theta = [0.0] * 4
        self._edges = [0] * 4
        self._last_edge_us = [0] * 4
        self._next = {"tick": 0.0, "motor": 0.0, "torque": 0.0, "vn": 0.0}
        self._pending = deque()          # (deliver_time, kind, values), controller clock
        self.torque_cmd = 0.0
        self.log, self.inputs, self._records = [], [], []
        # sit staged on the line: the controller sees a stationary car
        t, dt = 0.0, self.sensors.tick
        while t < self._t0 - 1e-9:
            self._advance(t, dt, (0.0, 0.0, 0.0, 0.0), 0.0, 0.0, 0.0, 0.0, 0.0)
            t += dt

    def __call__(self, t, dt, v, omega_rear, accel, motor_torque, driver_torque):
        """Advance sensors by one plant step from t to t + dt; return the torque command.
        Straight line (sim.simulate): front wheels roll at v, rears at omega_rear."""
        w_front = v / self._r
        return self._advance(t + self._t0, dt, (w_front, w_front, omega_rear, omega_rear),
                             omega_rear * self._gr * 60 / (2 * math.pi), v, accel,
                             motor_torque, driver_torque)

    def step(self, t, dt, wheel_omegas, motor_rpm, vel_x, accel_x, motor_torque, driver_torque):
        """Same, with each wheel's speed (rad/s; FL, FR, RL, RR) and the motor's, for
        the double-track model (drive.py). vel_x / accel_x: body forward velocity and
        acceleration, what the VectorNav frames carry here."""
        return self._advance(t + self._t0, dt, wheel_omegas, motor_rpm, vel_x, accel_x,
                             motor_torque, driver_torque)

    def _advance(self, t, dt, omegas, motor_rpm, v, accel, motor_torque, driver_torque):
        s = self.sensors
        # wheel speed sensors: count tooth crossings, interpolate edge times
        for i, w in enumerate(omegas):
            th0 = self._theta[i]
            th1 = th0 + w * dt
            n0, n1 = int(th0 / self._tooth), int(th1 / self._tooth)
            if n1 > n0:
                frac = (n1 * self._tooth - th0) / (th1 - th0)
                self._edges[i] += n1 - n0
                self._last_edge_us[i] = int((t + frac * dt) * 1e6)
            self._theta[i] = th1

        # CAN frames: sample now, deliver after latency
        nxt = self._next
        if t >= nxt["motor"]:
            nxt["motor"] += s.motor_rpm_period
            self._pending.append((t + s.can_latency, "motor", motor_rpm))
        if t >= nxt["torque"]:
            nxt["torque"] += s.motor_torque_period
            self._pending.append((t + s.can_latency, "torque", motor_torque))
        if t >= nxt["vn"]:
            nxt["vn"] += s.vn_period
            vel = v + self._rng.normal(0, s.vn_vel_noise)
            acc = accel + s.vn_accel_bias + self._rng.normal(0, s.vn_accel_noise)
            self._pending.append((t + s.can_latency, "vn", (vel, acc)))

        if t + dt < nxt["tick"]:
            return self.torque_cmd
        nxt["tick"] += s.tick
        now = t + dt
        inp = self._in
        inp.now_us = int(round(now * 1e6))
        for i in range(4):
            inp.ws_edges[i] = self._edges[i]
            inp.ws_last_edge_us[i] = self._last_edge_us[i]
            self._edges[i] = 0
        inp.motor_rpm_new = inp.motor_torque_new = inp.vn_new = 0
        while self._pending and self._pending[0][0] <= now:
            _, kind, val = self._pending.popleft()
            if kind == "motor":
                inp.motor_rpm_new, inp.motor_rpm = 1, val
            elif kind == "torque":
                inp.motor_torque_new, inp.motor_torque_fb = 1, val
            else:
                inp.vn_new, (inp.vn_vel_x, inp.vn_accel_x) = 1, val
        inp.driver_torque_nm = driver_torque
        self.inputs.append(_input_row(inp))
        self._lib.vcu_step(self._state, ctypes.byref(inp), ctypes.byref(self._out))
        self._lib.vcu_make_log_record(self._state, ctypes.byref(inp), ctypes.byref(self._out),
                                      self._rec)
        self._records.append(bytes(self._rec))
        o = self._out
        self.learned = {k: getattr(o.learned, k) for k in LEARNED}
        # inverter command frame carries 0.1 Nm
        self.torque_cmd = round(o.torque_cmd_nm * 10) / 10
        self.log.append(_output_row(inp.now_us * 1e-6 - self._t0, o))
        return self.torque_cmd

    def log_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.log, columns=OUTPUT_COLUMNS)

    def input_frame(self) -> pd.DataFrame:
        """Every input handed to the controller: the same thing a car log should record."""
        return pd.DataFrame(self.inputs, columns=INPUT_COLUMNS)

    def records(self) -> np.ndarray:
        """The vcu_log_record_t stream, exactly as the car would write it to SD."""
        return np.frombuffer(b"".join(self._records), dtype=LOG_DTYPE)


def _learned_struct(learned: dict | None) -> Learned:
    return Learned(**{k: float((learned or {}).get(k, 0.0)) for k in LEARNED})


def _input_row(inp: Inputs) -> tuple:
    return (inp.now_us, *inp.ws_edges, *inp.ws_last_edge_us, inp.motor_rpm_new, inp.motor_rpm,
            inp.motor_torque_new, inp.motor_torque_fb, inp.vn_new, inp.vn_vel_x, inp.vn_accel_x,
            inp.driver_torque_nm)


def _output_row(t: float, o: Outputs) -> tuple:
    return (t, o.torque_cmd_nm, o.v_est, o.accel_est, o.slip, o.scale, o.torque_ff_nm,
            o.gear_ratio, o.faults, *o.wheel_omega, *(getattr(o.learned, k) for k in LEARNED))


def replay(inputs: pd.DataFrame, config: Config, learned: dict | None = None,
           t_offset: float = 0.0) -> pd.DataFrame:
    """Run the controller on recorded inputs (input_frame(), or read_log() of a car
    log). Same code, config, learned values and inputs -> identical outputs."""
    lib = load_library()
    state = ctypes.create_string_buffer(lib.vcu_state_size())
    lib.vcu_init(state, ctypes.byref(config), ctypes.byref(_learned_struct(learned)), 0)
    inp, out, rows = Inputs(), Outputs(), []
    for row in inputs[INPUT_COLUMNS].itertuples(index=False):
        inp.now_us = row[0]
        for i in range(4):
            inp.ws_edges[i] = row[1 + i]
            inp.ws_last_edge_us[i] = row[5 + i]
        (inp.motor_rpm_new, inp.motor_rpm, inp.motor_torque_new, inp.motor_torque_fb,
         inp.vn_new, inp.vn_vel_x, inp.vn_accel_x, inp.driver_torque_nm) = row[9:17]
        lib.vcu_step(state, ctypes.byref(inp), ctypes.byref(out))
        rows.append(_output_row(inp.now_us * 1e-6 - t_offset, out))
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def write_log(path, records: np.ndarray):
    Path(path).write_bytes(records.astype(LOG_DTYPE).tobytes())


def read_log(path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read vcu_log_record_t records. Returns (inputs in INPUT_COLUMNS for replay(),
    the controller's logged outputs)."""
    rec = np.fromfile(path, dtype=LOG_DTYPE)
    if len(rec) and (rec["version"] != LOG_VERSION).any():
        raise ValueError(f"log version {set(rec['version'])} != {LOG_VERSION}")
    cols = {"now_us": rec["now_us"]}
    for i, w in enumerate(WHEELS):
        cols[f"ws_edges_{w}"] = rec["ws_edges"][:, i]
    for i, w in enumerate(WHEELS):
        cols[f"ws_last_edge_us_{w}"] = rec["ws_last_edge_us"][:, i]
    for k in INPUT_COLUMNS[9:]:
        cols[k] = rec[k]
    outputs = pd.DataFrame({k: rec[k] for k in
                            ["now_us", "torque_cmd_nm", "v_est", "slip", "scale", "mu", "faults"]})
    return pd.DataFrame(cols)[INPUT_COLUMNS], outputs


def warmup(car: Car, controller: FirmwareController, motor="228HV", tire=None,
           drive_torque: float = 80.0, drive_time: float = 3.0, coast_time: float = 4.0,
           **sim_kwargs) -> dict:
    """Drive gently then coast, like getting to the start line, so the controller can
    calibrate tire radii and VectorNav bias. Returns the learned values; pass them as
    FirmwareController(learned=...) for the run."""
    from .sim import simulate

    fc = replace(controller, mode="passthrough")
    simulate(car, motor, tire=tire, controller=fc, t_max=drive_time + coast_time,
             torque_request=lambda t, v: drive_torque if t < drive_time else 0.0, **sim_kwargs)
    return fc.learned
