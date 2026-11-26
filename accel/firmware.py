"""Run the VCU controller (controller/vcu_control.cpp) inside the sim.

The controller only sees what the car's VCU would: wheel speed sensor edges, motor
rpm and VectorNav frames at their CAN rates (with latency and noise), and the
driver's torque request. It commands torque at its own control period, held
between commands. Its car model comes from a `Car` you give it, which can differ
from the car being simulated, to test robustness to wrong coefficients.

    ctrl = FirmwareController(mode="mu_adaptive")
    res = simulate(car, controller=ctrl)
    res.controller_log  # what the controller saw and decided, per tick
"""
import ctypes
import math
import subprocess
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .params import Car
from .powertrain import MotorMap

CONTROLLER_DIR = Path(__file__).resolve().parent.parent / "controller"
LIB_PATH = CONTROLLER_DIR / "build" / "libvcu_control.so"

MODES = {"passthrough": 0, "slip_pi": 1, "mu_adaptive": 2}


class Config(ctypes.Structure):
    _fields_ = [
        ("mode", ctypes.c_int32),
        ("mass", ctypes.c_float), ("cg_height", ctypes.c_float),
        ("wheelbase", ctypes.c_float), ("cg_to_front", ctypes.c_float),
        ("tire_radius", ctypes.c_float), ("final_drive", ctypes.c_float),
        ("driveline_eff", ctypes.c_float), ("rear_inertia", ctypes.c_float),
        ("cla", ctypes.c_float), ("cda", ctypes.c_float), ("rho", ctypes.c_float),
        ("crr", ctypes.c_float), ("aero_front", ctypes.c_float),
        ("torque_peak", ctypes.c_float),
        ("ws_teeth", ctypes.c_int32), ("ws_min_window_us", ctypes.c_uint32),
        ("ws_timeout_us", ctypes.c_uint32), ("ws_valid_speed", ctypes.c_float),
        ("vn_vel_gain", ctypes.c_float), ("ws_vel_gain", ctypes.c_float),
        ("accel_filter_tau", ctypes.c_float),
        ("control_period_us", ctypes.c_uint32),
        ("slip_target", ctypes.c_float), ("slip_speed_floor", ctypes.c_float),
        ("kp", ctypes.c_float), ("ki", ctypes.c_float),
        ("recover_gain", ctypes.c_float), ("scale_max", ctypes.c_float),
        ("mu_init", ctypes.c_float), ("mu_min", ctypes.c_float),
        ("mu_max", ctypes.c_float), ("mu_adapt_rate", ctypes.c_float),
    ]


class Inputs(ctypes.Structure):
    _fields_ = [
        ("now_us", ctypes.c_uint32),
        ("ws_edges", ctypes.c_uint16 * 4),
        ("ws_last_edge_us", ctypes.c_uint32 * 4),
        ("motor_rpm_new", ctypes.c_uint8), ("motor_rpm", ctypes.c_float),
        ("vn_new", ctypes.c_uint8), ("vn_vel_x", ctypes.c_float),
        ("vn_accel_x", ctypes.c_float),
        ("driver_torque_nm", ctypes.c_float),
    ]


class Outputs(ctypes.Structure):
    _fields_ = [
        ("torque_cmd_nm", ctypes.c_float),
        ("wheel_omega", ctypes.c_float * 4),
        ("v_est", ctypes.c_float), ("accel_est", ctypes.c_float),
        ("slip", ctypes.c_float), ("scale", ctypes.c_float),
        ("mu_hat", ctypes.c_float), ("torque_ff_nm", ctypes.c_float),
    ]


_lib = None


def load_library():
    """Build controller/ if its sources are newer than the library, then load it."""
    global _lib
    if _lib is None:
        subprocess.run(["make", "-C", str(CONTROLLER_DIR), "-s"], check=True)
        lib = ctypes.CDLL(str(LIB_PATH))
        lib.vcu_state_size.restype = ctypes.c_size_t
        lib.vcu_init.argtypes = [ctypes.c_void_p, ctypes.POINTER(Config), ctypes.c_uint32]
        lib.vcu_step.argtypes = [ctypes.c_void_p, ctypes.POINTER(Inputs), ctypes.POINTER(Outputs)]
        _lib = lib
    return _lib


@dataclass
class Sensors:
    """What the VCU receives, and how imperfectly. Times in seconds."""
    ws_teeth: int = 19
    motor_rpm_period: float = 0.01     # inverter motor position frame
    vn_period: float = 0.01            # VectorNav frames
    can_latency: float = 0.002         # sample -> VCU receive
    vn_vel_noise: float = 0.05         # m/s 1-sigma
    vn_accel_noise: float = 0.15       # m/s^2 1-sigma
    vn_accel_bias: float = 0.0         # m/s^2
    tick: float = 0.001                # VCU loop period
    seed: int = 0


@dataclass
class FirmwareController:
    """The C++ controller plus sensor models, called by `simulate` every plant step.

    `belief` is the car model the controller is configured with; `overrides` set
    any vcu_config_t field (e.g. slip_target, kp, mu_init).
    """
    mode: str = "mu_adaptive"
    belief: Car = field(default_factory=Car)
    sensors: Sensors = field(default_factory=Sensors)
    overrides: dict = field(default_factory=dict)

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
            control_period_us=5_000,
            # gains swept with these sensor models: the Python SlipTC gains (kp 0.3,
            # ki 1) limit-cycle once CAN rates, latency and inverter lag are in the loop
            slip_target=0.10, slip_speed_floor=1.0, kp=0.1, ki=0.3, recover_gain=3.0,
            scale_max=1.3, mu_init=1.2, mu_min=0.6, mu_max=2.5, mu_adapt_rate=1.0,
        )
        c.update(self.overrides)
        return Config(**c)

    def reset(self, car: Car, motor: MotorMap):
        """Called by simulate() with the true car, before the run."""
        lib = load_library()
        self._lib = lib
        self._state = ctypes.create_string_buffer(lib.vcu_state_size())
        self._cfg = self.config(motor)
        lib.vcu_init(self._state, ctypes.byref(self._cfg), 0)
        self._in, self._out = Inputs(), Outputs()
        self._rng = np.random.default_rng(self.sensors.seed)
        self._r = car.tire_radius        # true radius drives the true sensors
        self._gr = car.final_drive
        self._tooth = 2 * math.pi / self.sensors.ws_teeth
        self._theta = [0.0] * 4
        self._edges = [0] * 4
        self._last_edge_us = [0] * 4
        self._next_tick = 0.0
        self._next_motor = 0.0
        self._next_vn = 0.0
        self._pending = deque()          # (deliver_time, kind, values)
        self._motor = None
        self._vn = None
        self.torque_cmd = 0.0
        self.log = []
        self.inputs = []

    def __call__(self, t, dt, v, omega_rear, accel, driver_torque):
        """Advance sensors by one plant step from t to t + dt; return the torque command."""
        s = self.sensors
        # wheel speed sensors: count tooth crossings, interpolate edge times
        omegas = (v / self._r, v / self._r, omega_rear, omega_rear)
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
        if t >= self._next_motor:
            self._next_motor += s.motor_rpm_period
            self._pending.append((t + s.can_latency, "motor", omega_rear * self._gr * 60 / (2 * math.pi)))
        if t >= self._next_vn:
            self._next_vn += s.vn_period
            vel = v + self._rng.normal(0, s.vn_vel_noise)
            acc = accel + s.vn_accel_bias + self._rng.normal(0, s.vn_accel_noise)
            self._pending.append((t + s.can_latency, "vn", (vel, acc)))

        if t + dt < self._next_tick:
            return self.torque_cmd
        self._next_tick += s.tick
        now = t + dt
        inp = self._in
        inp.now_us = int(now * 1e6)
        for i in range(4):
            inp.ws_edges[i] = self._edges[i]
            inp.ws_last_edge_us[i] = self._last_edge_us[i]
            self._edges[i] = 0
        inp.motor_rpm_new = inp.vn_new = 0
        while self._pending and self._pending[0][0] <= now:
            _, kind, val = self._pending.popleft()
            if kind == "motor":
                inp.motor_rpm_new, inp.motor_rpm = 1, val
            else:
                inp.vn_new, (inp.vn_vel_x, inp.vn_accel_x) = 1, val
        inp.driver_torque_nm = driver_torque
        self.inputs.append(_input_row(inp))
        self._lib.vcu_step(self._state, ctypes.byref(inp), ctypes.byref(self._out))
        o = self._out
        # inverter command frame carries 0.1 Nm
        self.torque_cmd = round(o.torque_cmd_nm * 10) / 10
        self.log.append((inp.now_us * 1e-6, o.torque_cmd_nm, o.v_est, o.accel_est, o.slip,
                         o.scale, o.mu_hat, o.torque_ff_nm, *o.wheel_omega))
        return self.torque_cmd

    def log_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.log, columns=OUTPUT_COLUMNS)

    def input_frame(self) -> pd.DataFrame:
        """Every input handed to the controller: the same thing a car log should record."""
        return pd.DataFrame(self.inputs, columns=INPUT_COLUMNS)


WHEELS = ("fl", "fr", "rl", "rr")
INPUT_COLUMNS = ["now_us", *[f"ws_edges_{w}" for w in WHEELS],
                 *[f"ws_last_edge_us_{w}" for w in WHEELS],
                 "motor_rpm_new", "motor_rpm", "vn_new", "vn_vel_x", "vn_accel_x",
                 "driver_torque_nm"]
OUTPUT_COLUMNS = ["t", "torque_cmd", "v_est", "accel_est", "slip", "scale", "mu_hat",
                  "torque_ff", *[f"omega_{w}" for w in WHEELS]]


def _input_row(inp: Inputs) -> tuple:
    return (inp.now_us, *inp.ws_edges, *inp.ws_last_edge_us, inp.motor_rpm_new, inp.motor_rpm,
            inp.vn_new, inp.vn_vel_x, inp.vn_accel_x, inp.driver_torque_nm)


def replay(inputs: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Run the controller on recorded inputs (from input_frame(), or decoded from a car
    log in the same columns). Same code, same inputs -> identical outputs."""
    lib = load_library()
    state = ctypes.create_string_buffer(lib.vcu_state_size())
    lib.vcu_init(state, ctypes.byref(config), 0)
    inp, out, rows = Inputs(), Outputs(), []
    for row in inputs[INPUT_COLUMNS].itertuples(index=False):
        inp.now_us = row[0]
        for i in range(4):
            inp.ws_edges[i] = row[1 + i]
            inp.ws_last_edge_us[i] = row[5 + i]
        (inp.motor_rpm_new, inp.motor_rpm, inp.vn_new, inp.vn_vel_x, inp.vn_accel_x,
         inp.driver_torque_nm) = row[9:15]
        lib.vcu_step(state, ctypes.byref(inp), ctypes.byref(out))
        rows.append((inp.now_us * 1e-6, out.torque_cmd_nm, out.v_est, out.accel_est, out.slip,
                     out.scale, out.mu_hat, out.torque_ff_nm, *out.wheel_omega))
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)
