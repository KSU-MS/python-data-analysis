"""The accel event plant as one CasADi model.

One step of the semi-implicit Euler integration is written symbolically, once:

    step(state, torque_cmd, params) -> (next state, outputs)

and used two ways:

- `CompiledStep`: CasADi generates C for the step, compiled once per (motor, tire
  shape) and called through ctypes, for `simulate()`'s loop with traction control or
  the VCU controller in it. (Calling a CasADi Function from Python costs ~20 us,
  four times a whole Python step; the compiled step costs ~1 us.)
- `rollout()`: the step unrolled over a run as a CasADi graph, for exact derivatives
  of the trajectory with respect to any parameter (system ID, optimization).

Tire grip and everything in `Car` are parameters, not baked in, so one compiled
model serves every Monte Carlo plant. The equations are in sim.py's docstring.
"""
import ctypes
import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path

import casadi as ca
import numpy as np

from .params import G, Car
from .powertrain import MotorMap, Pack

STATES = ["x", "v", "omega", "kappa", "a_filt", "t_lagged", "bus_v"]
OUTPUTS = ["accel", "fx", "fz_rear", "fz_front", "motor_torque", "bus_i", "drag", "motor_rpm"]
PARAMS = ["mass", "cg_height", "wheelbase", "front_weight", "tire_radius", "final_drive",
          "driveline_eff", "motor_eff", "cda", "cla", "aero_front", "rho", "crr",
          "wheel_inertia", "rear_driveline_inertia", "pitch_tau", "torque_tau", "power_limit",
          "relaxation_length", "tire_scale", "rotor_inertia", "pack_v_oc", "pack_resistance",
          "pack_max_current", "dt"]
RPM_PER_RADS = 60 / (2 * np.pi)

BUILD_DIR = Path(__file__).resolve().parent.parent / "build" / "casadi"


def params(car: Car, motor: MotorMap, pack: Pack, tire, relaxation_length: float,
           dt: float) -> np.ndarray:
    """Numeric parameter vector, in PARAMS order."""
    p = {f: getattr(car, f) for f in PARAMS if hasattr(car, f)}
    p.update(relaxation_length=relaxation_length, tire_scale=tire.scale,
             rotor_inertia=motor.rotor_inertia, pack_v_oc=pack.v_oc,
             pack_resistance=pack.resistance, pack_max_current=pack.max_current, dt=dt)
    return np.array([p[k] for k in PARAMS], dtype=float)


def _clamp(x, lo, hi):
    return ca.fmin(ca.fmax(x, lo), hi)


def _motor_max_torque(motor: MotorMap):
    """max torque(rpm, V): the motor table as a bilinear CasADi interpolant."""
    tbl = np.array(motor.table)
    rpms = np.arange(tbl.shape[0]) * motor.rpm_step
    volts = motor.v_min + np.arange(tbl.shape[1]) * motor.v_step
    interp = ca.interpolant("motor_map", "linear", [rpms, volts], tbl.ravel(order="F"))

    def max_torque(rpm, volts_):
        t = interp(ca.vertcat(_clamp(rpm, rpms[0], rpms[-1]), _clamp(volts_, volts[0], volts[-1])))
        return ca.if_else(rpm >= motor.max_rpm, 0, t)
    return max_torque


def _pack_solve(p, power):
    """accel.powertrain.Pack.solve: P = V I with V = Voc - I R, current limited."""
    v_oc, r, i_max = p["pack_v_oc"], p["pack_resistance"], p["pack_max_current"]
    disc = v_oc * v_oc - 4 * r * power
    current = ca.if_else(disc >= 0, (v_oc - ca.sqrt(ca.fmax(disc, 0))) / (2 * r), v_oc / (2 * r))
    current = ca.fmin(current, i_max)
    current = ca.if_else(power <= 0, 0, current)
    v = v_oc - current * r
    return v, current, v * current


def _powertrain(p, max_torque, t_cmd, omega_m, bus_v):
    """accel.sim._powertrain_torque: motor map, 80 kW rule, pack limits."""
    spinning = omega_m > 1e-6
    w = ca.fmax(omega_m, 1e-6)
    torque = ca.fmin(t_cmd, max_torque(omega_m * RPM_PER_RADS, bus_v))
    torque = ca.if_else(spinning, ca.fmin(torque, p["power_limit"] * p["motor_eff"] / w), torque)
    p_req = torque * omega_m / p["motor_eff"]
    v, current, p_del = _pack_solve(p, p_req)
    torque = ca.if_else(p_del < p_req - 1e-9, p_del * p["motor_eff"] / w, torque)
    return ca.fmax(torque, 0), v, current


def step_function(motor: MotorMap, tire) -> ca.Function:
    """step(state, torque_cmd, params) -> (next state, outputs), as an SX Function."""
    s = ca.SX.sym("state", len(STATES))
    t_cmd = ca.SX.sym("torque_cmd")
    pv = ca.SX.sym("params", len(PARAMS))
    x, v, omega, kappa, a_filt, t_lagged, bus_v = (s[i] for i in range(len(STATES)))
    p = {k: pv[i] for i, k in enumerate(PARAMS)}
    dt, r, gr = p["dt"], p["tire_radius"], p["final_drive"]
    m = p["mass"]
    max_torque = _motor_max_torque(motor)

    # Car.rear_load / total_load / drag
    q = 0.5 * p["rho"] * v * v
    cg_to_front = (1 - p["front_weight"]) * p["wheelbase"]
    fz_r = (m * G * cg_to_front / p["wheelbase"] + m * a_filt * p["cg_height"] / p["wheelbase"]
            + q * p["cla"] * (1 - p["aero_front"]))
    fz_tot = m * G + q * p["cla"]
    drag = q * p["cda"]
    fx = 2.0 * tire.symbolic_fx(kappa, 0.5 * fz_r, p["tire_scale"])

    # inverter torque response
    t_lagged = ca.if_else(p["torque_tau"] > 0,
                          t_lagged + dt * (t_cmd - t_lagged) / ca.fmax(p["torque_tau"], 1e-12),
                          t_cmd)
    omega_m = omega * gr
    torque, bus_v, current = _powertrain(p, max_torque, t_lagged, omega_m, bus_v)

    j_rear = (2 * p["wheel_inertia"] + p["rear_driveline_inertia"]
              + p["rotor_inertia"] * gr * gr)
    m_eff = m + 2 * p["wheel_inertia"] / (r * r)
    rolling = ca.if_else(ca.logic_or(v > 0, fx > 0), p["crr"] * fz_tot, 0)
    accel = (fx - drag - rolling) / m_eff
    omega_dot = (torque * gr * p["driveline_eff"] - fx * r) / j_rear

    # semi-implicit Euler
    v1 = ca.fmax(v + accel * dt, 0)
    omega1 = ca.fmax(omega + omega_dot * dt, 0)
    kappa1 = kappa + dt * ((omega1 * r - v1) - ca.fabs(v1) * kappa) / p["relaxation_length"]
    x1 = x + v1 * dt
    a_filt1 = ca.if_else(p["pitch_tau"] > 0,
                         a_filt + dt * (accel - a_filt) / ca.fmax(p["pitch_tau"], 1e-12), accel)

    nxt = ca.vertcat(x1, v1, omega1, kappa1, a_filt1, t_lagged, bus_v)
    out = ca.vertcat(accel, fx, fz_r, fz_tot - fz_r, torque, current, drag, omega_m * RPM_PER_RADS)
    return ca.Function("accel_step", [s, t_cmd, pv], [nxt, out],
                       ["state", "torque_cmd", "params"], ["next", "out"])


def initial_state(pack: Pack) -> np.ndarray:
    s = np.zeros(len(STATES))
    s[STATES.index("bus_v")] = pack.v_oc
    return s


def rollout(motor: MotorMap, tire, n: int) -> ca.Function:
    """The step unrolled n times: (state0, torque_cmd[n], params) -> (states[7 x n],
    outputs[8 x n]), column k being the state after step k and that step's outputs.
    Differentiable in everything, e.g. ca.jacobian of a final state w.r.t. params."""
    return step_function(motor, tire).mapaccum("accel_rollout", n, [0], [0])


@dataclass
class _Built:
    lib: ctypes.CDLL
    fn: object
    sizes: tuple


_compiled: dict = {}


class CompiledStep:
    """step_function compiled to C and called through ctypes, for simulate()'s loop.

    Owns the state, so the loop only passes the torque command:

        st = CompiledStep(motor, tire, param_vector, initial_state(pack))
        st.state        # list of STATES values before the step
        st(torque_cmd)  # advance one step; st.out holds OUTPUTS for that step
    """

    def __init__(self, motor: MotorMap, tire, param_vector, state0):
        b = _build(motor, tire)
        n_arg, n_res, n_iw, n_w = b.sizes
        self._fn = b.fn
        self._bufs = [(ctypes.c_double * len(STATES))(*state0),
                      (ctypes.c_double * len(STATES))()]
        self._cmd = ctypes.c_double(0.0)
        self._p = (ctypes.c_double * len(PARAMS))(*param_vector)
        self._out = (ctypes.c_double * len(OUTPUTS))()
        self._arg = (ctypes.c_void_p * max(n_arg, 3))()
        self._res = (ctypes.c_void_p * max(n_res, 2))()
        self._iw = (ctypes.c_longlong * max(n_iw, 1))()
        self._w = (ctypes.c_double * max(n_w, 1))()
        self._arg[1] = ctypes.addressof(self._cmd)
        self._arg[2] = ctypes.addressof(self._p)
        self._res[1] = ctypes.addressof(self._out)
        self._cur = 0
        self._wire()

    def _wire(self):
        self._arg[0] = ctypes.addressof(self._bufs[self._cur])
        self._res[0] = ctypes.addressof(self._bufs[1 - self._cur])

    @property
    def state(self) -> list:
        return self._bufs[self._cur][:]

    @property
    def out(self) -> list:
        return self._out[:]

    def __call__(self, torque_cmd: float):
        self._cmd.value = torque_cmd
        if self._fn(self._arg, self._res, self._iw, self._w, 0):
            raise RuntimeError("accel_step failed")
        self._cur = 1 - self._cur
        self._wire()


def _build(motor: MotorMap, tire) -> _Built:
    """Generate C for the step, compile it and load it. Cached in memory per (motor,
    tire shape) and on disk by everything the generated code depends on."""
    model_key = (motor.name, motor.rpm_step, motor.v_min, motor.v_step, tire.table_key())
    if model_key in _compiled:
        return _compiled[model_key]
    h = hashlib.sha1(repr(model_key).encode())
    h.update(np.array(motor.table).tobytes())
    h.update(ca.__version__.encode())
    for src in (Path(__file__), Path(__file__).with_name("tire.py")):
        h.update(src.read_bytes())
    key = h.hexdigest()[:16]
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    so = BUILD_DIR / f"accel_step_{key}.so"
    if not so.exists():
        cg = ca.CodeGenerator("accel_step.c", {"casadi_int": "long long int"})
        cg.add(step_function(motor, tire))
        c = BUILD_DIR / f"accel_step_{key}.c"
        c.write_text(cg.dump())
        tmp = so.with_suffix(f".{np.random.randint(1 << 30)}.tmp")
        subprocess.run(["cc", "-O2", "-fPIC", "-shared", "-o", str(tmp), str(c), "-lm"],
                       check=True)
        tmp.replace(so)   # atomic, for parallel test workers
    lib = ctypes.CDLL(str(so))
    fn = lib.accel_step
    fn.restype = ctypes.c_int
    fn.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int]
    work = lib.accel_step_work
    work.restype = ctypes.c_int
    sz = [ctypes.c_longlong() for _ in range(4)]
    work(*[ctypes.byref(z) for z in sz])
    built = _Built(lib, fn, tuple(z.value for z in sz))
    _compiled[model_key] = built
    return built


def accel_time_sensitivity(car: Car, motor: MotorMap, pack: Pack, tire, *,
                           relaxation_length: float = 0.15, ramp_time: float = 0.1,
                           dt: float = 2e-4, t_max: float = 6.0):
    """Accel time of an open-loop launch (driver ramp, no TC) and its exact gradient
    with respect to every parameter, from one reverse pass through the rollout.

    Returns (accel_time, {param: d accel_time / d param}).
    """
    from .sim import LENGTH, STAGE

    n = int(t_max / dt)
    p0 = params(car, motor, pack, tire, relaxation_length, dt)
    t = np.arange(n) * dt
    cmd = motor.peak_torque * (np.minimum(t / ramp_time, 1.0) if ramp_time > 0 else np.ones(n))
    pv = ca.MX.sym("p", len(PARAMS))
    states, _ = rollout(motor, tire, n)(initial_state(pack), cmd, pv)
    xs = states[0, :]
    xv = np.array(ca.Function("x", [pv], [xs])(p0)).ravel()

    def crossing(target):
        # column k is the state after step k, i.e. at time (k + 1) dt; the crossing
        # index is fixed at its nominal value, the interpolation within it is not
        i = int(np.argmax(xv >= target))
        if xv[i] < target or i == 0:
            raise ValueError(f"run doesn't reach x = {target} m in {t_max} s")
        return (i + 1) * dt - (xs[i] - target) / (xs[i] - xs[i - 1]) * dt

    accel_time = crossing(STAGE + LENGTH) - crossing(STAGE)
    value, grad = ca.Function("g", [pv], [accel_time, ca.gradient(accel_time, pv)])(p0)
    return float(value), {k: float(g) for k, g in zip(PARAMS, np.array(grad).ravel())}
