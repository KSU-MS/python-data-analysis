"""Fit the plant's uncertain coefficients to a VCU log (kmslap.vcu.firmware.read_log).

Equation-error identification: each physics equation is fit directly against the
measured signals, rather than re-simulating the whole run from logged torque. (A
car held at the grip limit is a knife edge open loop: re-simulating diverges into
wheelspin from tiny torque sampling errors, even with the true coefficients.)

  1. tire radius   front tooth count vs VectorNav distance (fronts roll free)
  2. mass, rear J  T*gr*eta/r - drag - rolling = m_eff*a + J*domega/dt/r, linear
                   least squares; m and J only separate when the wheels spin up
                   faster than the car (wheelspin), and the std says so
  3. grip, sigma   tire force from (2) vs the relaxation-length slip computed from
                   measured wheel and ground speed

The stds give narrowed ranges for kmslap.vcu.uncertainty, so each test day tightens
the Monte Carlo.

    inputs, _ = read_log("run.vcu")
    fit = fit_plant(inputs, Car())
    fit.params, fit.std, fit.ranges(RANGES)
"""
import dataclasses

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from kmslap.car.params import Car
from kmslap.car.powertrain import MotorMap
from kmslap.car.tire import PacejkaTire

RPM_TO_RADS = 2 * np.pi / 60

# Fit error measured on 30 random plants (kmslap.vcu.uncertainty.sample_plants, every
# coefficient off) with the VCU slip_pi controller. The regression stds alone are
# 10-70x too small: they can't see errors from coefficients held at nominal (drag,
# rolling, driveline efficiency, which is indistinguishable from mass here). Rear
# inertia is only identifiable from wheelspin, so expect no information on it.
ERROR_FLOOR = {"grip": 0.02, "mass": 0.03, "tire_radius": 0.0005, "rear_inertia": 0.9,
               "relaxation_length": 0.04}


@dataclasses.dataclass
class Fit:
    params: dict          # grip, mass, tire_radius, rear_inertia (multipliers on nominal),
                          # relaxation_length (m)
    std: dict             # max(regression std, ERROR_FLOOR)
    std_regression: dict  # what the residuals alone say
    signals: pd.DataFrame  # the 1 kHz signals the fit used, with model predictions

    def ranges(self, prior: dict, sigmas: float = 3.0) -> dict:
        """prior (like uncertainty.RANGES) narrowed to fit +- sigmas * std, for
        the parameters this fit covers. Never widens the prior."""
        out = dict(prior)
        for name, value in self.params.items():
            if name not in prior:
                continue
            lo, hi = prior[name]
            half = sigmas * self.std[name]
            out[name] = (max(lo, value - half), min(hi, value + half))
        return out

    def apply(self, car: Car) -> tuple[Car, PacejkaTire, float]:
        """The identified car, tire and relaxation length."""
        p = self.params
        car = dataclasses.replace(
            car, mass=car.mass * p["mass"], tire_radius=car.tire_radius * p["tire_radius"],
            wheel_inertia=car.wheel_inertia * p["rear_inertia"],
            rear_driveline_inertia=car.rear_driveline_inertia * p["rear_inertia"])
        return car, PacejkaTire(scale=0.6 * p["grip"]), p["relaxation_length"]


def _signals(inputs: pd.DataFrame, can_latency: float) -> pd.DataFrame:
    """Resample the log's CAN frames onto its 1 kHz tick (latency-corrected),
    with VectorNav accel bias (seen while staged) removed."""
    t_us = inputs["now_us"].to_numpy().astype(np.int64)   # uint32 in the log
    launch = t_us[np.argmax(inputs["driver_torque_nm"].to_numpy() > 0)] - 1000
    t = (t_us - launch) * 1e-6

    def frames(flag, col):
        m = inputs[flag].to_numpy() == 1
        return t[m] - can_latency, inputs[col].to_numpy()[m]

    vn_t, vn_v = frames("vn_new", "vn_vel_x")
    _, vn_a = frames("vn_new", "vn_accel_x")
    staged = (vn_t < -0.05) & (vn_t > vn_t[0] + 0.1)
    bias = float(np.mean(vn_a[staged])) if staged.any() else 0.0
    sig = pd.DataFrame({"t": t})
    sig["torque"] = np.interp(t, *frames("motor_torque_new", "motor_torque_fb"))
    sig["motor_rpm"] = np.interp(t, *frames("motor_rpm_new", "motor_rpm"))
    sig["v"] = np.interp(t, vn_t, vn_v)
    sig["a"] = np.interp(t, vn_t, vn_a) - bias
    edges = inputs["ws_edges_fl"].to_numpy().astype(float) + inputs["ws_edges_fr"].to_numpy()
    sig["front_teeth"] = np.cumsum(edges) / 2
    return sig


def fit_plant(inputs: pd.DataFrame, car: Car, motor="228HV", ws_teeth: int = 19,
              can_latency: float = 0.002, smooth_s: float = 0.03) -> Fit:
    """inputs: INPUT_COLUMNS frame of one launch (read_log / input_frame), including
    some staged time before it (for the VectorNav bias)."""
    motor = MotorMap(motor) if isinstance(motor, str) else motor
    sig = _signals(inputs, can_latency)
    run = sig[sig["t"] > 0].reset_index(drop=True)
    dt = float(np.median(np.diff(run["t"])))
    win = max(int(smooth_s / dt), 1)
    smooth = lambda x: pd.Series(x).rolling(win, center=True, min_periods=1).mean().to_numpy()
    tt = run["t"].to_numpy()
    v, a, torque = run["v"].to_numpy(), smooth(run["a"].to_numpy()), smooth(run["torque"].to_numpy())
    gr, eta = car.final_drive, car.driveline_eff
    omega = run["motor_rpm"].to_numpy() * RPM_TO_RADS / gr
    omega_dot = np.gradient(smooth(omega), tt)

    # 1. radius: distance from VectorNav vs front wheel revolutions (once rolling)
    x = np.concatenate([[0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(tt))])
    revs = (run["front_teeth"].to_numpy() - run["front_teeth"].iloc[0]) / ws_teeth
    rolling = v > 3.0
    A = 2 * np.pi * revs[rolling]
    r = float(np.dot(A, x[rolling]) / np.dot(A, A))
    r_std = float(np.std(x[rolling] - r * A) / np.sqrt(np.dot(A, A)))

    # 2. mass and rear inertia: T*gr*eta/r - drag - rolling = m_eff*a + J_rear*omega_dot/r
    jf = 2 * car.wheel_inertia / r ** 2           # front wheels spin up with the car
    resist = car.drag(v) + car.crr * car.total_load(v)
    y = torque * gr * eta / r - resist
    X = np.column_stack([a, omega_dot / r])
    use = (v > 0.3) & (torque > 5)
    coef, *_ = np.linalg.lstsq(X[use], y[use], rcond=None)
    resid = y[use] - X[use] @ coef
    cov = np.linalg.inv(X[use].T @ X[use]) * np.var(resid)
    m_eff, j_rear = coef
    m_std, j_std = np.sqrt(np.diag(cov))
    j_rot = motor.rotor_inertia * gr ** 2         # rotor inertia from the datasheet, not fit
    j_nom = 2 * car.wheel_inertia + car.rear_driveline_inertia + j_rot
    mass = m_eff - jf

    # 3. grip and relaxation length: tire force vs transient slip
    fx_meas = m_eff * a + resist
    fz_r = np.array([car.rear_load(ai, vi) for ai, vi in zip(a, v)]) * mass / car.mass
    slip_speed = omega * r - v

    def kappa(sigma):
        k = np.zeros_like(tt)
        for i in range(1, len(tt)):
            h = tt[i] - tt[i - 1]
            k[i] = k[i - 1] + h * (slip_speed[i] - abs(v[i]) * k[i - 1]) / sigma
        return k

    shape = PacejkaTire(scale=0.6)
    fit_mask = use & (v > 1.0)

    def predict(p):
        grip, sigma = p
        k = kappa(sigma)
        return np.array([2 * grip * shape.fx(ki, 0.5 * fz) for ki, fz in zip(k, fz_r)])

    res = least_squares(lambda p: (predict(p) - fx_meas)[fit_mask] / 100.0, [1.0, 0.15],
                        bounds=([0.5, 0.03], [1.6, 0.5]), diff_step=1e-3)
    s2 = 2 * res.cost / max(fit_mask.sum() - 2, 1)
    tire_std = np.sqrt(np.diag(np.linalg.inv(res.jac.T @ res.jac) * s2))

    run["fx_meas"] = fx_meas
    run["fx_model"] = predict(res.x)
    run["kappa"] = kappa(res.x[1])
    params = {
        "grip": float(res.x[0]),
        "mass": float(mass / car.mass),
        "tire_radius": float(r / car.tire_radius),
        # the multiplier applies to the wheel + driveline part, not the rotor
        "rear_inertia": float((j_rear - j_rot) / (j_nom - j_rot)),
        "relaxation_length": float(res.x[1]),
    }
    std_regression = {
        "grip": float(tire_std[0]),
        "mass": float(m_std / car.mass),
        "tire_radius": float(r_std / car.tire_radius),
        "rear_inertia": float(j_std / (j_nom - j_rot)),
        "relaxation_length": float(tire_std[1]),
    }
    std = {k: max(v, ERROR_FLOOR[k]) for k, v in std_regression.items()}
    return Fit(params=params, std=std, std_regression=std_regression, signals=run)
