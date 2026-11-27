"""The car's grip envelope (g-g-v diagram) from the double-track model, by
optimization, and speed targets for a path from it.

At each speed V and direction theta in the g-g plane (0 = pure acceleration,
pi/2 = pure cornering left, pi = pure braking), Ipopt finds the steady-state way of
driving the car that maximizes the acceleration rho along that direction:

    variables    body slip, steer, the four wheel slip ratios, motor torque, brake
    subject to   sum of tire forces = m (ax, ay) - drag - rolling, yaw moment = 0,
                 each wheel's torque balance (front: brake; rear: diff carrier torque,
                 LSD clutch torque, brake), motor map and 80 kW limit, steering
                 lock, slip angles within the tire data (|alpha| <= 0.2 rad: the TTC
                 fit's lateral force keeps rising past its data, so the solver would
                 otherwise find drifts), body slip within a drivable +-8 deg, and
                 stable: the lateral / yaw dynamics linearized at the solution (steer
                 and wheel slip held) have negative trace and positive determinant,
                 i.e. the car holds the corner without the driver catching a slide
    with         (ax, ay) = rho (cos theta, sin theta), yaw rate = ay / V, loads from
                 ax, ay and aero (vehicle.py's load transfer, settled)

Same tires, load transfer, aero, diff and motor as vehicle.py; rotational inertias
are neglected (quasi-steady state). Tables are B-splines here so the solver gets
smooth derivatives.

    env = gg_envelope(Car())
    env.ay_max(15.0)              # m/s^2 of steady cornering at 15 m/s
    v = speed_profile(path, env)  # target speed along a path, for PathFollower
"""
import dataclasses
import hashlib
import math
from dataclasses import dataclass
from pathlib import Path as FilePath

import casadi as ca
import numpy as np

from kmslap.driving.paths import Path
from kmslap.car.params import G, Car
from kmslap.car.powertrain import PACKS, MotorMap, Pack
from kmslap.car.tire import default_tire
from kmslap import dirs

WHEELS = ("fl", "fr", "rl", "rr")
CACHE_DIR = dirs.BUILD / "envelope"
VARIABLES = ["beta", "steer", "kappa_fl", "kappa_fr", "kappa_rl", "kappa_rr", "torque", "brake", "rho"]
MAX_SLIP_ANGLE = 0.2     # rad, about the TTC data's range
MAX_BETA = math.radians(8)
# starting guesses per regime (scaled variables): accelerating, cornering, braking
_STARTS = [np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 1.5, 0.0, 6.0]),
           np.array([0.02, 0.05, 0.0, 0.0, 0.02, 0.02, 0.3, 0.0, 10.0]),
           np.array([0.0, 0.0, -0.08, -0.08, -0.05, -0.05, 0.0, 0.8, 12.0])]


@dataclass
class Envelope:
    speeds: np.ndarray          # m/s
    angles: np.ndarray          # rad, 0 accel .. pi/2 cornering .. pi braking
    rho: np.ndarray             # [speed, angle] max acceleration along each angle, m/s^2
    solution: np.ndarray        # [speed, angle, VARIABLES] the car state achieving it
    converged: np.ndarray       # [speed, angle] bool

    @property
    def ax(self):
        return self.rho * np.cos(self.angles)

    @property
    def ay(self):
        return self.rho * np.sin(self.angles)

    def _at(self, v):
        """(ax, ay) boundary at speed v, interpolated between the solved speeds."""
        v = min(max(v, self.speeds[0]), self.speeds[-1])
        i = min(int(np.searchsorted(self.speeds, v)), len(self.speeds) - 1)
        i0 = max(i - 1, 0)
        f = 0.0 if i == i0 else (v - self.speeds[i0]) / (self.speeds[i] - self.speeds[i0])
        return ((1 - f) * self.ax[i0] + f * self.ax[i]), ((1 - f) * self.ay[i0] + f * self.ay[i])

    def ay_max(self, v: float) -> float:
        return float(self._at(v)[1].max())

    def ax_limits(self, v: float, ay: float) -> tuple[float, float]:
        """(max acceleration, max braking as a negative number) at speed v while
        cornering at |ay|; (0, 0) beyond the envelope."""
        ax, ay_b = self._at(v)
        ay = abs(ay)
        k = int(np.argmax(ay_b))
        if ay >= ay_b[k]:
            return 0.0, 0.0
        # accel branch: angles 0..k, ay rising; braking branch: k..end, ay falling
        acc = np.interp(ay, np.maximum.accumulate(ay_b[:k + 1]), ax[:k + 1])
        brk = np.interp(ay, np.maximum.accumulate(ay_b[k:][::-1]), ax[k:][::-1])
        return float(acc), float(brk)


def _problem(car: Car, motor: MotorMap, pack: Pack, tire, bus_v: float, max_steer: float):
    x = ca.MX.sym("x", len(VARIABLES))
    pr = ca.MX.sym("p", 3)                       # V, cos theta, sin theta
    beta, steer, *kap, t_s, b_s, rho = (x[i] for i in range(len(VARIABLES)))
    torque, brake = 100 * t_s, 1000 * b_s        # scaled for the solver
    V, c_th, s_th = pr[0], pr[1], pr[2]
    m, r_t, gr = car.mass, car.tire_radius, car.final_drive
    ax, ay = rho * c_th, rho * s_th
    # lateral velocity and yaw rate as their own symbols, for the stability
    # Jacobian; substituted by their steady values below
    vy, yaw = ca.MX.sym("vy"), ca.MX.sym("yaw")
    vx = V

    lf = (1 - car.front_weight) * car.wheelbase
    lr = car.front_weight * car.wheelbase
    pos = {"fl": (lf, car.track_front / 2), "fr": (lf, -car.track_front / 2),
           "rl": (-lr, car.track_rear / 2), "rr": (-lr, -car.track_rear / 2)}
    delta = {"fl": steer, "fr": steer, "rl": 0, "rr": 0}

    # loads: vehicle.py's, with the transfer filters settled
    q = 0.5 * car.rho * vx * vx
    down = q * car.cla
    long_tr = m * ax * car.cg_height / car.wheelbase
    lat = m * ay * car.cg_height
    lat_f = lat * car.roll_stiffness_front / car.track_front
    lat_r = lat * (1 - car.roll_stiffness_front) / car.track_rear
    front = m * G * lr / car.wheelbase - long_tr + down * car.aero_front
    rear = m * G * lf / car.wheelbase + long_tr + down * (1 - car.aero_front)
    fz = {"fl": front / 2 - lat_f, "fr": front / 2 + lat_f, "rl": rear / 2 - lat_r, "rr": rear / 2 + lat_r}

    fx, fy, fxb, fyb, alpha, omega = {}, {}, {}, {}, {}, {}
    for w, k in zip(WHEELS, kap):
        px, py = pos[w]
        cvx, cvy = vx - yaw * py, vy + yaw * px
        c, sn = ca.cos(delta[w]), ca.sin(delta[w])
        u, wl = c * cvx + sn * cvy, -sn * cvx + c * cvy
        alpha[w] = -wl / u                               # the slip states' steady values
        omega[w] = (1 + k) * u / r_t
        fx[w], fy[w] = tire.symbolic_fxy(k, alpha[w], fz[w], tire.scale, smooth=True)
        fxb[w] = c * fx[w] - sn * fy[w]
        fyb[w] = sn * fx[w] + c * fy[w]

    drag = q * car.cda
    rolling = car.crr * (m * G + down)
    t_c = torque * gr * car.driveline_eff
    lock = car.diff_preload + car.diff_lock_drive * t_c
    t_diff = 0.5 * lock * ca.tanh((omega["rl"] - omega["rr"]) / car.diff_slip_eps)
    bf, br = 0.5 * brake * car.brake_bias_front, 0.5 * brake * (1 - car.brake_bias_front)
    omega_m = 0.5 * (omega["rl"] + omega["rr"]) * gr

    eq = [sum(fxb.values()) - drag - rolling - m * ax,
          sum(fyb.values()) - m * ay,
          sum(pos[w][0] * fyb[w] - pos[w][1] * fxb[w] for w in WHEELS),
          fx["fl"] * r_t + bf, fx["fr"] * r_t + bf,
          fx["rl"] * r_t - (0.5 * t_c - t_diff - br),
          fx["rr"] * r_t - (0.5 * t_c + t_diff - br)]
    # motor: map at the loaded bus voltage, smooth; 80 kW at the pack
    rpm_grid = np.arange(0.0, motor.max_rpm, 100.0)
    t_max = ca.interpolant("tmax", "bspline", [rpm_grid],
                           [motor.max_torque(r, bus_v) for r in rpm_grid])
    ineq = [t_max(omega_m * 60 / (2 * math.pi)) - torque,
            car.power_limit * car.motor_eff - torque * omega_m]
    ineq += [alpha[w] for w in WHEELS]
    ineq += [fz[w] for w in WHEELS]
    lat_yaw = ca.vertcat(sum(fyb.values()) / m - yaw * vx,
                         sum(pos[w][0] * fyb[w] - pos[w][1] * fxb[w] for w in WHEELS) / car.yaw_inertia)
    jac = ca.jacobian(lat_yaw, ca.vertcat(vy, yaw))
    ineq += [-(jac[0, 0] + jac[1, 1]), jac[0, 0] * jac[1, 1] - jac[0, 1] * jac[1, 0]]
    g = ca.substitute(ca.vertcat(*eq, *ineq), ca.vertcat(vy, yaw), ca.vertcat(V * beta, ay / V))
    n_eq = len(eq)
    lbg = [0.0] * n_eq + [0.0, 0.0] + [-MAX_SLIP_ANGLE] * 4 + [50.0] * 4 + [0.0, 0.0]
    ubg = [0.0] * n_eq + [ca.inf, ca.inf] + [MAX_SLIP_ANGLE] * 4 + [ca.inf] * 4 + [ca.inf, ca.inf]
    lbx = [-MAX_BETA, -max_steer, -0.25, -0.25, -0.25, -0.25, 0.0, 0.0, 0.0]
    ubx = [MAX_BETA, max_steer, 0.0, 0.0, 0.95, 0.95, motor.peak_torque / 100,
           car.max_brake_torque / 1000, 4 * G]
    solver = ca.nlpsol("gg", "ipopt", {"x": x, "p": pr, "f": -rho, "g": g},
                       {"ipopt.print_level": 0, "print_time": False, "ipopt.max_iter": 300,
                        "ipopt.tol": 1e-8})
    return solver, dict(lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)


def gg_envelope(car: Car = None, motor: MotorMap | str = "228HV", pack: Pack | str = "cosmx600v",
                tire=None, *, speeds=np.arange(4.0, 34.01, 3.0), n_angles: int = 25,
                soc: float = 1.0, max_steering_wheel: float = math.radians(120),
                cache: bool = True) -> Envelope:
    """Solve the envelope over speeds x angles (0..pi; the car is symmetric).
    ~2 minutes; cached in build/envelope/ by everything it depends on."""
    car = car or Car()
    motor = MotorMap(motor) if isinstance(motor, str) else motor
    pack = Pack(PACKS[pack]) if isinstance(pack, str) else pack
    tire = tire or default_tire()
    h = hashlib.sha1(repr((dataclasses.astuple(car), motor.name, pack.name, vars(pack).get("v_oc"),
                           tire.table_key_2d(), tire.scale, list(speeds), n_angles, soc,
                           max_steering_wheel)).encode())
    h.update(np.array(motor.table).tobytes())
    for src in (FilePath(__file__), dirs.PACKAGE / "car" / "tire.py"):
        h.update(src.read_bytes())
    path = CACHE_DIR / f"{h.hexdigest()[:16]}.npz"
    if cache and path.exists():
        z = np.load(path)
        return Envelope(z["speeds"], z["angles"], z["rho"], z["solution"], z["converged"])
    v_oc = pack.series * float(np.interp(soc, pack.ocv_soc, pack.ocv_cell))
    i_80kw = car.power_limit / v_oc
    bus_v = v_oc - i_80kw * pack.resistance
    solver, bounds = _problem(car, motor, pack, tire, bus_v, max_steering_wheel / car.steer_ratio)

    speeds = np.asarray(speeds, float)
    angles = np.linspace(0.0, math.pi, n_angles)
    rho = np.full((len(speeds), n_angles), np.nan)
    sol = np.full((len(speeds), n_angles, len(VARIABLES)), np.nan)
    ok = np.zeros((len(speeds), n_angles), bool)
    for i, v in enumerate(speeds):
        guess = None
        for j, th in enumerate(angles):
            # the problem isn't convex: start from the neighbouring angle's solution
            # and from each regime's guess, keep the best that converged
            for start in ([guess] if guess is not None else []) + _STARTS:
                res = solver(x0=start, p=[v, math.cos(th), math.sin(th)], **bounds)
                xs = np.array(res["x"]).ravel()
                if solver.stats()["success"] and not (ok[i, j] and xs[-1] <= rho[i, j]):
                    rho[i, j], sol[i, j], ok[i, j] = xs[-1], xs, True
            guess = sol[i, j] if ok[i, j] else None
    sol[..., 6] *= 100     # back to Nm
    sol[..., 7] *= 1000
    env = Envelope(speeds, angles, rho, sol, ok)
    if cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(path, **dataclasses.asdict(env))
    return env


def speed_profile(path: Path, env: Envelope, margin: float = 0.9, v_start: float | None = None,
                  v_max: float = 40.0) -> np.ndarray:
    """Target speed along the path within margin * the car's envelope: as fast as
    cornering allows at each point, then forward (traction / power) and backward
    (braking) passes using the acceleration left over at each point's lateral g.

    margin: the envelope is the steady-state limit; in the time-domain model the
    PathFollower holds ~90% of it in steady cornering (the transient wheel spin /
    diff dynamics the envelope's stability check leaves out take the rest)."""
    k = np.abs(path.k)
    vs = np.linspace(env.speeds[0], env.speeds[-1], 200)
    ay_cap = np.array([env.ay_max(v) for v in vs]) * margin
    # highest speed whose cornering requirement v^2 k fits the envelope at that speed
    v = np.empty(len(k))
    for n, kk in enumerate(k):
        fits = vs[vs * vs * kk <= ay_cap]
        v[n] = min(v_max, fits.max() if len(fits) else vs[0])
    if v_start is not None:
        v[0] = min(v[0], v_start)
    ds = np.diff(path.s)
    for n in range(1, len(v)):
        acc, _ = env.ax_limits(v[n - 1], v[n - 1] ** 2 * k[n - 1] / margin)
        v[n] = min(v[n], math.sqrt(v[n - 1] ** 2 + 2 * margin * max(acc, 0.0) * ds[n - 1]))
    for n in range(len(v) - 2, -1, -1):
        _, brk = env.ax_limits(v[n + 1], v[n + 1] ** 2 * k[n + 1] / margin)
        v[n] = min(v[n], math.sqrt(v[n + 1] ** 2 + 2 * margin * max(-brk, 0.0) * ds[n]))
    return v
