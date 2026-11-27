"""Minimum lap time with the double-track car: racing line, speed, steering and wheel
slips solved together, with yaw and body slip as dynamic states.

models/mintime.py keeps the car inside its quasi-steady g-g-v envelope; that leaves
out what it takes to change direction (yaw inertia, the body slip building up), so a
line can plan transients the car can't drive. Here the car is the envelope's own
double-track model (models/envelope.py: combined-slip Pacejka tires, load transfer,
aero, Drexler LSD torque balance, motor map, 80 kW) at every point of the lap, with
the yaw rate and body slip integrated instead of held steady: the formulation of
Christ et al. (TUM, "Time-optimal trajectory planning for a race car considering
variable tire-road friction coefficients"; their global_racetrajectory_optimization),
on our car.

Curvilinear coordinates on the track centerline, arc length s independent:
    states    v speed, beta body slip, r yaw rate, n lateral offset, xi heading
              relative to the centerline
    controls  steer (road wheel), the four wheel slip ratios, motor torque, brake,
              each hub motor's torque (car.hub_front / hub_rear: one per driven wheel,
              so the plan torque-vectors); ax, ay (body-frame accelerations, for the
              load transfer) are algebraic, pinned by the force balance
    dt/ds     = (1 - n k) / (v cos(xi + beta))
    dn/ds     = dt/ds v sin(xi + beta)
    dxi/ds    = dt/ds r - k
    dv/ds     = dt/ds (Fx cos beta + Fy sin beta - drag - rolling) / m
    dbeta/ds  = dt/ds ((-Fx sin beta + Fy cos beta) / (m v) - r)
    dr/ds     = dt/ds Mz / Iz
with (Fx, Fy, Mz) the tires' summed body-frame forces and yaw moment, each wheel's
torque balanced (quasi-steady wheel spin, as in the envelope). Minimize the lap time
plus small penalties on control rates; trapezoidal collocation, Ipopt, started from
the quasi-steady solution (mintime.solve) with the envelope's steady-state controls.

The solution's steer, body slip and yaw rate along the line are feedforward for the
PathFollower (steer_ff / beta_ff / yaw_ff).
"""
import math
from dataclasses import dataclass

import casadi as ca
import numpy as np

from kmslap.car.params import G, Car
from kmslap.car.powertrain import PACKS, MotorMap, Pack
from kmslap.car.tire import default_tire
from kmslap.driving.paths import Path
from kmslap.events import track as track_mod
from kmslap.models import envelope as env_mod
from kmslap.models import mintime as qss

WHEELS = env_mod.WHEELS
STATES = ["v", "beta", "r", "n", "xi"]
CONTROLS = ["steer", "kappa_fl", "kappa_fr", "kappa_rl", "kappa_rr", "torque", "brake", "ax", "ay"]
# variable scales (the solver works in x / scale)
X_SCALE = np.array([20.0, 0.1, 1.0, 1.0, 0.2])
U_SCALE = np.array([0.1, 0.05, 0.05, 0.05, 0.05, 100.0, 1000.0, 10.0, 10.0])
HUB_SCALE = 10.0                                         # N m at a hub motor


def hub_wheels(car: Car) -> list[str]:
    """The wheels with a hub motor."""
    return [w for w in WHEELS if (car.hub_front if w[0] == "f" else car.hub_rear)]


def controls(car: Car) -> list[str]:
    """CONTROLS, then a torque per hub motor."""
    return CONTROLS + [f"hub_{w}" for w in hub_wheels(car)]


@dataclass
class MinTimeDT:
    line: Path                # the optimal line (driving resolution)
    v: np.ndarray             # speed along `line`
    steer_ff: np.ndarray      # road wheel angle along `line`, rad
    beta_ff: np.ndarray       # body slip along `line`, rad
    yaw_ff: np.ndarray        # yaw rate along `line`, rad/s
    lap_time: float           # s, the OCP's
    s: np.ndarray             # centerline arc length of the nodes
    x: dict                   # states at the nodes
    u: dict                   # controls at the nodes
    stats: dict


def _by_sign(t, pos, neg, eps):
    """pos where t > 0, neg where t < 0, smoothly across +-eps (a torque's driving and
    generating efficiencies)."""
    return 0.5 * (pos + neg) + 0.5 * (pos - neg) * ca.tanh(t / eps)


def _model(car: Car, motor: MotorMap, tire, bus_v: float, grip: float, p_charge: float | None = None) -> ca.Function:
    """(x, u, k) -> (state rates d/ds, dt/ds, equalities, inequalities, the motors' net
    torque at the wheels), physical units.
    p_charge: W the pack takes in (regen on: the motors' torque may be negative, within
    their torque curves, the net power down to -p_charge); None: motoring only."""
    regen = p_charge is not None
    hubs = hub_wheels(car)
    x, u, k = ca.MX.sym("x", len(STATES)), ca.MX.sym("u", len(CONTROLS) + len(hubs)), ca.MX.sym("k")
    v, beta, r, n, xi = (x[i] for i in range(len(STATES)))
    steer, *kap, torque, brake, ax, ay = (u[i] for i in range(len(CONTROLS)))
    hub_t = {w: u[len(CONTROLS) + i] for i, w in enumerate(hubs)}
    m, r_t, gr = car.mass, car.tire_radius, car.final_drive

    lf = (1 - car.front_weight) * car.wheelbase
    lr = car.front_weight * car.wheelbase
    pos = {"fl": (lf, car.track_front / 2), "fr": (lf, -car.track_front / 2),
           "rl": (-lr, car.track_rear / 2), "rr": (-lr, -car.track_rear / 2)}
    delta = {"fl": steer, "fr": steer, "rl": 0, "rr": 0}

    # loads: the envelope's (vehicle.py's, transfer settled)
    q = 0.5 * car.rho * v * v
    down = q * car.cla
    long_tr = m * ax * car.cg_height / car.wheelbase
    lat = m * ay * car.cg_height
    lat_f = lat * car.roll_stiffness_front / car.track_front
    lat_r = lat * (1 - car.roll_stiffness_front) / car.track_rear
    front = m * G * lr / car.wheelbase - long_tr + down * car.aero_front
    rear = m * G * lf / car.wheelbase + long_tr + down * (1 - car.aero_front)
    fz = {"fl": front / 2 - lat_f, "fr": front / 2 + lat_f, "rl": rear / 2 - lat_r, "rr": rear / 2 + lat_r}

    vx, vy = v * ca.cos(beta), v * ca.sin(beta)
    fx, fxb, fyb, alpha, omega = {}, {}, {}, {}, {}
    for w, kw in zip(WHEELS, kap):
        px, py = pos[w]
        cvx, cvy = vx - r * py, vy + r * px
        c, sn = ca.cos(delta[w]), ca.sin(delta[w])
        uw, wl = c * cvx + sn * cvy, -sn * cvx + c * cvy
        alpha[w] = -wl / uw
        omega[w] = (1 + kw) * uw / r_t
        fx[w], fy = tire.symbolic_fxy(kw, alpha[w], fz[w], tire.scale * grip, smooth=True)
        fxb[w] = c * fx[w] - sn * fy
        fyb[w] = sn * fx[w] + c * fy
    Fx, Fy = sum(fxb.values()), sum(fyb.values())
    Mz = sum(pos[w][0] * fyb[w] - pos[w][1] * fxb[w] for w in WHEELS)
    resist = q * car.cda + car.crr * (m * G + down)        # along the velocity

    sf = (1 - n * k) / (v * ca.cos(xi + beta))
    rates = ca.vertcat(sf * (Fx * ca.cos(beta) + Fy * ca.sin(beta) - resist) / m,
                       sf * ((-Fx * ca.sin(beta) + Fy * ca.cos(beta)) / (m * v) - r),
                       sf * Mz / car.yaw_inertia,
                       sf * v * ca.sin(xi + beta),
                       sf * r - k)

    if regen:
        # the chain loses on the way out driving, on the way in generating; the LSD locks
        # by its coast ramp generating
        t_c = torque * gr * _by_sign(torque, car.driveline_eff, 1 / car.driveline_eff, 1.0)
        lock = (car.diff_preload + 0.5 * (car.diff_lock_drive + car.diff_lock_coast) * ca.sqrt(t_c * t_c + 1.0)
                + 0.5 * (car.diff_lock_drive - car.diff_lock_coast) * t_c)
    else:
        t_c = torque * gr * car.driveline_eff
        lock = car.diff_preload + car.diff_lock_drive * t_c
    t_diff = 0.5 * lock * ca.tanh((omega["rl"] - omega["rr"]) / car.diff_slip_eps)
    bf, br = 0.5 * brake * car.brake_bias_front, 0.5 * brake * (1 - car.brake_bias_front)
    omega_m = 0.5 * (omega["rl"] + omega["rr"]) * gr
    # hub motors through their gearboxes (generating too, with regen)
    gear = (lambda t: _by_sign(t, car.hub_gear_eff, 1 / car.hub_gear_eff, 0.1)) if regen else (lambda t: car.hub_gear_eff)
    hub_wheel = {w: hub_t[w] * car.hub_ratio * gear(hub_t[w]) if w in hub_t else 0 for w in WHEELS}
    if car.central_drive:
        rear_eq = [fx["rl"] * r_t - (0.5 * t_c - t_diff - br + hub_wheel["rl"]),
                   fx["rr"] * r_t - (0.5 * t_c + t_diff - br + hub_wheel["rr"])]
    else:                                                # no diff: each rear wheel on its own
        rear_eq = [fx[w] * r_t - (hub_wheel[w] - br) for w in ("rl", "rr")]
    eq = ca.vertcat(m * ax - (Fx - resist * ca.cos(beta)),
                    m * ay - (Fy - resist * ca.sin(beta)),
                    fx["fl"] * r_t + bf - hub_wheel["fl"], fx["fr"] * r_t + bf - hub_wheel["fr"],
                    *rear_eq)
    # the torque curve to the motor's max speed (0 there), the speed held inside it: a
    # spline extrapolates wildly, and with a short final drive the solver's steps reach it
    rpm_grid = np.arange(0.0, motor.max_rpm + 1.0, 100.0)
    t_max = ca.interpolant("tmax", "bspline", [rpm_grid], [motor.max_torque(rr, bus_v) for rr in rpm_grid])
    rpm = ca.fmin(ca.fmax(omega_m * 60 / (2 * math.pi), 0.0), rpm_grid[-1])
    p_hub, hub_lim, hub_gen = 0, [], []
    if hubs:
        from kmslap.car.drivetrain import HubMotor
        hub = HubMotor(car.hub_motor)
        rpms, torques, t_tbl, eff_tbl = hub.tables()
        hub_tmax = ca.interpolant("hub_tmax", "bspline", [rpms], t_tbl)
        hub_eff = ca.interpolant("hub_eff", "bspline", [rpms, torques], eff_tbl.ravel(order="F"))
        for w in hubs:
            w_m = omega[w] * car.hub_ratio
            n_m = ca.fmin(ca.fmax(w_m * 60 / (2 * math.pi), 0.0), rpms[-1])
            if regen:
                eta = hub_eff(ca.vertcat(n_m, ca.fmin(ca.sqrt(hub_t[w] ** 2 + 1e-4), torques[-1])))
                p_hub += hub_t[w] * w_m * _by_sign(hub_t[w], 1 / eta, eta, 0.1)
                hub_gen.append(hub_tmax(n_m) + hub_t[w])
            else:
                t_m = ca.fmin(ca.fmax(hub_t[w], 0.0), torques[-1])
                p_hub += hub_t[w] * w_m / hub_eff(ca.vertcat(n_m, t_m))
            hub_lim.append(hub_tmax(n_m) - hub_t[w])
    # >= 0: the motor's torque curve, the 80 kW (EMRAX and hub motors together), each
    # hub motor's torque curve (regen: generating too, and the pack's charge limit);
    # |.| <= max slip angle: the slip angles; >= 50 N: the loads
    if regen:
        p_emrax = torque * omega_m * _by_sign(torque, 1 / car.motor_eff, car.motor_eff * car.regen_inverter_eff, 1.0)
        pos = [t_max(rpm) - torque, car.motor_eff * (car.power_limit - p_emrax - p_hub), *hub_lim,
               t_max(rpm) + torque, car.motor_eff * (p_emrax + p_hub + p_charge), *hub_gen]
    else:
        pos = [t_max(rpm) - torque, car.power_limit * car.motor_eff - torque * omega_m - p_hub * car.motor_eff,
               *hub_lim]
    ineq = ca.vertcat(*pos, *[alpha[w] for w in WHEELS], *[fz[w] for w in WHEELS])
    net = t_c + sum(hub_wheel.values())
    return ca.Function("dt_model", [x, u, k], [rates, sf, eq, ineq, net])


def _warm_controls(env, v, ax, ay):
    """The envelope's steady-state controls nearest each (v, ax, ay): body slip,
    steer, wheel slips, torque, brake, front / rear hub torque (left turns solved;
    mirrored for right)."""
    sol, ok = env.solution, env.converged
    th = np.arctan2(np.abs(ay), ax)
    out = np.zeros((len(v), 10))
    for i in range(len(v)):
        si = int(np.argmin(np.abs(env.speeds - v[i])))
        aj = np.flatnonzero(ok[si])
        j = aj[np.argmin(np.abs(env.angles[aj] - th[i]))]
        beta, steer, kfl, kfr, krl, krr, torque, brake, hf, hr = sol[si, j][:10]
        if ay[i] < 0:
            beta, steer, kfl, kfr, krl, krr = -beta, -steer, kfr, kfl, krr, krl
        out[i] = [beta, steer, kfl, kfr, krl, krr, torque, brake, hf, hr]
    return out


def solve(center, env=None, *, car: Car = None, motor: MotorMap | str = "228HV",
          pack: Pack | str = "cosmx600v", tire=None, width: float = track_mod.TRACK_WIDTH,
          car_width: float = track_mod.CAR_WIDTH, margin: float = track_mod.EDGE_MARGIN,
          grip: float = 1.0, step: float = 2.0, max_steering_wheel: float = math.radians(120),
          max_beta: float = math.radians(10), w_steer: float = 1.0, w_drive: float = 1e-3,
          w_slip: float = 1e-2, max_iter: int = 3000, verbose: bool = False, warm=None,
          regen: bool = False, soc: float = 1.0, label: str = "",
          cones: bool = True) -> MinTimeDT:
    """center: a Track, or a centerline (then the track is `width` wide).
    grip: the tires' friction scale (a driver uses ~0.9 of it).
    w_*: penalties on the rates of steer (rad/m), torque / brake (per 100 Nm / kNm
        per m) and wheel slip ratios (per m): smooth, drivable controls.
    warm: a mintime.MinTime (quasi-steady) to start from; solved here if None.
    regen: the motors brake too (a brake pedal whose first travel is regen only, FSAE
        T.3.2.4), within their torque curves and the pack's charge current. With
        regen in the brake pedal's travel some motors can drive while others
        generate, as long as their net torque is the driver's (EV.3.1.3): so they
        torque-vector braking too. Their net torque is never driving while the
        hydraulic brakes are on (the pedal's last travel). The pack takes its charge
        peak (the cells' 2 s rating: a braking zone's worth; the car model holds it to
        that) within the cells' max voltage, which at a full pack leaves little.
    soc: the pack's state of charge the lap is driven at (its voltage).
    label: what's being planned, for the progress line.
    cones: on a coned track, keep every cone `margin` clear of the car's footprint (its
        axles + overhang ahead and behind, as track.penalties scores cones): at each node
        whose footprint spans the cone, the car's side there (its heading to the track
        swings it out) stays clear of the cone, on the side the cone marks. Exact at the cones, where the edges
        between gates are interpolated. False: the edges only."""
    car = car or Car()
    motor = MotorMap(motor) if isinstance(motor, str) else motor
    pack = Pack(PACKS[pack]) if isinstance(pack, str) else pack
    tire = tire or default_tire()
    env = env or env_mod.gg_envelope(car, motor, pack, tire)
    track = center if isinstance(center, track_mod.Track) else track_mod.constant_width(center, width)
    tr = track.resampled(step)
    ref = tr.center
    lo, hi = tr.limits(car_width, margin)
    N = len(ref.s)
    ds = ref.length / N
    k = ref.k
    max_steer = max_steering_wheel / car.steer_ratio

    v_oc = pack.series * float(np.interp(soc, pack.ocv_soc, pack.ocv_cell))
    bus_v = v_oc - car.power_limit / v_oc * pack.resistance
    i_chg = min(pack.peak_charge_current, max(pack.series * pack.v_cell_max - v_oc, 0.0) / pack.resistance)
    p_charge = i_chg * (v_oc + i_chg * pack.resistance) if regen else None
    model = _model(car, motor, tire, bus_v, grip, p_charge).map(N)

    # warm start: the quasi-steady solution, sampled at these nodes
    if warm is None:
        warm = qss.solve(track, env, car=car, car_width=car_width, margin=margin,
                         grip=grip, step=step, label=label)
    sw = np.interp(ref.s, warm.s, warm.n, period=ref.length)
    vw = np.interp(ref.s, warm.s, warm.speed, period=ref.length)
    axw = np.interp(ref.s, warm.s, warm.ax, period=ref.length)
    ayw = np.interp(ref.s, warm.s, warm.ay, period=ref.length)
    nw = np.clip(sw, lo + 0.02, hi - 0.02)
    dn = (np.roll(nw, -1) - np.roll(nw, 1)) / (2 * ds)
    xiw = np.arctan(dn / np.maximum(1 - nw * k, 0.2))
    ctl = _warm_controls(env, vw, axw, ayw)
    betaw = np.clip(ctl[:, 0], -0.9 * max_beta, 0.9 * max_beta)
    xiw = xiw - betaw                                    # velocity, not body, along the line
    x0 = np.vstack([vw, betaw, ayw / np.maximum(vw, 1.0), nw, xiw])
    hubs = hub_wheels(car)
    names = controls(car)
    u_scale = np.concatenate([U_SCALE, np.full(len(hubs), HUB_SCALE)])
    u0 = np.vstack([ctl[:, 1], *ctl[:, 2:6].T, ctl[:, 6], ctl[:, 7], axw, ayw,
                    *[ctl[:, 8 if w[0] == "f" else 9] for w in hubs]])

    opti = ca.Opti()
    Xs = opti.variable(len(STATES), N)
    Us = opti.variable(len(names), N)
    X = ca.diag(ca.DM(X_SCALE)) @ Xs
    U = ca.diag(ca.DM(u_scale)) @ Us
    nxt = lambda A: ca.horzcat(A[:, 1:], A[:, :1])         # periodic
    rates, sf, eq, ineq, net = model(X, U, ca.DM(k).T)
    opti.subject_to(nxt(X) - X == ds / 2 * (rates + nxt(rates)))
    lap_time = ds / 2 * ca.sum2(sf + nxt(sf))
    opti.subject_to(ca.vec(eq) / 100 == 0)               # N, N m -> O(1)
    n_pos = (2 + len(hubs)) * (2 if regen else 1)
    opti.subject_to(ca.vec(ineq[0:n_pos, :]) >= 0)
    opti.subject_to(opti.bounded(-env_mod.MAX_SLIP_ANGLE, ca.vec(ineq[n_pos:n_pos + 4, :]), env_mod.MAX_SLIP_ANGLE))
    opti.subject_to(ca.vec(ineq[n_pos + 4:n_pos + 8, :]) >= 50)
    v_, beta_, r_, n_, xi_ = (X[i, :] for i in range(len(STATES)))
    opti.subject_to(opti.bounded(2.0, v_, 40.0))
    opti.subject_to(opti.bounded(-max_beta, beta_, max_beta))
    opti.subject_to(opti.bounded(ca.DM(lo).T, n_, ca.DM(hi).T))
    if cones and track.cones is not None:
        # each cone against the side of the footprint at its place along the track
        lf = (1 - car.front_weight) * car.wheelbase
        lr = car.front_weight * car.wheelbase
        front, rear = lf + car.overhang_front, lr + car.overhang_rear
        nxr, nyr = track_mod.normals(ref)
        idx, ahead, n_cone, side = [], [], [], []
        for cx, cy in track.cones:
            j = int(np.argmin(np.hypot(ref.x - cx, ref.y - cy)))
            nc = (cx - ref.x[j]) * nxr[j] + (cy - ref.y[j]) * nyr[j]
            sc = ref.s[j] + (cx - ref.x[j]) * np.cos(ref.heading[j]) + (cy - ref.y[j]) * np.sin(ref.heading[j])
            d = (sc - ref.s + ref.length / 2) % ref.length - ref.length / 2      # the cone ahead of each node
            # the nodes whose footprint spans the cone, and a node beyond at each end (the
            # path between nodes is interpolated): those at the footprint's nearer end
            for i in np.flatnonzero((d <= front + ds) & (d >= -rear - ds)):
                idx.append(int(i)); n_cone.append(nc); side.append(1.0 if nc > 0 else -1.0)
                ahead.append(float(np.clip(d[i], -rear, front)))
        if idx:
            a, sgn = ca.DM(ahead).T, ca.DM(side).T
            nn, xx = n_[0, idx], xi_[0, idx]
            flank = nn + a * ca.sin(xx) + sgn * (car_width / 2) * ca.cos(xx)   # the car's side facing the cone
            clear = track_mod.CONE_RADIUS + margin
            opti.subject_to(sgn * (ca.DM(n_cone).T - flank) >= clear)
    opti.subject_to(opti.bounded(-1.0, xi_, 1.0))
    steer_, kfl, kfr, krl, krr, torque_, brake_, ax_, ay_ = (U[i, :] for i in range(len(CONTROLS)))
    hub_ = U[len(CONTROLS):, :]
    opti.subject_to(opti.bounded(-max_steer, steer_, max_steer))
    for kk in (kfl, kfr):
        opti.subject_to(opti.bounded(-0.25, kk, 0.95 if car.hub_front else 0.0))
    for kk in (krl, krr):
        opti.subject_to(opti.bounded(-0.25, kk, 0.95))
    t_lo = -motor.peak_torque if regen else 0.0
    opti.subject_to(opti.bounded(t_lo if car.central_drive else 0.0, torque_,
                                 motor.peak_torque if car.central_drive else 0.0))
    opti.subject_to(opti.bounded(0.0, brake_, car.max_brake_torque))
    if hubs:
        from kmslap.car.drivetrain import HubMotor
        peak = HubMotor(car.hub_motor).peak_torque
        opti.subject_to(opti.bounded(-peak if regen else 0.0, ca.vec(hub_), peak))
    if hubs or regen:
        # the motors' net torque never drives against the hydraulic brakes (the driver is
        # on the brake: EV.3.1.3 lets the controller only take torque away, and EV.4.7
        # cuts power with the accelerator pressed). Relaxed complementarity: 1000 N m of
        # brake leaves 50 N m of drive at the wheels
        opti.subject_to(ca.vec(brake_ / 1000 * net / 1000) <= 0.05)

    d = lambda A: (nxt(A) - A) / ds
    penalty = ds * (w_steer * ca.sumsqr(d(steer_))
                    + w_drive * (ca.sumsqr(d(torque_ / 100)) + ca.sumsqr(d(brake_ / 1000)) + ca.sumsqr(d(hub_ / HUB_SCALE)))
                    + w_slip * ca.sumsqr(d(ca.vertcat(kfl, kfr, krl, krr))))
    opti.minimize(lap_time + penalty)
    opti.set_initial(Xs, x0 / X_SCALE[:, None])
    opti.set_initial(Us, u0 / u_scale[:, None])
    opti.solver("ipopt", {"print_time": verbose},
                {"print_level": 5 if verbose else 0, "max_iter": max_iter, "tol": 1e-6,
                 "mu_strategy": "adaptive", "nlp_scaling_method": "gradient-based"})
    from kmslap import progress
    try:
        with progress.task(f"{label + ': ' if label else ''}lap plan, double-track min-time OCP "
                           f"({len(ref.s)} nodes, Ipopt)") as task:
            opti.callback(lambda i: task.note(f"iteration {i}"))
            sol = opti.solve()
        stats, val = sol.stats(), sol.value
    except RuntimeError:
        stats, val = opti.stats(), opti.debug.value
    xv = np.asarray(val(X))
    uv = np.asarray(val(U))
    xs = dict(zip(STATES, xv))
    us = dict(zip(names, uv))

    nx_, ny_ = track_mod.normals(ref)
    line = Path(ref.x + xs["n"] * nx_, ref.y + xs["n"] * ny_, closed=True)
    fine = track_mod._resample(line, track.center.s[1] - track.center.s[0])
    at = lambda a: np.interp(fine.s, line.s, a, period=line.length)
    return MinTimeDT(fine, at(xs["v"]), at(us["steer"]), at(xs["beta"]), at(xs["r"]),
                     float(val(lap_time)), ref.s, xs, us,
                     {"success": stats.get("success", False), "iterations": stats.get("iter_count"),
                      "status": stats.get("return_status"), "lap_time_qss": warm.lap_time})
