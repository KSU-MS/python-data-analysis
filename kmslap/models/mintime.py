"""Minimum lap time: racing line and speed solved together, optimal control.

The free-trajectory quasi-steady-state formulation (Christ et al., TUM; Veneri &
Massaro, Padova): the car follows any line within the track, and at every point its
acceleration (longitudinal ax, lateral ay) stays inside its grip envelope at that speed
(models/envelope.py, which carries the double-track model's tires, load transfer, aero,
diff, motor map, 80 kW and stability). So corner entry, apex and exit are traded
against each other with the real car's limits, unlike the minimum-curvature line
(geometry only) with a speed profile laid on afterwards.

Curvilinear coordinates on the track centerline, arc length s as the independent
variable:
    states    n  lateral offset (+ left), xi  heading relative to the centerline, v  speed
    controls  ax  longitudinal acceleration, kv  the car's path curvature (ay = v^2 kv)
    dt/ds  = (1 - n k(s)) / (v cos xi)
    dn/ds  = (1 - n k(s)) tan xi
    dxi/ds = dt/ds v kv - k(s)
    dv/ds  = dt/ds ax
minimize the lap time (integral of dt/ds) plus a small penalty on control rates,
subject to n within the track's free width (per side), (ax, ay) inside grip x the envelope at v
(a smooth convex fit of it, gg_fit), the steering lock on kv, a yaw acceleration limit
(the transient the quasi-steady envelope leaves out); periodic (a flying lap).
Trapezoidal collocation, solved with Ipopt, started from the minimum-curvature line.

Optional energy budget per lap (endurance): battery energy = integral of P dt with
P = F v / eta_drive when driving and F v eta_regen when braking with regen (F the
tractive force m ax + drag + rolling); the optimizer then decides where to lift.
"""
from dataclasses import dataclass

import casadi as ca
import numpy as np

from kmslap.driving.paths import Path
from kmslap.events import track as track_mod


@dataclass
class MinTime:
    line: Path                # the optimal line (driving resolution)
    v: np.ndarray             # optimal speed along `line`
    lap_time: float           # s, the OCP's (quasi-steady) lap time
    s: np.ndarray             # centerline arc length of the nodes
    n: np.ndarray             # lateral offset at the nodes
    speed: np.ndarray         # speed at the nodes
    ax: np.ndarray
    ay: np.ndarray
    energy_kwh: float | None  # battery energy per lap, if modelled
    stats: dict


def gg_fit(env, low_speed: float = 7.0) -> dict:
    """The envelope as a smooth convex super-ellipse per speed (the usual way to give
    g-g-v limits to an optimizer: the raw envelope has ripple and steering-lock dips
    that leave Ipopt wandering):
        (|ay| / ay_max)^p + (|ax| / ax_lim)^p <= 1
    with ax_lim and p the acceleration side's or the braking side's. Fitted radially to
    the envelope's points at each solved speed, then scaled to sit inside them all.
    Below low_speed the lateral limit is steering lock, not grip: the tire limits there
    are taken from low_speed and the lock is its own constraint in solve()."""
    from scipy.optimize import least_squares

    out = {k: [] for k in ("ay_max", "acc", "brk", "p_acc", "p_brk")}
    ref_i = int(np.searchsorted(env.speeds, low_speed))
    for i in range(len(env.speeds)):
        j = max(i, ref_i)
        ax, ay, rho = env.ax[j], env.ay[j], env.rho[j]
        aymax, acc, brk = ay.max(), rho[0], rho[-1]
        ps, shrink = [], 1.0
        for sel, lim in [(ax >= 0, acc), (ax < 0, brk)]:
            th = np.arctan2(ay[sel], ax[sel])
            c, s_ = np.abs(np.cos(th)), np.abs(np.sin(th))
            r_fit = lambda p: ((s_ / aymax) ** p + (c / lim) ** p) ** (-1 / p)
            p = least_squares(lambda x: r_fit(x[0]) - rho[sel], [2.0], bounds=(1.0, 8.0)).x[0]
            ps.append(p)
            shrink = min(shrink, float(np.min(rho[sel] / r_fit(p))))
        # conservative: scaled to sit inside the envelope everywhere (an optimizer finds
        # and uses any grip a fit invents)
        for key, val in zip(out, (aymax * shrink, acc * shrink, brk * shrink, *ps)):
            out[key].append(val)
    return {"speeds": np.asarray(env.speeds, float), **{k: np.asarray(v) for k, v in out.items()}}


def _gg_constraint(fit, v, ax, ay, grip, eps=1e-4):
    """<= 1 inside the fitted envelope (grip scales it)."""
    sp = fit["speeds"]
    f = {k: ca.interpolant(f"gg_{k}", "bspline", [sp], fit[k]) for k in ("ay_max", "acc", "brk", "p_acc", "p_brk")}
    vc = ca.fmin(ca.fmax(v, sp[0]), sp[-1])
    at = lambda name: f[name](vc.T).T
    w = 0.5 * (1 + ca.tanh(ax / 0.3))                     # accelerating vs braking side
    lim = w * at("acc") + (1 - w) * at("brk")
    p = w * at("p_acc") + (1 - w) * at("p_brk")
    return (ca.sqrt(ay ** 2 + eps) / (grip * at("ay_max"))) ** p + (ca.sqrt(ax ** 2 + eps) / (grip * lim)) ** p


def solve(center, env, *, car=None, width: float = track_mod.TRACK_WIDTH,
          car_width: float = track_mod.CAR_WIDTH, margin: float = track_mod.EDGE_MARGIN,
          grip: float = 1.0, step: float = 2.0, max_steering_wheel: float = np.radians(120),
          max_yaw_accel: float = 6.0, energy_kwh: float | None = None,
          regen_eff: float | None = None, smooth_weight: float = 1e-3,
          max_iter: int = 6000, verbose: bool = False, build_only: bool = False) -> MinTime:
    """center: a Track, or a centerline (then the track is `width` wide).
    grip: fraction of the envelope used (1 = the car's limit; a driver holds ~0.9).
    energy_kwh: battery energy allowed per lap (None: unconstrained).
    regen_eff: battery energy recovered per unit of braking work (None: no regen,
        braking costs nothing and returns nothing)."""
    from kmslap.car.params import G, Car
    from kmslap.models.envelope import speed_profile

    car = car or Car()
    track = center if isinstance(center, track_mod.Track) else track_mod.constant_width(center, width)
    center = track.center
    tr = track.resampled(step)
    ref = tr.center
    lo, hi = tr.limits(car_width, margin)
    N = len(ref.s)
    ds = ref.length / N
    k = ref.k
    fit = gg_fit(env)
    kappa_max = np.tan(max_steering_wheel / car.steer_ratio) / car.wheelbase   # steering lock

    # warm start: the minimum-curvature line and its envelope speed profile, sampled at
    # each node (the nearest point of that line, projected on the node's normal)
    # in this grid's own coordinates, so the start is consistent (projecting another
    # line onto the grid made kinks); slightly inside the limit to start interior
    n0 = track_mod.mincurv_offsets(ref, lo + 0.05 * (hi - lo), hi - 0.05 * (hi - lo))
    nx0, ny0 = track_mod.normals(ref)
    line0 = Path(ref.x + n0 * nx0, ref.y + n0 * ny0, closed=True)
    # the rest follows from the model's own kinematics
    dn = (np.roll(n0, -1) - np.roll(n0, 1)) / (2 * ds)
    xi0 = np.arctan(dn / np.maximum(1 - n0 * k, 0.2))
    dxi = (np.roll(xi0, -1) - np.roll(xi0, 1)) / (2 * ds)
    kv0 = (dxi + k) * np.cos(xi0) / np.maximum(1 - n0 * k, 0.2)
    v0 = np.clip(speed_profile(line0, env, margin=0.8 * grip), 3.0, float(env.speeds[-1]) - 0.5)   # well inside the (shrunk) fit
    dv = (np.roll(v0, -1) - np.roll(v0, 1)) / (2 * ds)
    ax0 = dv * v0 * np.cos(xi0) / np.maximum(1 - n0 * k, 0.2)

    opti = ca.Opti()
    n = opti.variable(N)
    xi = opti.variable(N)
    v = opti.variable(N)
    ax = opti.variable(N)
    kv = opti.variable(N)
    nxt = lambda x: ca.vertcat(x[1:], x[0])          # periodic

    def rates(n_, xi_, v_, ax_, kv_, k_):
        sf = (1 - n_ * k_) / (v_ * ca.cos(xi_))
        return sf, ca.vertcat((1 - n_ * k_) * ca.tan(xi_), sf * v_ * kv_ - k_, sf * ax_)

    kk = ca.DM(k)
    sf, f = rates(n, xi, v, ax, kv, kk)
    sf1, f1 = rates(nxt(n), nxt(xi), nxt(v), nxt(ax), nxt(kv), ca.vertcat(kk[1:], kk[0]))
    for i, x in enumerate([n, xi, v]):
        opti.subject_to(nxt(x) - x == ds / 2 * (f[i * N:(i + 1) * N] + f1[i * N:(i + 1) * N]))
    lap_time = ca.sum1(ds / 2 * (sf + sf1))
    ay = v ** 2 * kv
    opti.subject_to(opti.bounded(lo, n, hi))
    opti.subject_to(opti.bounded(-1.0, xi, 1.0))
    opti.subject_to(opti.bounded(2.0, v, float(env.speeds[-1])))
    opti.subject_to(_gg_constraint(fit, v, ax, ay, grip) <= 1)
    opti.subject_to(opti.bounded(-0.97 * kappa_max, kv, 0.97 * kappa_max))
    # yaw acceleration: the quasi-steady envelope assumes any corner's grip is there at
    # once, but changing direction takes yaw moment from the same tires. Unbounded, the
    # optimum flicks through slaloms the double-track car can't follow (it spins); the
    # limit is what the double-track car shows on clean laps (p99 ~5.8 rad/s^2).
    yaw_acc = ax * kv + v * (nxt(kv) - kv) / (ds * sf)
    opti.subject_to(opti.bounded(-max_yaw_accel, yaw_acc, max_yaw_accel))

    energy = None
    if energy_kwh is not None:
        m = car.mass
        force = m * ax + 0.5 * car.rho * car.cda * v ** 2 + car.crr * (m * G + 0.5 * car.rho * car.cla * v ** 2)
        eta_d = car.motor_eff * car.driveline_eff
        # smooth split of tractive force into driving (>0) and braking (<0)
        pos = 0.5 * (force + ca.sqrt(force ** 2 + 100.0))
        neg = force - pos
        power = pos * v / eta_d + (neg * v * regen_eff if regen_eff else 0)
        energy = ca.sum1(ds * power * sf)
        opti.subject_to(energy <= energy_kwh * 3.6e6)

    dax, dkv = nxt(ax) - ax, nxt(kv) - kv
    opti.minimize(lap_time + smooth_weight * (ca.sumsqr(dax) / ds + 1e3 * ca.sumsqr(dkv) / ds))

    opti.set_initial(n, n0)
    opti.set_initial(xi, xi0)
    opti.set_initial(v, v0)
    opti.set_initial(kv, kv0)
    opti.set_initial(ax, ax0)
    if build_only:
        return opti, dict(n=n, xi=xi, v=v, ax=ax, kv=kv, lap_time=lap_time)
    opti.solver("ipopt", {"print_time": verbose},
                {"print_level": 5 if verbose else 0, "max_iter": max_iter, "tol": 1e-6,
                 "mu_strategy": "adaptive"})
    try:
        sol = opti.solve()
        stats = sol.stats()
        val = sol.value
    except RuntimeError:
        stats = opti.stats()
        val = opti.debug.value
    n_opt, v_opt = np.asarray(val(n)).ravel(), np.asarray(val(v)).ravel()
    nx_, ny_ = track_mod.normals(ref)
    line = Path(ref.x + n_opt * nx_, ref.y + n_opt * ny_, closed=True)
    fine = track_mod._resample(line, center.s[1] - center.s[0])
    v_fine = np.interp(fine.s, line.s, v_opt, period=line.length)
    return MinTime(fine, v_fine, float(val(lap_time)), ref.s, n_opt, v_opt,
                   np.asarray(val(ax)).ravel(), np.asarray(val(ay)).ravel(),
                   float(val(energy)) / 3.6e6 if energy is not None else None,
                   {"success": stats.get("success", False), "iterations": stats.get("iter_count"),
                    "status": stats.get("return_status")})
