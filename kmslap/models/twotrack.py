"""Double-track (four-wheel) vehicle model in CasADi, for driving the car around:
corners, a skidpad, laps. Same approach as model.py: one semi-implicit Euler step
written symbolically, compiled to C for the loop in drive.py, differentiable.

Body frame: x forward, y left, yaw counter-clockwise. Wheels FL, FR, RL, RR.

    chassis       m (dvx/dt - r vy) = sum Fx_body - drag - rolling
                  m (dvy/dt + r vx) = sum Fy_body
                  Iz dr/dt          = sum (x_i Fy_i - y_i Fx_i)
    front wheels  J_w domega/dt = -Fx r - brake
    rear wheels   both coupled to the diff carrier (motor rotor reflected through the
                  final drive); a limited-slip clutch torque, preload + lock * input
                  torque, resists the wheels' speed difference (Drexler-style)
    tires         relaxation-length slip states per wheel:
                  sigma_x dkappa/dt = (omega r - u) - |u| kappa
                  sigma_y dalpha/dt = -w - |u| alpha
                  (u, w: contact patch velocity along / across the wheel), forces
                  from the combined-slip Pacejka table, PacejkaTire.symbolic_fxy
    loads         static + longitudinal transfer (lagged, pitch_tau) + lateral
                  transfer split by roll_stiffness_front (lagged, roll_tau) + aero
    powertrain    inverter lag, motor map, 80 kW, pack (model.py's), with the pack's
                  open-circuit voltage from its state of charge and its resistance
                  from its temperature
    pack          dSOC/dt = -I / capacity;  C dT/dt = I^2 R(T) - hA (T - ambient)
                  (no active cooling); current limited to hold the loaded cells at
                  the BMS undervoltage limit
    regen         off unless enabled (params(regen=...)): then a negative torque
                  command generates, rear wheels only (through the diff, coast side),
                  limited by the regen torque setting, the motor map, a minimum speed
                  with a fade band, the pack's charge current and cell max voltage;
                  shaft -> pack through the motor efficiency map, the inverter's and
                  the cells' coulombic efficiency. Off, the step is exactly the
                  motoring-only one.

Inputs per step: motor torque command (Nm), road wheel steer angle (rad, + left),
total brake torque at the wheels (Nm), DRS wing position (0 closed .. 1 open: drag,
downforce and aero balance blend between Car's closed and drs_*_open values).
"""
from pathlib import Path

import casadi as ca
import numpy as np

from kmslap.models import accel1d as model
from kmslap.car.params import G, Car
from kmslap.car.powertrain import MotorMap, Pack, motor_efficiency_map
from kmslap import dirs

WHEELS = ("fl", "fr", "rl", "rr")
STATES = (["X", "Y", "psi", "vx", "vy", "r"] + [f"omega_{w}" for w in WHEELS]
          + [f"kappa_{w}" for w in WHEELS] + [f"alpha_{w}" for w in WHEELS]
          + ["ax_filt", "ay_filt", "t_lagged", "bus_v", "soc", "pack_temp"]
          + [f"hub_t_{w}" for w in WHEELS]                   # hub motors' lagged torque
          + ["charge_peak_used"])                            # s of the pack's charge peak used
INPUTS = ["torque_cmd", "steer", "brake_torque", "drs"] + [f"hub_cmd_{w}" for w in WHEELS]
OUTPUTS = (["ax", "ay"] + [f"fx_{w}" for w in WHEELS] + [f"fy_{w}" for w in WHEELS]
           + [f"fz_{w}" for w in WHEELS]
           + ["motor_torque", "bus_i", "motor_rpm", "drag", "diff_torque", "pack_heat",
              "pack_v_oc", "pack_resistance"]
           + [f"hub_torque_{w}" for w in WHEELS] + ["hub_power", "drive_power"])
CAR_PARAMS = ["mass", "cg_height", "wheelbase", "front_weight", "tire_radius", "final_drive",
              "driveline_eff", "motor_eff", "cda", "cla", "aero_front", "rho", "crr",
              "wheel_inertia", "rear_driveline_inertia", "pitch_tau", "roll_tau", "torque_tau",
              "power_limit", "track_front", "track_rear", "yaw_inertia", "roll_stiffness_front",
              "diff_preload", "diff_lock_drive", "diff_lock_coast", "diff_slip_eps",
              "brake_bias_front", "drs_cda_open", "drs_cla_open", "drs_aero_front_open",
              "regen_inverter_eff", "regen_coulombic_eff", "regen_min_speed", "regen_fade_speed"]
PARAMS = CAR_PARAMS + ["relaxation_length", "relaxation_length_lat", "tire_scale",
                       "rotor_inertia", "pack_series", "pack_capacity_ah", "pack_resistance_25c",
                       "pack_max_current", "pack_thermal_mass", "pack_h_a", "pack_ambient_c",
                       "pack_resistance_temp_coeff", "pack_v_cell_min", "pack_v_cell_max",
                       "pack_max_charge_current", "pack_peak_charge_current", "pack_peak_charge_s",
                       "regen_enabled", "regen_max_torque",
                       "pack_ext", "pack_ext_v_oc", "pack_ext_r",
                       "central_drive"] + [f"hub_{w}" for w in WHEELS] + [
                       "hub_ratio", "hub_gear_eff", "hub_rotor_inertia", "dt"]
BRAKE_EPS = 2.0   # rad/s, brake torque fades to 0 on a stopped wheel (numerical)


def params(car: Car, motor: MotorMap, pack: Pack, tire, *, relaxation_length: float = 0.15,
           relaxation_length_lat: float = 0.3, dt: float = 2e-4, regen=None) -> np.ndarray:
    """regen: a driver.RegenPolicy (or anything with max_torque) enables regen; None:
    off, and the step is exactly the motoring-only one."""
    p = {k: getattr(car, k) for k in CAR_PARAMS}
    for k in ("cda", "cla", "aero_front"):       # no DRS: open = closed
        if p[f"drs_{k}_open"] is None:
            p[f"drs_{k}_open"] = p[k]
    p.update(relaxation_length=relaxation_length, relaxation_length_lat=relaxation_length_lat,
             tire_scale=tire.scale, rotor_inertia=motor.rotor_inertia, pack_series=pack.series,
             pack_capacity_ah=pack.capacity_ah, pack_resistance_25c=pack.resistance,
             pack_max_current=pack.max_current, pack_thermal_mass=pack.thermal_mass,
             pack_h_a=pack.h_a, pack_ambient_c=pack.ambient_c,
             pack_resistance_temp_coeff=pack.resistance_temp_coeff, pack_v_cell_min=pack.v_cell_min,
             pack_v_cell_max=pack.v_cell_max, pack_max_charge_current=pack.max_charge_current,
             pack_peak_charge_current=pack.peak_charge_current, pack_peak_charge_s=pack.peak_charge_s,
             regen_enabled=0.0 if regen is None else 1.0,
             regen_max_torque=0.0 if regen is None else regen.max_torque,
             pack_ext=0.0, pack_ext_v_oc=0.0, pack_ext_r=0.0,
             central_drive=float(car.central_drive), hub_fl=float(car.hub_front), hub_fr=float(car.hub_front),
             hub_rl=float(car.hub_rear), hub_rr=float(car.hub_rear), hub_ratio=car.hub_ratio,
             hub_gear_eff=car.hub_gear_eff, hub_rotor_inertia=_hub(car.hub_motor).rotor_inertia, dt=dt)
    return np.array([p[k] for k in PARAMS], dtype=float)


def _hub(kind: str):
    from kmslap.car.drivetrain import HubMotor
    return HubMotor(kind)


def initial_state(car: Car, pack: Pack, *, vx: float = 0.0, soc: float = 1.0,
                  pack_temp: float | None = None, X: float = 0.0, Y: float = 0.0,
                  psi: float = 0.0) -> np.ndarray:
    """Rolling straight at vx (wheels at road speed), pack at soc and temperature
    (ambient by default)."""
    s = dict.fromkeys(STATES, 0.0)
    s.update(X=X, Y=Y, psi=psi, vx=vx, soc=soc,
             pack_temp=pack.ambient_c if pack_temp is None else pack_temp)
    for w in WHEELS:
        s[f"omega_{w}"] = vx / car.tire_radius
    s["bus_v"] = pack.series * float(np.interp(soc, pack.ocv_soc, pack.ocv_cell))
    return np.array([s[k] for k in STATES])


def step_function(motor: MotorMap, tire, pack: Pack, hub_kind: str = "dti") -> ca.Function:
    """step(state, inputs, params) -> (next state, outputs). hub_kind: the hub motors'
    tables (car.hub_motor), used where params put hub motors."""
    s = ca.SX.sym("state", len(STATES))
    u = ca.SX.sym("inputs", len(INPUTS))
    pv = ca.SX.sym("params", len(PARAMS))
    x = {k: s[i] for i, k in enumerate(STATES)}
    p = {k: pv[i] for i, k in enumerate(PARAMS)}
    t_cmd, steer, brake, drs = u[0], u[1], u[2], _clamp(u[3], 0, 1)
    hub_cmd = {w: u[4 + i] for i, w in enumerate(WHEELS)}
    dt, r_t, gr, m = p["dt"], p["tire_radius"], p["final_drive"], p["mass"]
    vx, vy, yaw_rate = x["vx"], x["vy"], x["r"]

    lf = (1 - p["front_weight"]) * p["wheelbase"]   # front axle to CG
    lr = p["front_weight"] * p["wheelbase"]         # CG to rear axle
    pos = {"fl": (lf, p["track_front"] / 2), "fr": (lf, -p["track_front"] / 2),
           "rl": (-lr, p["track_rear"] / 2), "rr": (-lr, -p["track_rear"] / 2)}
    delta = {"fl": steer, "fr": steer, "rl": 0, "rr": 0}

    # --- aero, between closed and open wing
    cla = p["cla"] + drs * (p["drs_cla_open"] - p["cla"])
    cda = p["cda"] + drs * (p["drs_cda_open"] - p["cda"])
    aero_front = p["aero_front"] + drs * (p["drs_aero_front_open"] - p["aero_front"])

    # --- normal loads
    q = 0.5 * p["rho"] * vx * vx
    down = q * cla
    long_tr = m * x["ax_filt"] * p["cg_height"] / p["wheelbase"]       # to the rear axle
    lat = m * x["ay_filt"] * p["cg_height"]                            # roll moment
    lat_f = lat * p["roll_stiffness_front"] / p["track_front"]         # left -> right
    lat_r = lat * (1 - p["roll_stiffness_front"]) / p["track_rear"]
    front_axle = m * G * lr / p["wheelbase"] - long_tr + down * aero_front
    rear_axle = m * G * lf / p["wheelbase"] + long_tr + down * (1 - aero_front)
    fz = {"fl": front_axle / 2 - lat_f, "fr": front_axle / 2 + lat_f,
          "rl": rear_axle / 2 - lat_r, "rr": rear_axle / 2 + lat_r}
    fz = {w: ca.fmax(f, 0) for w, f in fz.items()}

    # --- tire forces (wheel frame), then body frame
    fx, fy, fxb, fyb = {}, {}, {}, {}
    for w in WHEELS:
        fx[w], fy[w] = tire.symbolic_fxy(x[f"kappa_{w}"], x[f"alpha_{w}"], fz[w], p["tire_scale"])
        c, sn = ca.cos(delta[w]), ca.sin(delta[w])
        fxb[w] = c * fx[w] - sn * fy[w]
        fyb[w] = sn * fx[w] + c * fy[w]

    # --- powertrain: inverter lag, motor map, 80 kW, pack at its SOC and temperature
    t_lagged = ca.if_else(p["torque_tau"] > 0,
                          x["t_lagged"] + dt * (t_cmd - x["t_lagged"]) / ca.fmax(p["torque_tau"], 1e-12),
                          t_cmd)
    ocv = ca.interpolant("ocv", "linear", [pack.ocv_soc], pack.ocv_cell)
    v_oc = p["pack_series"] * ocv(_clamp(x["soc"], 0, 1))
    r_pack = p["pack_resistance_25c"] * ca.exp(-p["pack_resistance_temp_coeff"] * (x["pack_temp"] - 25))
    # pack in the loop (kmslap hil): open-circuit voltage and resistance from outside,
    # each tick (a real cell's, fitted live); off, these are the model's own, bit for bit
    v_oc = ca.if_else(p["pack_ext"] > 0, p["pack_ext_v_oc"], v_oc)
    r_pack = ca.if_else(p["pack_ext"] > 0, p["pack_ext_r"], r_pack)
    # BMS undervoltage: no more current than holds the loaded cells at v_cell_min
    i_limit = ca.fmin(p["pack_max_current"],
                      ca.fmax((v_oc - p["pack_series"] * p["pack_v_cell_min"]) / r_pack, 0))
    pp = dict(p, pack_v_oc=v_oc, pack_resistance=r_pack, pack_max_current=i_limit)
    omega_c = 0.5 * (x["omega_rl"] + x["omega_rr"])                  # diff carrier
    omega_m = omega_c * gr
    torque, bus_v, current = model._powertrain(pp, model._motor_max_torque(motor), t_lagged,
                                               omega_m, x["bus_v"])

    # --- regen: picked only when enabled and the command is negative, so with regen
    # off every value below is the motoring branch's, bit for bit
    regen_on = ca.logic_and(p["regen_enabled"] > 0, t_lagged < 0)
    rpm = omega_m * model.RPM_PER_RADS
    fade = _clamp((vx - p["regen_min_speed"])
                  / ca.fmax(p["regen_fade_speed"] - p["regen_min_speed"], 1e-6), 0, 1)
    t_gen = -ca.fmin(ca.fmin(-t_lagged, p["regen_max_torque"]),
                     model._motor_max_torque(motor)(rpm, x["bus_v"])) * fade     # <= 0
    rpms, torques, eff_map = motor_efficiency_map(motor.name)
    motor_eff = ca.interpolant("motor_eff", "linear", [rpms, torques], eff_map.ravel(order="F"))
    eff_gen = motor_eff(ca.vertcat(_clamp(rpm, rpms[0], rpms[-1]), _clamp(-t_gen, torques[0], torques[-1])))
    to_dc = eff_gen * p["regen_inverter_eff"]
    p_dc = t_gen * omega_m * to_dc                                     # <= 0: into the pack
    # pack: P = V I with V = Voc - I R, I < 0; BMS limits charge current and the
    # loaded cell voltage
    i_req = (v_oc - ca.sqrt(v_oc * v_oc - 4 * r_pack * p_dc)) / (2 * r_pack)
    # the BMS's charge current limit: the cells' peak while the peak's time lasts (used
    # up charging above the continuous rating, recovering at a tenth of that rate,
    # ESTIMATE), then the continuous rating; an external pack (HIL) keeps its continuous
    i_chg_max = ca.if_else(ca.logic_and(p["pack_ext"] <= 0, x["charge_peak_used"] < p["pack_peak_charge_s"]),
                           p["pack_peak_charge_current"], p["pack_max_charge_current"])
    i_floor = ca.fmax(-i_chg_max,
                      ca.fmin(0, (v_oc - p["pack_series"] * p["pack_v_cell_max"]) / r_pack))
    i_gen = ca.fmax(i_req, i_floor)
    t_gen = ca.if_else(i_req < i_floor,
                       (v_oc - i_gen * r_pack) * i_gen / ca.fmax(omega_m * to_dc, 1e-6), t_gen)
    torque = ca.if_else(regen_on, t_gen, torque)
    current = ca.if_else(regen_on, i_gen, current)
    bus_v = ca.if_else(regen_on, v_oc - i_gen * r_pack, bus_v)
    # --- hub motors (car.hub_front / hub_rear): each through its gearbox to its wheel.
    # Their power and the rear motor's share the pack and the 80 kW; motoring torques
    # scale down together when the total is over (or the pack can't give it). With no
    # hub motors every value below is the rear-only one, bit for bit.
    hub = _hub(hub_kind)
    rpms, torques, t_max_tbl, eff_tbl = hub.tables()
    hub_tmax = ca.interpolant("hub_tmax", "linear", [rpms], t_max_tbl)
    hub_eff = ca.interpolant("hub_eff", "linear", [rpms, torques], eff_tbl.ravel(order="F"))
    any_hub = sum(p[f"hub_{w}"] for w in WHEELS) > 0
    torque = torque * p["central_drive"]                      # no EMRAX: nothing at the diff
    hub_t, hub_p, hub_wheel = {}, {}, {}
    for w in WHEELS:
        lagged = ca.if_else(p["torque_tau"] > 0, x[f"hub_t_{w}"] + dt * (hub_cmd[w] - x[f"hub_t_{w}"])
                            / ca.fmax(p["torque_tau"], 1e-12), hub_cmd[w])
        om = x[f"omega_{w}"] * p["hub_ratio"]                # motor rad/s
        hrpm = _clamp(om * model.RPM_PER_RADS, 0, rpms[-1])
        lim = hub_tmax(hrpm)
        hub_t[w] = _clamp(lagged, -lim, lim) * p[f"hub_{w}"]
        eta = hub_eff(ca.vertcat(hrpm, _clamp(ca.fabs(hub_t[w]), torques[0], torques[-1])))
        hub_p[w] = ca.if_else(hub_t[w] >= 0, hub_t[w] * om / eta, hub_t[w] * om * eta)
    p_hub = sum(hub_p.values())
    # the rear motor's electrical power as the pack sees it
    p_rear = ca.if_else(regen_on, p_dc * p["central_drive"], ca.fmax(torque, 0) * omega_m / p["motor_eff"])
    p_tot = p_rear + p_hub
    p_drive = ca.fmax(p_rear, 0) + sum(ca.fmax(hub_p[w], 0) for w in WHEELS)
    # pack: motoring within the 80 kW and the pack's current; regen within its charge limit
    v_dis, i_dis, p_del = model._pack_solve(pp, ca.fmax(p_tot, 0))
    scale_m = ca.if_else(p_drive > 1e-6, ca.fmin(1, ca.fmin(p["power_limit"], p_del + ca.fmax(-ca.fmin(p_tot, 0), 0))
                                                         / ca.fmax(p_drive, 1e-6)), 1)
    i_chg = (v_oc - ca.sqrt(ca.fmax(v_oc * v_oc - 4 * r_pack * ca.fmin(p_tot, 0), 0))) / (2 * r_pack)
    i_floor_h = ca.fmax(-i_chg_max,
                        ca.fmin(0, (v_oc - p["pack_series"] * p["pack_v_cell_max"]) / r_pack))
    scale_r = ca.if_else(i_chg < i_floor_h, i_floor_h / ca.fmin(i_chg, -1e-9), 1)
    for w in WHEELS:
        hub_t[w] = ca.if_else(hub_t[w] >= 0, hub_t[w] * scale_m, hub_t[w] * scale_r)
        hub_wheel[w] = ca.if_else(hub_t[w] >= 0, hub_t[w] * p["hub_ratio"] * p["hub_gear_eff"],
                                  hub_t[w] * p["hub_ratio"] / p["hub_gear_eff"])
    torque_h = ca.if_else(torque >= 0, torque * scale_m, torque * scale_r)
    p_tot_s = sum(ca.if_else(hub_t[w] >= 0, hub_p[w] * scale_m, hub_p[w] * scale_r) for w in WHEELS) \
        + ca.if_else(torque >= 0, p_rear * scale_m, p_rear * scale_r)
    i_h = ca.if_else(p_tot_s >= 0, (v_oc - ca.sqrt(ca.fmax(v_oc * v_oc - 4 * r_pack * p_tot_s, 0))) / (2 * r_pack),
                     ca.fmax((v_oc - ca.sqrt(ca.fmax(v_oc * v_oc - 4 * r_pack * p_tot_s, 0))) / (2 * r_pack), i_floor_h))
    torque = ca.if_else(any_hub, torque_h, torque)
    current = ca.if_else(any_hub, i_h, current)
    bus_v = ca.if_else(any_hub, v_oc - i_h * r_pack, bus_v)
    hub_power = ca.if_else(any_hub, p_tot_s - ca.if_else(torque >= 0, p_rear * scale_m, p_rear * scale_r), 0)
    drive_power = ca.if_else(any_hub, p_tot_s, current * bus_v)

    # at the diff: driving loses the chain's efficiency on the way out, regen on the way in
    t_c = ca.if_else(torque >= 0, torque * gr * p["driveline_eff"], torque * gr / p["driveline_eff"])

    # --- wheel spin. Front: free rolling + brakes.
    jw = p["wheel_inertia"]
    brake_f = 0.5 * brake * p["brake_bias_front"]
    brake_r = 0.5 * brake * (1 - p["brake_bias_front"])
    domega = {}
    j_hub = p["hub_rotor_inertia"] * p["hub_ratio"] * p["hub_ratio"]        # rotor, at the wheel
    for w in ("fl", "fr"):
        domega[w] = (hub_wheel[w] - fx[w] * r_t - brake_f * ca.tanh(x[f"omega_{w}"] / BRAKE_EPS)) \
            / (jw + p[f"hub_{w}"] * j_hub)
    # Rear: carrier inertia couples the wheels, the LSD clutch resists their difference
    lock = p["diff_preload"] + ca.if_else(t_c >= 0, p["diff_lock_drive"], p["diff_lock_coast"]) * ca.fabs(t_c)
    t_diff = 0.5 * lock * ca.tanh((x["omega_rl"] - x["omega_rr"]) / p["diff_slip_eps"])
    tau_l = 0.5 * t_c - t_diff - fx["rl"] * r_t - brake_r * ca.tanh(x["omega_rl"] / BRAKE_EPS)
    tau_r = 0.5 * t_c + t_diff - fx["rr"] * r_t - brake_r * ca.tanh(x["omega_rr"] / BRAKE_EPS)
    jc = p["rear_driveline_inertia"] + p["rotor_inertia"] * gr * gr
    a_, b_ = jw + jc / 4, jc / 4                                     # [[a b] [b a]] omega' = tau
    det = a_ * a_ - b_ * b_
    domega["rl"] = (a_ * tau_l - b_ * tau_r) / det
    domega["rr"] = (a_ * tau_r - b_ * tau_l) / det
    # no EMRAX and diff: each rear wheel on its own (its hub motor, if it has one)
    for w in ("rl", "rr"):
        free = (hub_wheel[w] - fx[w] * r_t - brake_r * ca.tanh(x[f"omega_{w}"] / BRAKE_EPS)) \
            / (jw + p[f"hub_{w}"] * j_hub)
        domega[w] = ca.if_else(p["central_drive"] > 0, domega[w], free)
    t_diff = t_diff * p["central_drive"]

    # --- chassis
    fz_tot = m * G + down
    drag = q * cda
    rolling = p["crr"] * fz_tot * ca.tanh(vx / 0.1)
    fx_tot = sum(fxb.values()) - drag - rolling
    fy_tot = sum(fyb.values())
    mz = sum(pos[w][0] * fyb[w] - pos[w][1] * fxb[w] for w in WHEELS)
    ax, ay = fx_tot / m, fy_tot / m                                  # what an IMU at the CG reads
    vx1 = vx + (ax + yaw_rate * vy) * dt
    vy1 = vy + (ay - yaw_rate * vx) * dt
    r1 = yaw_rate + mz / p["yaw_inertia"] * dt
    psi1 = x["psi"] + r1 * dt
    X1 = x["X"] + (vx1 * ca.cos(psi1) - vy1 * ca.sin(psi1)) * dt
    Y1 = x["Y"] + (vx1 * ca.sin(psi1) + vy1 * ca.cos(psi1)) * dt

    # --- slip states, from the updated velocities (semi-implicit, like model.py)
    nxt = {"X": X1, "Y": Y1, "psi": psi1, "vx": vx1, "vy": vy1, "r": r1}
    for w in WHEELS:
        om1 = ca.fmax(x[f"omega_{w}"] + domega[w] * dt, 0)
        px, py = pos[w]
        cvx, cvy = vx1 - r1 * py, vy1 + r1 * px                      # contact patch velocity
        c, sn = ca.cos(delta[w]), ca.sin(delta[w])
        uw, ww = c * cvx + sn * cvy, -sn * cvx + c * cvy
        k, al = x[f"kappa_{w}"], x[f"alpha_{w}"]
        nxt[f"omega_{w}"] = om1
        nxt[f"kappa_{w}"] = k + dt * ((om1 * r_t - uw) - ca.fabs(uw) * k) / p["relaxation_length"]
        nxt[f"alpha_{w}"] = al + dt * (-ww - ca.fabs(uw) * al) / p["relaxation_length_lat"]

    lag = lambda prev, new, tau: ca.if_else(tau > 0, prev + dt * (new - prev) / ca.fmax(tau, 1e-12), new)
    heat = current * current * r_pack
    nxt.update(
        ax_filt=lag(x["ax_filt"], ax, p["pitch_tau"]),
        ay_filt=lag(x["ay_filt"], ay, p["roll_tau"]),
        t_lagged=t_lagged, bus_v=bus_v,
        **{f"hub_t_{w}": ca.if_else(p["torque_tau"] > 0, x[f"hub_t_{w}"] + dt * (hub_cmd[w] - x[f"hub_t_{w}"])
                                    / ca.fmax(p["torque_tau"], 1e-12), hub_cmd[w]) * p[f"hub_{w}"] for w in WHEELS},
        soc=x["soc"] - ca.if_else(current < 0, current * p["regen_coulombic_eff"], current)
        * dt / (p["pack_capacity_ah"] * 3600),
        pack_temp=x["pack_temp"] + dt * (heat - p["pack_h_a"] * (x["pack_temp"] - p["pack_ambient_c"]))
        / p["pack_thermal_mass"],
        charge_peak_used=_clamp(x["charge_peak_used"] + dt * ca.if_else(current < -p["pack_max_charge_current"], 1.0, -0.1),
                                0, p["pack_peak_charge_s"] + 0.1),
    )
    out = ([ax, ay] + [fx[w] for w in WHEELS] + [fy[w] for w in WHEELS] + [fz[w] for w in WHEELS]
           + [torque, current, omega_m * model.RPM_PER_RADS, drag, 2 * t_diff, heat, v_oc, r_pack]
           + [hub_t[w] for w in WHEELS] + [hub_power, drive_power])
    return ca.Function("car_step", [s, u, pv],
                       [ca.vertcat(*[nxt[k] for k in STATES]), ca.vertcat(*out)],
                       ["state", "inputs", "params"], ["next", "out"])


def _clamp(v, lo, hi):
    return ca.fmin(ca.fmax(v, lo), hi)


def car_step(motor: MotorMap, tire, pack: Pack, param_vector, state0, hub_kind: str = "dti") -> model.CompiledStep:
    """step_function, compiled, for drive()'s loop."""
    from kmslap.car.drivetrain import DATASHEETS
    key = model._motor_key(motor) + (tire.table_key_2d(),
                                     model._hash_arrays(pack.ocv_soc, pack.ocv_cell),
                                     model._hash_arrays(*motor_efficiency_map(motor.name)), hub_kind)
    here = Path(__file__)
    return model.CompiledStep("car_step", lambda: step_function(motor, tire, pack, hub_kind), key,
                              [here, here.with_name("accel1d.py"), dirs.PACKAGE / "car" / "tire.py",
                               dirs.PACKAGE / "car" / "powertrain.py", dirs.PACKAGE / "car" / "drivetrain.py",
                               *[dirs.DATASHEETS / f for f in DATASHEETS.values()]],
                              len(INPUTS), len(OUTPUTS), param_vector, state0)
