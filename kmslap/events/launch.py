"""Launch torque strategies: functions f(t, v) -> requested motor torque (Nm), passed to
simulate(torque_request=...).

- linear_ramp: what a driver does with the pedal (KS5e drivers: 0.06-0.1 s to full)
- grip_launch_map: torque vs vehicle speed that holds the rear tires at peak grip as
  load transfers rearward. The analytic optimum for a known tire, and implementable
  on the car from front wheel speed alone.
- optimize_launch_map: free-form torque vs speed, optimized numerically against accel
  time (a check on grip_launch_map)

Time-based ramps don't help here: any ramp that ends at full torque eventually asks
for more than the tires can take, so what matters is the torque cap, not the ramp.
"""
import numpy as np
from scipy.optimize import minimize

from kmslap.car.params import Car
from kmslap.car.powertrain import MotorMap
from kmslap.events.accel import grip_limited_accel, simulate
from kmslap.car.tire import default_tire


def linear_ramp(peak: float, ramp_time: float):
    if ramp_time <= 0:
        return lambda t, v: peak
    return lambda t, v: peak * min(t / ramp_time, 1.0)


def grip_launch_map(car: Car = None, motor: MotorMap | str = "228HV", tire=None,
                    margin: float = 1.0, v_max: float = 40.0, dv: float = 0.25):
    """Motor torque vs speed that puts the rear tires at margin * peak grip.

    Includes the torque that goes into spinning up the rear driveline (wheels, rotor
    reflected through the final drive) along with the car. Returns f(t, v).
    """
    car = car or Car()
    motor = MotorMap(motor) if isinstance(motor, str) else motor
    tire = tire or default_tire()
    r, gr = car.tire_radius, car.final_drive
    j_rear = 2 * car.wheel_inertia + car.rear_driveline_inertia + motor.rotor_inertia * gr * gr
    speeds = np.arange(0.0, v_max + dv, dv)
    torques, a = [], 0.0
    for v in speeds:
        a, fx = grip_limited_accel(car, tire, v, j_rear, a, margin)
        torques.append((fx * r + j_rear * a / r) / (gr * car.driveline_eff))
    speeds, torques = speeds.tolist(), np.minimum(torques, motor.peak_torque).tolist()
    f = lambda t, v: float(np.interp(v, speeds, torques))
    f.speeds, f.torques = speeds, torques
    return f


def speed_map(knot_speeds, knot_torques):
    """Torque limit vs vehicle speed, linear between knots, held flat past the last."""
    vs, qs = list(knot_speeds), list(knot_torques)
    f = lambda t, v: float(np.interp(v, vs, qs))
    f.speeds, f.torques = vs, qs
    return f


def optimize_launch_map(knot_speeds=(0.0, 1.0, 2.0, 4.0, 7.0, 10.0, 15.0, 20.0),
                        start_margin: float = 0.97, max_evals: int = 400, **sim_kwargs):
    """Numerically optimize an open-loop torque-vs-speed launch map (no TC) for accel
    time, starting from grip_launch_map(margin=start_margin).

    Returns (map f(t, v), Result). ~0.2 s per evaluation, so ~1 min.
    """
    car = sim_kwargs.pop("car", None) or Car()
    motor = sim_kwargs.pop("motor", "228HV")
    motor = MotorMap(motor) if isinstance(motor, str) else motor
    tire = sim_kwargs.pop("tire", None) or default_tire()
    peak = motor.peak_torque
    start = grip_launch_map(car, motor, tire, margin=start_margin)
    x0 = np.array([start(0, v) for v in knot_speeds]) / peak

    def run(q):
        f = speed_map(knot_speeds, np.clip(q, 0, 1) * peak)
        return simulate(car, motor, tire=tire, torque_request=f, **sim_kwargs)

    def cost(q):
        t = run(q).accel_time
        return t if np.isfinite(t) else 99.0

    res = minimize(cost, x0, method="Powell", bounds=[(0, 1)] * len(knot_speeds),
                   options={"maxfev": max_evals, "xtol": 1e-3, "ftol": 1e-5})
    return speed_map(knot_speeds, np.clip(res.x, 0, 1) * peak), run(res.x)
