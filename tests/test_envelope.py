"""g-g-v envelope by optimization, and driving to it."""
import math

import numpy as np
import pytest

from accel import Car
from accel.drive import drive
from accel.driver import Path, PathFollower
from accel.envelope import gg_envelope, speed_profile


@pytest.fixture(scope="module")
def env():
    return gg_envelope(Car(), speeds=[8.0, 14.0, 20.0], n_angles=9, cache=False)


def test_envelope_solves(env):
    assert env.converged.all()
    for i in range(len(env.speeds)):
        accel, lat, brake = env.rho[i, 0], env.ay_max(env.speeds[i]), env.rho[i, -1]
        assert 0.6 * 9.81 < accel < 1.0 * 9.81      # traction / motor limited launch
        assert 1.1 * 9.81 < lat < 2.2 * 9.81      # downforce adds grip with speed
        assert 1.3 * 9.81 < brake < 2.6 * 9.81      # all four tires
        # combined: less acceleration available while cornering hard
        acc_hi, brk_hi = env.ax_limits(env.speeds[i], 0.8 * lat)
        assert 0 < acc_hi < accel and 0 > brk_hi > -brake


def test_driving_to_the_envelope_holds(env):
    # steady circle at 90% of the envelope's lateral limit: the time-domain car holds it
    radius = 15.0
    v = math.sqrt(0.9 * env.ay_max(14.0) * radius)
    circle = Path.from_segments([("arc", radius, 540.0)])
    d = drive(PathFollower(circle, speed=v), Car(), vx0=v, t_max=10).df.query("t > 5")
    assert abs(np.degrees(d.beta)).max() < 8
    assert d.speed.mean() == pytest.approx(v, rel=0.02)


def test_speed_profile(env):
    radius = 15.0
    path = Path.from_segments([("straight", 30), ("arc", radius, 180), ("straight", 30)])
    v = speed_profile(path, env, margin=0.9)
    mid = v[np.argmin(np.abs(path.s - (30 + radius * math.pi / 2)))]
    assert mid ** 2 / radius == pytest.approx(0.9 * env.ay_max(mid), rel=0.05)
    assert v[-1] > mid                      # accelerates out of the corner
    assert v[0] > mid                       # brakes into it
