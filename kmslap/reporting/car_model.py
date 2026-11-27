"""The FSAE car drawn from primitives, for the 3D views: the simulated log's Foxglove
scene (mcap_log.car_scene) and the Rerun race (race3d). Tub and nose cone, sidepods,
wings with endplates and mounts, roll hoop, driver, wheels with rims.

    for p in body(car): ...          # p.kind "cube" / "cylinder" / "sphere", in the car's frame
    for w in wheels(car): ...        # each wheel's position and whether it steers

The car's frame: x forward, y left, z up, origin on the ground under the CG. Colours are
RGB 0..1; `paint` (the tub and nose) is the team's gold unless a view recolours it.
"""
import math
from dataclasses import dataclass

import numpy as np

GOLD, BLACK, SILVER, WHITE = (0.99, 0.73, 0.19), (0.09, 0.09, 0.1), (0.75, 0.76, 0.78), (0.95, 0.95, 0.95)


def quat(axis: str, angle):
    """Quaternion(s) (x, y, z, w) of a rotation about one axis."""
    angle = np.asarray(angle, float)
    s, c = np.sin(angle / 2), np.cos(angle / 2)
    q = np.zeros(angle.shape + (4,))
    q[..., "xyz".index(axis)] = s
    q[..., 3] = c
    return q


def quat_mul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz])


@dataclass
class Part:
    kind: str                   # "cube", "cylinder" (axis along its z) or "sphere"
    center: tuple               # m, the car's frame
    size: tuple                 # m: cube edges; cylinder (diameter, diameter, length); sphere diameters
    color: tuple                # RGB 0..1
    quat: tuple = (0.0, 0.0, 0.0, 1.0)
    top: float = 1.0            # cylinder: the top's diameter over the bottom's (a cone < 1)
    paint: bool = False         # the livery: a view may recolour it


def body(car) -> list[Part]:
    """Everything but the wheels."""
    lf, lr = car.cg_to_front, car.wheelbase - car.cg_to_front
    front, rear = lf + 0.1, -lr - 0.05
    parts = [Part("cube", ((front + rear) / 2 - 0.05, 0, 0.33), (front - rear - 0.1, 0.62, 0.42), GOLD, paint=True),
             Part("cylinder", (front + 0.25, 0, 0.3), (0.4, 0.55, 0.55), GOLD,
                  tuple(quat("y", math.pi / 2)), top=0.35, paint=True)]                    # tub, nose cone
    for side in (1, -1):
        parts += [Part("cube", (-0.15, side * 0.42, 0.26), (0.9, 0.26, 0.32), BLACK),           # sidepods
                  Part("cube", (front + 0.55, side * 0.7, 0.18), (0.45, 0.02, 0.22), BLACK),    # front endplates
                  Part("cube", (rear - 0.35, side * 0.5, 1.0), (0.55, 0.02, 0.4), BLACK),       # rear endplates
                  Part("cube", (rear - 0.15, side * 0.15, 0.75), (0.04, 0.03, 0.55), BLACK)]    # wing mounts
    parts += [Part("cube", (front + 0.55, 0, 0.1), (0.4, 1.4, 0.04), BLACK),                    # front wing
              Part("cube", (rear - 0.35, 0, 1.05), (0.45, 1.0, 0.04), BLACK),                   # rear wing
              Part("cube", (rear - 0.42, 0, 1.17), (0.28, 1.0, 0.03), BLACK),                   # its flap
              Part("cube", (-0.3, 0, 0.8), (0.06, 0.5, 0.5), BLACK),                            # roll hoop
              Part("sphere", (-0.05, 0.0, 0.78), (0.27, 0.27, 0.27), WHITE)]                    # driver
    return parts


@dataclass
class Wheel:
    name: str                   # fl, fr, rl, rr
    x: float
    y: float
    steered: bool


def wheels(car) -> list[Wheel]:
    lf, lr = car.cg_to_front, car.wheelbase - car.cg_to_front
    return [Wheel("fl", lf, car.track_front / 2, True), Wheel("fr", lf, -car.track_front / 2, True),
            Wheel("rl", -lr, car.track_rear / 2, False), Wheel("rr", -lr, -car.track_rear / 2, False)]


def wheel_parts(car) -> list[Part]:
    """A wheel at its own origin (axle along y): tire and rim."""
    r = car.tire_radius
    q = tuple(quat("x", math.pi / 2))                    # a cylinder's axis is z: onto y
    return [Part("cylinder", (0, 0, 0), (2 * r, 2 * r, 0.2), BLACK, q),
            Part("cylinder", (0, 0, 0), (1.25 * r, 1.25 * r, 0.21), SILVER, q)]
