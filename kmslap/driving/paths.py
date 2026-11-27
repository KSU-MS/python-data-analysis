"""The lines drivers follow: Path (points, arc length, heading, curvature; open or
a closed loop), the skidpad, and a speed profile from constant acceleration limits
(models.envelope.speed_profile uses the car's own grip envelope instead)."""
import math
from dataclasses import dataclass, field

import numpy as np

SKIDPAD_RADIUS = 9.125   # m, FSAE skidpad driven line (15.25 m inner circle + 3 m lane)


@dataclass
class Path:
    """A driven line: points (x, y) with arc length s, heading (rad) and curvature k
    (1/m, + left)."""
    x: np.ndarray
    y: np.ndarray
    closed: bool = False
    s: np.ndarray = field(init=False)
    heading: np.ndarray = field(init=False)
    k: np.ndarray = field(init=False)

    def __post_init__(self):
        self.x, self.y = np.asarray(self.x, float), np.asarray(self.y, float)
        if self.closed:
            # a loop: the last point connects back to the first (don't repeat it)
            if np.hypot(self.x[-1] - self.x[0], self.y[-1] - self.y[0]) < 1e-6:
                self.x, self.y = self.x[:-1], self.y[:-1]
            dx = 0.5 * (np.roll(self.x, -1) - np.roll(self.x, 1))
            dy = 0.5 * (np.roll(self.y, -1) - np.roll(self.y, 1))
            ds = np.hypot(np.diff(self.x, append=self.x[0]), np.diff(self.y, append=self.y[0]))
            self._length = float(ds.sum())
            self.s = np.concatenate([[0.0], np.cumsum(ds[:-1])])
            self.heading = np.unwrap(np.arctan2(dy, dx))
            dh = (np.roll(self.heading, -1) - np.roll(self.heading, 1) + np.pi) % (2 * np.pi) - np.pi
            self.k = 0.5 * dh / np.maximum(np.hypot(dx, dy), 1e-9)
        else:
            ds = np.hypot(np.diff(self.x), np.diff(self.y))
            self.s = np.concatenate([[0.0], np.cumsum(ds)])
            self._length = float(self.s[-1])
            self.heading = np.unwrap(np.arctan2(np.gradient(self.y), np.gradient(self.x)))
            self.k = np.gradient(self.heading) / np.maximum(np.gradient(self.s), 1e-9)

    @property
    def length(self) -> float:
        return self._length

    def ahead(self, i: int, dist: float) -> int:
        """Index of the point `dist` metres past point i (wrapping on a loop)."""
        if self.closed:
            rem = (self.s[i] + dist) % self._length
            return int(np.searchsorted(self.s, rem)) % len(self.s)
        return min(int(np.searchsorted(self.s, self.s[i] + dist)), len(self.s) - 1)

    def span(self, i: int, j: int) -> np.ndarray:
        """Indices from i to j inclusive, going forward (wrapping on a loop)."""
        if j >= i:
            return np.arange(i, j + 1)
        return np.concatenate([np.arange(i, len(self.s)), np.arange(0, j + 1)])

    @staticmethod
    def from_segments(segments, start=(0.0, 0.0), heading=0.0, step=0.25, closed=False):
        """segments: ("straight", length) or ("arc", radius, degrees; + left)."""
        x, y, h = [start[0]], [start[1]], heading
        for seg in segments:
            if seg[0] == "straight":
                n = max(int(seg[1] / step), 1)
                for _ in range(n):
                    x.append(x[-1] + seg[1] / n * math.cos(h))
                    y.append(y[-1] + seg[1] / n * math.sin(h))
            else:
                radius, sweep = seg[1], math.radians(seg[2])
                n = max(int(abs(sweep) * radius / step), 1)
                dh = sweep / n
                for _ in range(n):
                    h += dh
                    chord = 2 * radius * math.sin(abs(dh) / 2)
                    x.append(x[-1] + chord * math.cos(h - dh / 2))
                    y.append(y[-1] + chord * math.sin(h - dh / 2))
        return Path(np.array(x), np.array(y), closed=closed)


def skidpad(radius: float = SKIDPAD_RADIUS, laps: int = 2, direction: int = 1) -> Path:
    """Straight entry then `laps` circles (left for direction=1), starting tangent at
    the origin heading +x, like the skidpad's timed laps without the figure-eight."""
    return Path.from_segments([("straight", 15.0), ("arc", radius, 360.0 * laps * direction),
                               ("straight", 15.0)])


def speed_profile(path: Path, a_lat: float, a_accel: float, a_brake: float,
                  v_max: float = 35.0, v_start: float | None = None) -> np.ndarray:
    """Fastest speed along the path within lateral / longitudinal acceleration
    limits (forward-backward pass). A quasi-steady target for the driver, not a
    prediction: the car may not manage it."""
    v = np.minimum(v_max, np.sqrt(a_lat / np.maximum(np.abs(path.k), 1e-6)))
    ds = np.diff(path.s)
    if v_start is not None:
        v[0] = min(v[0], v_start)
    for i in range(1, len(v)):
        v[i] = min(v[i], math.sqrt(v[i - 1] ** 2 + 2 * a_accel * ds[i - 1]))
    for i in range(len(v) - 2, -1, -1):
        v[i] = min(v[i], math.sqrt(v[i + 1] ** 2 + 2 * a_brake * ds[i]))
    return v
