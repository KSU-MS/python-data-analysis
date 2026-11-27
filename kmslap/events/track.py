"""Real track lines, as driver.Path loops.

tracks/michigan_2026_ev_endurance.geojson is the official FSAE Electric 2026
endurance line at Michigan International Speedway, traced and georeferenced from the
fsaeonline map (from the fsae-ev-endurance-map repo, kithek.github.io/fsae-ev-endurance-map).
"""
import json
from pathlib import Path as FilePath

import numpy as np
from scipy.interpolate import splev, splprep

from kmslap.driving.paths import Path
from kmslap import dirs

TRACKS = dirs.TRACKS
EARTH_RADIUS = 6371008.8   # m


def load_line(geojson, smoothing: float = 0.3, step: float = 0.5) -> Path:
    """A closed GeoJSON LineString (lon, lat) as a Path loop in metres (x east, y
    north, origin at the first point), through a periodic smoothing spline.

    smoothing: m, roughly the tracing error to smooth out. Hand-traced points make
    curvature spiky; 0.3 m keeps the line within ~0.6 m of the trace.
    step: m between resampled points.
    """
    feature = json.loads(FilePath(geojson).read_text())["features"][0]
    lonlat = np.array(feature["geometry"]["coordinates"], float)
    lon0, lat0 = lonlat[0]
    x = np.radians(lonlat[:, 0] - lon0) * EARTH_RADIUS * np.cos(np.radians(lat0))
    y = np.radians(lonlat[:, 1] - lat0) * EARTH_RADIUS
    keep = np.r_[True, np.hypot(np.diff(x), np.diff(y)) > 1e-6]
    x, y = x[keep], y[keep]
    if np.hypot(x[-1] - x[0], y[-1] - y[0]) < 1e-6:
        x, y = x[:-1], y[:-1]
    tck, _ = splprep([np.r_[x, x[0]], np.r_[y, y[0]]], s=len(x) * smoothing ** 2, per=1)
    dense = np.linspace(0, 1, 20 * len(x), endpoint=False)
    xs, ys = splev(dense, tck)
    length = np.hypot(np.diff(xs, append=xs[0]), np.diff(ys, append=ys[0])).sum()
    u = np.linspace(0, 1, int(length / step), endpoint=False)
    xs, ys = splev(u, tck)
    return Path(np.asarray(xs), np.asarray(ys), closed=True)


def michigan_2026_endurance() -> Path:
    """FSAE Michigan 2026 EV endurance: ~1.22 km a lap, ~18 laps for the 22 km event."""
    return load_line(TRACKS / "michigan_2026_ev_endurance.geojson")
