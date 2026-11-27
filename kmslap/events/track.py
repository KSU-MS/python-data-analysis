"""Real tracks: centerlines as driver.Path loops, and Tracks (a centerline with the free
width to each side, and the cones if the course is coned).

tracks/michigan_2026_ev_endurance.geojson is the official FSAE Electric 2026
endurance line at Michigan International Speedway, traced and georeferenced from the
fsaeonline map (from the fsae-ev-endurance-map repo, kithek.github.io/fsae-ev-endurance-map).
A line only: its width is a guess (TRACK_WIDTH).

tracks/michigan_ac18_cones.geojson is a coned course at the same site, the Assetto
Corsa fsae_michigan18 track's (from the same repo): 179 cones, mostly gates ~4.8 m
wide every ~25 m, hairpins ~3.5-4 m wide, and a hand-ordered route through them (the lane centres at the
gates and through the two hairpins) that gives the driving order and direction. It is
not the 2026 layout.
"""
import json
from dataclasses import dataclass
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


# FSAE endurance tracks are at least ~4.5 m wide (a guess at this one's width: the map
# trace is a centerline only)
TRACK_WIDTH = 4.5           # m
CAR_WIDTH = 1.42            # m, track + tire width
EDGE_MARGIN = 0.15          # m kept from the edge (cones)
CONE_RADIUS = 0.15          # m, at the base (a 0.3 m FSAE cone)


@dataclass
class Track:
    """A course: its centerline, and the free width from it to each edge at each
    centerline point (to the cone centres on a coned course)."""
    name: str
    center: Path
    w_left: np.ndarray
    w_right: np.ndarray
    cones: np.ndarray | None = None      # (n, 2) cone positions, the centerline's frame
    origin: tuple | None = None          # (lon, lat) of the frame's (0, 0): x east, y north
    kind: str = "endurance"              # or "autocross": open, entry gate to exit gate, no laps
    gates: np.ndarray | None = None      # (n, 2, 2) the gates' two sides, course order (autocross)

    @property
    def closed(self) -> bool:
        return self.center.closed

    def limits(self, car_width: float = CAR_WIDTH, margin: float = EDGE_MARGIN):
        """(lo, hi): the car centre's lateral offset bounds (+ left) at each centerline
        point, keeping its side `margin` from the edge (from the cones' sides)."""
        pad = car_width / 2 + margin + (CONE_RADIUS if self.cones is not None else 0.0)
        return -(self.w_right - pad), self.w_left - pad

    def edges(self):
        """(left x, left y, right x, right y)."""
        nx, ny = normals(self.center)
        c = self.center
        return c.x + self.w_left * nx, c.y + self.w_left * ny, c.x - self.w_right * nx, c.y - self.w_right * ny

    def resampled(self, step: float) -> "Track":
        ref = _resample(self.center, step)
        period = self.center.length if self.closed else None
        at = lambda w: np.interp(ref.s, self.center.s, w, period=period)
        return Track(self.name, ref, at(self.w_left), at(self.w_right), self.cones, self.origin, self.kind,
                     self.gates)

    def lonlat(self, x, y):
        """(lon, lat) of points in the track's frame (the frame load_line / cone_track
        projected from)."""
        lon0, lat0 = self.origin
        lat = lat0 + np.degrees(np.asarray(y) / EARTH_RADIUS)
        lon = lon0 + np.degrees(np.asarray(x) / (EARTH_RADIUS * np.cos(np.radians(lat0))))
        return lon, lat


def constant_width(center: Path, width: float = TRACK_WIDTH, name: str = "") -> Track:
    n = len(center.s)
    return Track(name, center, np.full(n, width / 2), np.full(n, width / 2))


def michigan_2026(width: float = TRACK_WIDTH) -> Track:
    """The 2026 EV endurance line with a guessed constant width."""
    t = constant_width(michigan_2026_endurance(), width, "Michigan 2026 EV endurance")
    feature = json.loads((TRACKS / "michigan_2026_ev_endurance.geojson").read_text())["features"][0]
    t.origin = tuple(feature["geometry"]["coordinates"][0][:2])    # load_line's origin
    return t


def michigan_ac18() -> Track:
    """The Assetto Corsa fsae_michigan18 coned course (see the module docstring)."""
    return cone_track(TRACKS / "michigan_ac18_cones.geojson")


def michigan2025_autox() -> Track:
    """FSAE Michigan 2025 autocross, from the coneLogger's cones (events/conelogger.py):
    a series of gates from an entry gate to an exit gate.

    WIP, not in TRACK_NAMES: the minimum-time solvers and the drives assume a loop, and
    the course still has a kink near the exit gates (s ~915 m) and a few gates the
    centerline barely clears (s ~424, ~602 m)."""
    return gate_track(TRACKS / "michigan2025_autox.geojson")


def gate_track(geojson, step: float = 0.5, smoothing: float = 0.3) -> Track:
    """An open course from its gates (GeoJSON LineStrings, layer "gate", in course order)
    and cones (Points). The centerline runs through the gates' middles (a smoothing spline,
    so within ~smoothing m of them); its width at each gate is the gate's opening, and
    between gates the edges run straight from one gate's sides to the next's."""
    feats = json.loads(FilePath(geojson).read_text())
    gates = sorted((f for f in feats["features"] if f["properties"].get("layer") == "gate"),
                   key=lambda f: f["properties"]["order"])
    cones_ll = np.array([f["geometry"]["coordinates"][:2] for f in feats["features"]
                         if f["geometry"]["type"] == "Point"], float)
    origin = tuple(cones_ll.mean(0))
    G = np.array([np.column_stack(_project(np.array(g["geometry"]["coordinates"], float), origin)) for g in gates])
    mids = G.mean(axis=1)
    tck, _ = splprep([mids[:, 0], mids[:, 1]], s=len(mids) * smoothing ** 2, k=3)
    dense = Path(*splev(np.linspace(0, 1, 50 * len(mids)), tck), closed=False)
    center = _resample(dense, step)
    nx, ny = normals(center)
    gi = _nearest(center, mids[:, 0], mids[:, 1])
    order = np.argsort(gi)
    gi, G = gi[order], G[order]
    off = (G[..., 0] - center.x[gi, None]) * nx[gi, None] + (G[..., 1] - center.y[gi, None]) * ny[gi, None]
    left, right = off.max(axis=1), -off.min(axis=1)       # each side's post, from the centerline
    wl = np.interp(center.s, center.s[gi], left)
    wr = np.interp(center.s, center.s[gi], right)
    cx, cy = _project(cones_ll, origin)
    props = feats.get("properties", {})
    return Track(props.get("name", FilePath(geojson).stem), center, wl, wr, np.column_stack([cx, cy]),
                 origin, props.get("kind", "autocross"), G)


TRACK_NAMES = {"michigan2026": michigan_2026, "ac18": michigan_ac18}
TRACK_KINDS = {"michigan2026": "endurance", "ac18": "endurance"}    # michigan2025_autox: "autocross", WIP


def _project(lonlat: np.ndarray, origin) -> tuple[np.ndarray, np.ndarray]:
    lon0, lat0 = origin
    return (np.radians(lonlat[:, 0] - lon0) * EARTH_RADIUS * np.cos(np.radians(lat0)),
            np.radians(lonlat[:, 1] - lat0) * EARTH_RADIUS)


def _ray_hits(px, py, dx, dy, ex, ey, max_dist):
    """Distance along each ray (p + t d, t > 0) to the nearest crossing of the closed
    polyline e, or nan within max_dist."""
    ax, ay = ex, ey
    bx, by = np.roll(ex, -1), np.roll(ey, -1)
    sx, sy = (bx - ax)[None], (by - ay)[None]
    qx, qy = ax[None] - px[:, None], ay[None] - py[:, None]
    den = dx[:, None] * sy - dy[:, None] * sx
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (qx * sy - qy * sx) / den
        u = (qx * dy[:, None] - qy * dx[:, None]) / den
    t = np.where((u >= 0) & (u <= 1) & (t > 0) & (t < max_dist), t, np.inf)
    t = t.min(axis=1)
    return np.where(np.isfinite(t), t, np.nan)


def cone_track(geojson, smoothing: float = 0.5, step: float = 0.5, max_offset: float = 6.0) -> Track:
    """A Track from cones and a route through them (a GeoJSON of Point cones and one
    LineString, the driving order: roughly the lane centre, closed).

    Each cone goes to the left or right edge by which side of the route it's on (cones
    more than max_offset m from the route are kept as cones but not edges); each edge
    is its cones joined in route order, straight between cones as the lines between
    cones are walked. The centerline is midway between the edges, found along the
    route's normals, then smoothed, and the widths measured again from it.
    """
    from scipy.ndimage import uniform_filter1d

    feats = json.loads(FilePath(geojson).read_text())
    route = next(f for f in feats["features"] if f["geometry"]["type"] == "LineString")
    route_ll = np.array(route["geometry"]["coordinates"], float)
    # cones with a role (a start chute) aren't on the lap
    cones_ll = np.array([f["geometry"]["coordinates"][:2] for f in feats["features"]
                         if f["geometry"]["type"] == "Point" and "role" not in f["properties"]], float)
    origin = route_ll[0]
    rx, ry = _project(route_ll, origin)
    cx, cy = _project(cones_ll, origin)
    tck, _ = splprep([np.r_[rx, rx[0]], np.r_[ry, ry[0]]], s=len(rx) * smoothing ** 2, per=1)
    guide = _resample(Path(*splev(np.linspace(0, 1, 40 * len(rx), endpoint=False), tck), closed=True), step)

    idx = _nearest(guide, cx, cy)
    nx, ny = normals(guide)
    off = (cx - guide.x[idx]) * nx[idx] + (cy - guide.y[idx]) * ny[idx]
    edge = np.abs(off) < max_offset
    sides = []
    for sel in (edge & (off > 0), edge & (off < 0)):
        order = np.argsort(guide.s[idx[sel]])
        sides.append((cx[sel][order], cy[sel][order]))

    def widths(c: Path):
        nx, ny = normals(c)
        wl = _ray_hits(c.x, c.y, nx, ny, *sides[0], 3 * max_offset)
        wr = _ray_hits(c.x, c.y, -nx, -ny, *sides[1], 3 * max_offset)
        fill = lambda w: np.interp(c.s, c.s[~np.isnan(w)], w[~np.isnan(w)], period=c.length)
        return fill(wl), fill(wr)

    wl, wr = widths(guide)
    shift = (wl - wr) / 2
    k = max(int(4.0 / step), 1)          # smooth the midline over ~4 m
    shift = uniform_filter1d(shift, k, mode="wrap")
    center = _resample(Path(guide.x + shift * nx, guide.y + shift * ny, closed=True), step)
    wl, wr = widths(center)
    name = feats.get("properties", {}).get("name", FilePath(geojson).stem)
    return Track(name, center, wl, wr, np.column_stack([cx, cy]), tuple(origin[:2]))


def normals(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Unit left normals of a path."""
    return -np.sin(path.heading), np.cos(path.heading)


def edges(path, width: float = TRACK_WIDTH):
    """(left x, left y, right x, right y) of a track of this width around its centerline
    (or of a Track, its own widths)."""
    if isinstance(path, Track):
        return path.edges()
    nx, ny = normals(path)
    h = width / 2
    return path.x + h * nx, path.y + h * ny, path.x - h * nx, path.y - h * ny


def racing_line(path, width: float = TRACK_WIDTH, car_width: float = CAR_WIDTH,
                margin: float = EDGE_MARGIN, step: float = 1.0, iterations: int = 4) -> Path:
    """Minimum-curvature line within the track: the closed line whose summed squared
    curvature is smallest with the car's side at least `margin` inside each edge.
    path: a Track, or a centerline (then the track is `width` wide).

    Curvature is approximated by the line's second differences on near-uniform spacing,
    so each solve is a bounded linear least squares in the lateral offsets; the line is
    re-sampled and re-solved a few times, the bounds always measured from the original
    centerline. The car isn't modelled here: this is the geometric line, which gives
    most of the gain of a racing line; a minimum-time line (speed and line together)
    would come from optimal control.
    """
    from scipy import sparse
    from scipy.optimize import lsq_linear

    import hashlib
    track = path if isinstance(path, Track) else constant_width(path, width)
    path = track.center
    key = hashlib.sha1(np.concatenate([path.x, path.y, track.w_left, track.w_right,
                                       [car_width, margin, step, iterations]]).tobytes()).hexdigest()[:16]
    cache = dirs.BUILD / "racing_line" / f"{key}.npz"
    if cache.exists():
        z = np.load(cache)
        return Path(z["x"], z["y"], closed=track.closed)

    track = track.resampled(step)
    center = track.center
    lo, hi = track.limits(car_width, margin)
    ref = center
    for _ in range(iterations):
        n = len(ref.x)
        nx, ny = normals(ref)
        # how far this reference already is from the centerline (signed, along its normal)
        idx = _nearest(center, ref.x, ref.y)
        off = (ref.x - center.x[idx]) * nx + (ref.y - center.y[idx]) * ny
        D = _second_differences(n, ref.closed)
        A = sparse.vstack([D @ sparse.diags(nx), D @ sparse.diags(ny)]).tocsr()
        b = -np.concatenate([D @ ref.x, D @ ref.y])
        res = lsq_linear(A, b, bounds=(lo[idx] - off, hi[idx] - off), method="trf", lsmr_tol="auto")
        ref = _resample(Path(ref.x + res.x * nx, ref.y + res.x * ny, closed=track.closed), step)
    # hold the final line inside the limit (the passes measure offsets approximately)
    idx = _nearest(center, ref.x, ref.y)
    cnx, cny = normals(center)
    off = (ref.x - center.x[idx]) * cnx[idx] + (ref.y - center.y[idx]) * cny[idx]
    fix = np.clip(off, lo[idx], hi[idx]) - off
    line = _resample(Path(ref.x + fix * cnx[idx], ref.y + fix * cny[idx], closed=track.closed),
                     path.s[1] - path.s[0])        # the centerline's finer spacing, for driving
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, x=line.x, y=line.y)
    return line


def mincurv_offsets(ref: Path, lo, hi=None) -> np.ndarray:
    """Lateral offsets (along ref's normals) of the minimum-curvature line around a
    uniformly spaced closed ref, within [lo, hi] (per point; a scalar lo alone means
    +-lo): one bounded least-squares pass (see racing_line)."""
    from scipy import sparse
    from scipy.optimize import lsq_linear

    n = len(ref.x)
    nx, ny = normals(ref)
    D = _second_differences(n, ref.closed)
    A = sparse.vstack([D @ sparse.diags(nx), D @ sparse.diags(ny)]).tocsr()
    b = -np.concatenate([D @ ref.x, D @ ref.y])
    if hi is None:
        lo, hi = -lo, lo
    lo, hi = np.broadcast_to(lo, n).astype(float), np.broadcast_to(hi, n).astype(float)
    return lsq_linear(A, b, bounds=(lo, hi), method="trf", lsmr_tol="auto").x


def _second_differences(n: int, closed: bool):
    """x[i-1] - 2 x[i] + x[i+1] for each point (wrapping on a loop; interior only on an
    open path, whose ends have no curvature to give)."""
    from scipy import sparse
    D = sparse.diags([np.ones(n), -2 * np.ones(n), np.ones(n)], [-1, 0, 1], shape=(n, n), format="lil")
    if closed:
        D[0, n - 1] = 1
        D[n - 1, 0] = 1
        return D.tocsr()
    return D.tocsr()[1:-1]


def _resample(path: Path, step: float) -> Path:
    """A path at equal spacing. Closed: a whole number of segments, so there's no short
    closing segment (which would spike the curvature at the seam). Open: both ends kept."""
    if not path.closed:
        n = max(int(round(path.length / step)), 2) + 1
        s = np.linspace(0, path.length, n)
        return Path(np.interp(s, path.s, path.x), np.interp(s, path.s, path.y), closed=False)
    n = max(int(round(path.length / step)), 3)
    s = np.linspace(0, path.length, n, endpoint=False)
    return Path(np.interp(s, path.s, path.x, period=path.length),
                np.interp(s, path.s, path.y, period=path.length), closed=True)


def _nearest(path: Path, x, y) -> np.ndarray:
    from scipy.spatial import cKDTree
    return cKDTree(np.column_stack([path.x, path.y])).query(np.column_stack([x, y]))[1]


def lateral_offset(center: Path, line: Path) -> np.ndarray:
    """Signed distance of each point of `line` from the centerline (+ left)."""
    idx = _nearest(center, line.x, line.y)
    nx, ny = normals(center)
    return (line.x - center.x[idx]) * nx[idx] + (line.y - center.y[idx]) * ny[idx]


@dataclass
class Penalties:
    cones: int                 # cones down or out (each once)
    off_course: int            # times all four wheels left the track
    cone_s: list               # centerline s of each cone hit
    off_course_s: list         # where each off-course began
    seconds: float             # FSAE endurance: 2 s a cone, 20 s an off-course

    def __str__(self):
        return f"{self.cones} cones, {self.off_course} off course (+{self.seconds:g} s)"


def cone_hits(track: Track, X, Y, psi, car=None, overhang: float = 0.45) -> list[tuple[int, int]]:
    """(cone index, first sample touching it) for each cone the trajectory hits: within
    its base radius of the car's footprint (wheelbase + overhang at each end, CAR_WIDTH
    wide). None on a track without cones: an empty list."""
    from kmslap.car.params import Car

    car = car or Car()
    if track.cones is None:
        return []
    X, Y, psi = (np.asarray(a, float) for a in (X, Y, psi))
    c, s_ = np.cos(psi), np.sin(psi)
    lf, lr = car.cg_to_front, car.wheelbase - car.cg_to_front
    hits = []
    for i, (cx, cy) in enumerate(track.cones):
        near = np.flatnonzero(np.hypot(X - cx, Y - cy) < lf + overhang + CAR_WIDTH)
        if not len(near):
            continue
        dx, dy = cx - X[near], cy - Y[near]
        u, v = c[near] * dx + s_[near] * dy, -s_[near] * dx + c[near] * dy     # car frame
        du = np.maximum(np.maximum(u - (lf + overhang), -(lr + overhang) - u), 0)
        dv = np.maximum(np.abs(v) - CAR_WIDTH / 2, 0)
        touch = np.flatnonzero(np.hypot(du, dv) < CONE_RADIUS)
        if len(touch):
            hits.append((i, int(near[touch[0]])))
    return hits


def off_course(track: Track, X, Y, psi, car=None) -> np.ndarray:
    """The samples where the car goes off course: all four tire contact points outside
    the edges at once (each excursion's first sample)."""
    from kmslap.car.params import Car

    car = car or Car()
    X, Y, psi = (np.asarray(a, float) for a in (X, Y, psi))
    c, s_ = np.cos(psi), np.sin(psi)
    lf, lr = car.cg_to_front, car.wheelbase - car.cg_to_front
    out = np.ones(len(X), bool)
    cn = track.center
    nx, ny = normals(cn)
    for a, b in [(lf, car.track_front / 2), (lf, -car.track_front / 2), (-lr, car.track_rear / 2), (-lr, -car.track_rear / 2)]:
        wx, wy = X + a * c - b * s_, Y + a * s_ + b * c
        i = _nearest(cn, wx, wy)
        off = (wx - cn.x[i]) * nx[i] + (wy - cn.y[i]) * ny[i]
        out &= (off > track.w_left[i]) | (off < -track.w_right[i])
    return np.flatnonzero(np.diff(np.r_[False, out].astype(int)) == 1)


def penalties(track: Track, X, Y, psi, car=None, overhang: float = 0.45) -> Penalties:
    """FSAE penalties of a driven trajectory (world X, Y, heading psi, in the track's
    frame): a cone is hit when it's within its base radius of the car's footprint
    (cone_hits); off course is all four tire contact points outside the edges at once
    (off_course). Cones are only checked on a coned track."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    cn = track.center
    hits = cone_hits(track, X, Y, psi, car, overhang)
    cone_s = [float(cn.s[_nearest(cn, [track.cones[i][0]], [track.cones[i][1]])[0]]) for i, _ in hits]
    cone_hits_n = len(hits)
    starts = off_course(track, X, Y, psi, car)
    off_s = [float(cn.s[_nearest(cn, [X[k]], [Y[k]])[0]]) for k in starts]
    return Penalties(cone_hits_n, len(starts), cone_s, off_s, 2.0 * cone_hits_n + 20.0 * len(starts))
