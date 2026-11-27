"""Courses from the coneLogger (github.com/KSU-MS/coneLogger): a VN200 carried around a
course, logging a point at each button press (PPS_*.CSV: latitude, longitude per cone, in
the order they were pressed) and its position continuously (GPS_*.CSV, the walk).

An autocross course is a series of gates, from an entry gate to an exit gate. The cones
come in logging order, which is the course's order:

  posts    presses less than `post_gap` m from the one before: one side of a gate (often
           several cones, and repeat presses of one)
  gates    two consecutive posts `gate_min`-`gate_max` m apart, the opening between their
           nearest cones; or one post that spans that wide on its own (the whole gate
           logged in one sweep, cone by cone across it)
  slaloms  runs of 3+ single cones: each becomes a gate with its other post `slalom_gap`
           m to the side, alternating (the side the first is passed on: whichever makes
           the smoother line from the gate before)
  the rest cones the car mustn't hit, nothing more

A "gate" that lies along the course (within `along_deg` of its direction) is a line of
edge cones, not a gate: it's kept as cones. Positions are the VN200's GNSS, logged as
float32 (~0.5 m steps) without RTK: about a metre either way.
"""
import json
import math
from pathlib import Path as FilePath

import numpy as np
import pandas as pd

EARTH_RADIUS = 6371008.8


def read_presses(pps_csv) -> np.ndarray:
    """(lon, lat) of each press, in order."""
    d = pd.read_csv(pps_csv, skipinitialspace=True)
    d = d[(d.latitude != 0) | (d.longitude != 0)]
    return d[["longitude", "latitude"]].to_numpy(float)


def _xy(lonlat, origin):
    lon0, lat0 = origin
    return np.column_stack([np.radians(lonlat[:, 0] - lon0) * EARTH_RADIUS * math.cos(math.radians(lat0)),
                            np.radians(lonlat[:, 1] - lat0) * EARTH_RADIUS])


def _lonlat(xy, origin):
    lon0, lat0 = origin
    return np.column_stack([lon0 + np.degrees(xy[:, 0] / (EARTH_RADIUS * math.cos(math.radians(lat0)))),
                            lat0 + np.degrees(xy[:, 1] / EARTH_RADIUS)])


def find_gates(P: np.ndarray, *, post_gap=2.5, gate_min=2.5, gate_max=8.0, along_deg=15.0,
               slalom_gap=4.0, slalom_min=3):
    """P: cone positions (m), in logging order. Returns (gates, slaloms, leftovers):
    gates as a list of (a, b, kind) in course order (kind "gate" or "slalom"; a, b the
    opening's two sides), and the indices of P that aren't part of any gate."""
    posts, cur = [], [0]
    for i in range(1, len(P)):
        if np.hypot(*(P[i] - P[i - 1])) < post_gap:
            cur.append(i)
        else:
            posts.append(cur)
            cur = [i]
    posts.append(cur)
    cent = np.array([P[g].mean(0) for g in posts])

    def span(g):
        q = P[g]
        d = np.hypot(*(q[:, None, :] - q[None, :, :]).transpose(2, 0, 1))
        i, j = np.unravel_index(np.argmax(d), d.shape)
        return d[i, j], q[i], q[j]

    items, k = [], 0          # ("gate", a, b, [cone idx]) or ("single", centroid, idx)
    while k < len(posts):
        w, e0, e1 = span(posts[k])
        if gate_min <= w <= gate_max:
            items.append(("gate", e0, e1, posts[k]))
            k += 1
        elif k + 1 < len(posts) and gate_min <= np.hypot(*(cent[k + 1] - cent[k])) <= gate_max:
            a, b = P[posts[k]], P[posts[k + 1]]
            d = np.hypot(*(a[:, None, :] - b[None, :, :]).transpose(2, 0, 1))
            i, j = np.unravel_index(np.argmin(d), d.shape)
            items.append(("gate", a[i], b[j], posts[k] + posts[k + 1]))
            k += 2
        else:
            items.append(("single", cent[k], posts[k]))
            k += 1

    # edge lines: "gates" along the course (direction from the gates either side)
    mids = [((it[1] + it[2]) / 2 if it[0] == "gate" else it[1]) for it in items]
    for n, it in enumerate(items):
        if it[0] != "gate":
            continue
        prev, nxt = mids[max(n - 1, 0)], mids[min(n + 1, len(mids) - 1)]
        travel = nxt - prev
        g = it[2] - it[1]
        cos = abs(travel @ g) / max(np.linalg.norm(travel) * np.linalg.norm(g), 1e-9)
        if math.degrees(math.acos(min(cos, 1.0))) < along_deg:
            items[n] = ("edge", None, None, it[3])

    gates, leftovers = [], []
    n = 0
    while n < len(items):
        it = items[n]
        if it[0] == "gate":
            gates.append((it[1], it[2], "gate"))
            n += 1
            continue
        run = []
        while n < len(items) and items[n][0] == "single" and len(items[n][2]) == 1:
            run.append(items[n])
            n += 1
        if len(run) >= slalom_min:
            gates.extend(_slalom([r[1] for r in run], gates[-1] if gates else None, slalom_gap))
            continue
        for r in run or [items[n]]:
            leftovers.extend(r[2] if r[0] == "single" else r[3])
        if not run:
            n += 1
    return gates, leftovers


def _slalom(cones, before, gap):
    """Alternating gates beside a line of cones, starting on the smoother side."""
    C = np.asarray(cones)
    t = C[-1] - C[0]
    t /= np.linalg.norm(t)
    nrm = np.array([-t[1], t[0]])
    best = None
    for first in (1, -1):
        sides = [first * (-1) ** i for i in range(len(C))]
        mids = [c + s * gap / 2 * nrm for c, s in zip(C, sides)]
        pts = ([((before[0] + before[1]) / 2)] if before else []) + mids
        length = sum(np.hypot(*(b - a)) for a, b in zip(pts, pts[1:]))
        if best is None or length < best[0]:
            best = (length, sides)
    return [(c, c + s * gap * nrm, "slalom") for c, s in zip(C, best[1])]


def build(pps_csv, out_geojson, name: str, *, skip=(), kind: str = "autocross", source: str = "", **kw) -> dict:
    """Read a coneLogger PPS file, find its gates, write the course GeoJSON (gates as
    LineStrings in course order, every cone as a Point). skip: press numbers (0-based,
    in the file) that aren't course cones. Returns a summary."""
    lonlat = read_presses(pps_csv)
    keep = np.array([i for i in range(len(lonlat)) if i not in set(skip)])
    lonlat = lonlat[keep]
    origin = tuple(lonlat.mean(0))
    P = _xy(lonlat, origin)
    gates, leftovers = find_gates(P, **kw)
    feats = []
    for n, (a, b, k) in enumerate(gates):
        ll = _lonlat(np.array([a, b]), origin)
        feats.append({"type": "Feature", "properties": {"layer": "gate", "order": n, "kind": k},
                      "geometry": {"type": "LineString", "coordinates": ll.round(9).tolist()}})
    for i, c in enumerate(lonlat):
        feats.append({"type": "Feature", "properties": {"layer": "cone", "press": int(keep[i])},
                      "geometry": {"type": "Point", "coordinates": c.round(9).tolist()}})
    widths = [np.hypot(*(b - a)) for a, b, k in gates if k == "gate"]
    gj = {"type": "FeatureCollection",
          "properties": {"name": name, "kind": kind, "closed": False, "source": source,
                         "gates": len(gates), "slalom_gates": sum(k == "slalom" for _, _, k in gates)},
          "features": feats}
    FilePath(out_geojson).write_text(json.dumps(gj, indent=0))
    return {"gates": len(gates), "slalom gates": sum(k == "slalom" for _, _, k in gates),
            "cones": len(lonlat), "cones in no gate": len(leftovers),
            "opening m (min / median / max)": (round(min(widths), 1), round(float(np.median(widths)), 1),
                                               round(max(widths), 1))}
