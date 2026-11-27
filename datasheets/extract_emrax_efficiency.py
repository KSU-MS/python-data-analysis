"""Digitize the EMRAX 228 HV CC efficiency map (emrax_228_technical_data_4.5.pdf, page 2)
into a grid: emrax_228hv_efficiency.csv (rpm, torque_nm, efficiency, source).

    python datasheets/extract_emrax_efficiency.py

Source image: emrax_228hv_cc_efficiency_map.png, extracted from the PDF with
`pdfimages -png -f 2 -l 2`. The map shows contour lines at 86, 90, 94, 95 and 96 %
(labels "86 - 90 %" etc. sit between them), 0-5000 rpm, 0-250 Nm.

Method:
  1. Calibrate pixels to rpm / Nm from the gridlines (every 500 rpm, every 50 Nm).
  2. Mask each contour by its line colour; drop the same-coloured labels (small
     connected blobs); close contours that run off the 5000 rpm edge along it; fill.
  3. Between two contours, interpolate efficiency by distance to each (in pixels,
     which are close to isotropic here); inside 96 %, rise to 96.5 % at the deepest
     point (the datasheet's 92-98 % range tops out higher, but the map doesn't show it).
  4. Outside the 86 % contour the map has no data: use a loss model fitted (non-
     negative) to the contour points, P_loss = d + a w + b w^2 + c T^2, efficiency
     = P / (P + P_loss),
     so efficiency falls off at low torque like a real motor's.
  5. Above 5000 rpm (the map's edge; the motor runs to 5500 / 6500 peak), hold the
     5000 rpm values.

The same map is used for generating (regen), as a symmetric-losses assumption.
Writes emrax_228hv_efficiency_check.png: the reconstruction's contours over the original.
"""
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage, optimize

HERE = Path(__file__).resolve().parent
SRC = HERE / "emrax_228hv_cc_efficiency_map.png"
OUT = HERE / "emrax_228hv_efficiency.csv"

# pixel calibration from the gridlines
X0, PX_PER_500RPM = 57.0, (650 - 57) / 9
Y0, PX_PER_50NM = 493.0, (493 - 74) / 5
CONTOURS = [  # (efficiency %, line colour), outermost first
    (86.0, (49, 133, 156)),
    (90.0, (96, 74, 123)),
    (94.0, (119, 147, 60)),
    (95.0, (192, 80, 77)),
    (96.0, (56, 93, 138)),
]
PEAK = 96.5


def to_px(rpm, torque):
    return X0 + rpm / 500 * PX_PER_500RPM, Y0 - torque / 50 * PX_PER_50NM


def to_data(px, py):
    return (px - X0) / PX_PER_500RPM * 500, (Y0 - py) / PX_PER_50NM * 50


def contour_masks(im):
    h, w, _ = im.shape
    x_edge = int(round(to_px(5000, 0)[0]))
    lines, regions = [], []
    for _, colour in CONTOURS:
        mask = (np.abs(im - np.array(colour)).max(-1) < 40)
        labels, n = ndimage.label(mask, structure=np.ones((3, 3)))
        sizes = ndimage.find_objects(labels)
        keep = np.zeros_like(mask)
        for k, sl in enumerate(sizes, start=1):
            extent = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start)
            if extent > 60:                                  # a line, not a label glyph
                keep |= labels == k
        line = ndimage.binary_dilation(keep, iterations=1)
        # close along the 5000 rpm edge where the contour runs off the map
        ys = np.where(line[:, x_edge - 4:x_edge + 1].any(1))[0]
        closed = line.copy()
        if len(ys):
            closed[ys.min():ys.max() + 1, x_edge - 1:x_edge + 1] = True
        lines.append(line)
        regions.append(ndimage.binary_fill_holes(closed))
    return lines, regions


def fit_loss_model(lines):
    """Non-negative least squares P_loss = d + a w + b w^2 + c T^2 on the contour
    points, leaving out where a contour is really the map's edge (5000 rpm, the
    ~243 Nm top)."""
    rows, rhs = [], []
    for (eff, _), line in zip(CONTOURS, lines):
        py, px = np.nonzero(line)
        rpm, torque = to_data(px, py)
        ok = (rpm > 200) & (rpm < 4900) & (torque > 5) & (torque < 238)
        w = rpm[ok] * 2 * np.pi / 60
        p_out = w * torque[ok]
        p_loss = p_out * (100 / eff - 1)
        rows.append(np.column_stack([np.ones_like(w), w, w ** 2, torque[ok] ** 2]))
        rhs.append(p_loss)
    A, y = np.vstack(rows), np.concatenate(rhs)
    scale = A.max(0)
    coef = optimize.nnls(A / scale, y)[0] / scale
    pred = A @ coef
    return coef, float(np.sqrt(np.mean((pred - y) ** 2) / np.mean(y ** 2)))


def loss_model_eff(coef, rpm, torque):
    w = rpm * 2 * np.pi / 60
    p = w * torque
    loss = coef[0] + coef[1] * w + coef[2] * w ** 2 + coef[3] * torque ** 2
    return 100 * p / np.maximum(p + loss, 1e-9)


def main():
    im = np.array(Image.open(SRC).convert("RGB")).astype(int)
    lines, regions = contour_masks(im)
    dist = [ndimage.distance_transform_edt(~line) for line in lines]
    coef, rel_rms = fit_loss_model(lines)

    rpms = np.arange(0.0, 6501.0, 50.0)
    torques = np.arange(0.0, 250.01, 2.5)
    R, T = np.meshgrid(rpms, torques, indexing="ij")
    px, py = to_px(np.minimum(R, 5000.0), T)
    ix = np.clip(np.round(px).astype(int), 0, im.shape[1] - 1)
    iy = np.clip(np.round(py).astype(int), 0, im.shape[0] - 1)
    depth = sum(reg[iy, ix].astype(int) for reg in regions)   # contours containing the point
    eff = np.full(R.shape, np.nan)
    source = np.full(R.shape, "digitized", dtype=object)
    inner_max = dist[-1][regions[-1]].max()
    for k in range(1, len(CONTOURS) + 1):
        sel = depth == k
        lo = CONTOURS[k - 1][0]
        d_out = dist[k - 1][iy[sel], ix[sel]]
        if k < len(CONTOURS):
            hi = CONTOURS[k][0]
            d_in = dist[k][iy[sel], ix[sel]]
            eff[sel] = lo + (hi - lo) * d_out / np.maximum(d_out + d_in, 1e-9)
        else:
            eff[sel] = lo + (PEAK - lo) * np.minimum(d_out / inner_max, 1.0)
    out = depth == 0
    eff[out] = np.minimum(loss_model_eff(coef, R[out], T[out]), CONTOURS[0][0])
    source[out] = "loss_model"
    eff = np.clip(np.nan_to_num(eff, nan=0.0), 0.0, 100.0)

    with OUT.open("w") as f:
        f.write("# EMRAX 228 HV CC efficiency (motor, shaft vs electrical), digitized from\n")
        f.write("# emrax_228_technical_data_4.5.pdf p.2 by extract_emrax_efficiency.py.\n")
        f.write(f"# Outside the 86% contour: loss model P_loss = {coef[0]:.1f} + {coef[1]:.3f} w"
                f" + {coef[2]:.5f} w^2 + {coef[3]:.4f} T^2 W (fit rel. RMS {rel_rms:.2f}).\n")
        f.write("# Above 5000 rpm: the 5000 rpm values held. Used for generating too (assumption).\n")
        f.write("rpm,torque_nm,efficiency,source\n")
        for i, r in enumerate(rpms):
            for j, t in enumerate(torques):
                f.write(f"{r:.0f},{t:.1f},{eff[i, j] / 100:.4f},{source[i, j]}\n")
    print(f"wrote {OUT.name}: {eff.size} points; loss model rel. RMS {rel_rms:.3f}, coef {coef}")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 6.5))
    x_max, y_min = to_px(5000, 0)
    x_min, y_max = to_px(0, 250)
    ax.imshow(im.astype(np.uint8), extent=(*to_data(0, im.shape[0])[:1], *to_data(im.shape[1], 0)[:1],
                                            to_data(0, im.shape[0])[1], to_data(0, 0)[1]), aspect="auto")
    cs = ax.contour(R, T, eff, levels=[80, 86, 90, 94, 95, 96], colors="k", linewidths=1.2, linestyles="--")
    ax.clabel(cs, fmt="%g%%", fontsize=8)
    ax.set(xlim=(0, 5000), ylim=(0, 250), xlabel="rpm", ylabel="Nm",
           title="Digitized map (black dashed) over the datasheet's")
    fig.tight_layout()
    fig.savefig(HERE / "emrax_228hv_efficiency_check.png", dpi=110)


if __name__ == "__main__":
    main()
