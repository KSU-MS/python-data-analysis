"""Longitudinal tire models. Each exposes, per wheel:

    fx(kappa, fz) -> longitudinal force (N)
    peak(fz)      -> (slip ratio at peak Fx, peak mu)
"""
import bisect
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

_CURVE_CSV = Path(__file__).resolve().parent.parent / "tire" / "mu-vs-slip.csv"


class CurveTire:
    """Load-independent mu(slip) curve: the shape of tire/mu-vs-slip.csv scaled to `mu_peak`.

    Negative slip (braking) mirrors positive.
    """

    def __init__(self, mu_peak: float = 1.4, csv=_CURVE_CSV):
        data = pd.read_csv(csv).sort_values("slip")
        shape = data["mu"].to_numpy() / data["mu"].max()
        self.mu_peak = mu_peak
        self._k = [float(k) for k in data["slip"]]
        self._mu = [float(m) * mu_peak for m in shape]
        self._k_peak = self._k[int(np.argmax(shape))]

    def mu(self, kappa: float) -> float:
        sign = 1.0 if kappa >= 0 else -1.0
        k = abs(kappa)
        ks, mus = self._k, self._mu
        if k >= ks[-1]:
            return sign * mus[-1]
        i = bisect.bisect_right(ks, k) - 1
        return sign * (mus[i] + (mus[i + 1] - mus[i]) * (k - ks[i]) / (ks[i + 1] - ks[i]))

    def fx(self, kappa: float, fz: float) -> float:
        return self.mu(kappa) * fz

    def peak(self, fz: float):
        return self._k_peak, self.mu_peak


class PacejkaTire:
    """TTC Pacejka fit from tire/, tabulated over (slip, Fz) at zero slip angle.

    `scale` maps belt grip to road grip; the old notebook used 0.5-0.6.
    """

    def __init__(self, coeffs=None, scale: float = 0.6, v: float = 15.0,
                 fz_grid=np.arange(100.0, 2001.0, 100.0),
                 k_grid=np.round(np.arange(-0.3, 1.0001, 0.01), 4)):
        from tire import tire_utils
        from tire.tire_model_pacejka_2010 import tire_model_from_arr

        model = tire_model_from_arr(tire_utils.H_LC0_18X6_7 if coeffs is None else coeffs)
        self.scale = scale
        self._fz = [float(f) for f in fz_grid]
        self._k = [float(k) for k in k_grid]
        self._k0, self._dk = self._k[0], self._k[1] - self._k[0]
        # mu = Fx / Fz, so interpolating across Fz stays well-behaved
        self._mu = []
        self._peaks = []
        pos = k_grid >= 0
        for fz in fz_grid:
            fx, _, _ = model.s_r_sweep(float(fz), 0.0, k_grid, v=v)
            mu = np.asarray(fx) / fz * scale
            self._mu.append([float(m) for m in mu])
            i = int(np.argmax(mu[pos]))
            self._peaks.append((float(k_grid[pos][i]), float(mu[pos][i])))

    def _fz_index(self, fz):
        fzs = self._fz
        if fz <= fzs[0]:
            return 0, 0.0
        if fz >= fzs[-1]:
            return len(fzs) - 2, 1.0
        i = bisect.bisect_right(fzs, fz) - 1
        return i, (fz - fzs[i]) / (fzs[i + 1] - fzs[i])

    def fx(self, kappa: float, fz: float) -> float:
        if fz <= 0:
            return 0.0
        fk = min(max((kappa - self._k0) / self._dk, 0.0), len(self._k) - 1.000001)
        j = int(fk)
        dj = fk - j
        i, di = self._fz_index(fz)
        a, b = self._mu[i], self._mu[i + 1]
        m0 = a[j] + (a[j + 1] - a[j]) * dj
        m1 = b[j] + (b[j + 1] - b[j]) * dj
        return (m0 + (m1 - m0) * di) * fz

    def peak(self, fz: float):
        i, di = self._fz_index(fz)
        (k0, m0), (k1, m1) = self._peaks[i], self._peaks[i + 1]
        return k0 + (k1 - k0) * di, m0 + (m1 - m0) * di


@lru_cache(maxsize=1)
def default_tire() -> PacejkaTire:
    """The sim's default tire: TTC LC0 18x6-7 fit, scaled 0.6 to road grip. Matches
    KS5e wheelspin better than CurveTire, which loses grip too fast past the peak."""
    return PacejkaTire()
