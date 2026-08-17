"""Affine device-clock -> wall-clock fit, used for the Polar H10.

Vectorized adaptation of polar-h10-data-collection/clock_sync.py's
`LinearClockFit` (same math, credited there): fits

    wall_ns = wall_ns0 + rate * (device_ns - device_ns0)

by least squares over every (device_ns, wall_ns) checkpoint at once (the
"full-session hindsight" pass polar_h10_postprocess.py does offline, as
opposed to polar_h10_capture.py's live/causal incremental refit -- see
docs/Dataset_Sync_Design.md §3). With fewer than 2 checkpoints this
degenerates to a single fixed offset (rate=1.0), same as the live capture's
fallback behaviour.

Deltas are computed relative to the first checkpoint before fitting (not raw
device_ns/wall_ns, which are ~1e18-scale) to avoid float precision loss --
same reason clock_sync.py does it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LinearClockFitResult:
    device_ns0: int
    wall_ns0: int
    rate: float  # ~1.0; (rate-1)*1e6 = drift in ppm, device fast if positive
    xmean: float
    ymean: float
    n_checkpoints: int

    def rate_ppm(self) -> float:
        return (self.rate - 1.0) * 1e6

    def predict_wall_ns(self, device_ns: np.ndarray) -> np.ndarray:
        x = device_ns.astype(np.float64) - self.device_ns0
        y = self.rate * (x - self.xmean) + self.ymean
        return self.wall_ns0 + np.round(y).astype(np.int64)


def fit_linear_clock(device_ns: np.ndarray, wall_ns: np.ndarray) -> LinearClockFitResult:
    """`device_ns`/`wall_ns` are the full set of checkpoints for one session
    (e.g. every row of *_polar_checkpoints.csv), not just two."""
    device_ns = np.asarray(device_ns, dtype=np.int64)
    wall_ns = np.asarray(wall_ns, dtype=np.int64)
    if device_ns.size == 0:
        raise ValueError("need at least 1 checkpoint to anchor a fit")

    device_ns0 = int(device_ns[0])
    wall_ns0 = int(wall_ns[0])
    n = device_ns.size

    if n < 2:
        return LinearClockFitResult(device_ns0, wall_ns0, rate=1.0, xmean=0.0, ymean=0.0, n_checkpoints=n)

    xs = (device_ns - device_ns0).astype(np.float64)
    ys = (wall_ns - wall_ns0).astype(np.float64)
    xmean = float(xs.mean())
    ymean = float(ys.mean())
    sxx = float(np.sum((xs - xmean) ** 2))
    sxy = float(np.sum((xs - xmean) * (ys - ymean)))
    rate = (sxy / sxx) if sxx else 1.0

    return LinearClockFitResult(device_ns0, wall_ns0, rate=rate, xmean=xmean, ymean=ymean, n_checkpoints=n)


def residuals_ns(fit: LinearClockFitResult, device_ns: np.ndarray, wall_ns: np.ndarray) -> np.ndarray:
    """actual - predicted wall_ns per checkpoint, for QC reporting (large
    residuals flag a noisy checkpoint, e.g. one captured during a scheduling
    hiccup, rather than real clock behaviour)."""
    predicted = fit.predict_wall_ns(np.asarray(device_ns, dtype=np.int64))
    return np.asarray(wall_ns, dtype=np.int64) - predicted
