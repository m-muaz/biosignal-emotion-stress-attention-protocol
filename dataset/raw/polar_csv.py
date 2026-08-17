"""Parser for the Polar H10 CSVs produced by polar_h10_capture.py.

Files:
    <ts>_polar_ecg.csv          130 Hz  device_timestamp_ns,wall_clock_time,ecg_uV,participant
    <ts>_polar_acc.csv          200 Hz  device_timestamp_ns,wall_clock_time,x_mg,y_mg,z_mg,participant
    <ts>_polar_checkpoints.csv          seq,device_ns,wall_ns,wall_iso,participant

`wall_clock_time` in the ecg/acc files is a *causal* (improves-as-it-goes)
regression estimate computed live by polar_h10_capture.py -- see
clock_sync.py there. We recompute it here using every checkpoint in the
file at once (dataset.clock_fit.fit_linear_clock), which is strictly more
accurate -- exactly what polar_h10_postprocess.py does offline, just
integrated into this pipeline instead of run as a separate script.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from dataset.clock_fit import LinearClockFitResult, fit_linear_clock, residuals_ns

ECG_SAMPLE_RATE_HZ = 130
ACC_SAMPLE_RATE_HZ = 200


@dataclass
class PolarStream:
    path: Path
    device_ns: np.ndarray
    values: pd.DataFrame  # value columns only (ecg_uV, or x_mg/y_mg/z_mg)
    participant: pd.Series


def _read_stream_csv(path: Path, value_cols: list[str]) -> PolarStream:
    path = Path(path)
    df = pd.read_csv(path)
    missing = {"device_timestamp_ns", *value_cols} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing expected column(s) {missing}")
    return PolarStream(
        path=path,
        device_ns=df["device_timestamp_ns"].to_numpy(dtype=np.int64),
        values=df[value_cols].reset_index(drop=True),
        participant=df.get("participant", pd.Series([""] * len(df))),
    )


def parse_polar_ecg_csv(path: Path) -> PolarStream:
    return _read_stream_csv(path, ["ecg_uV"])


def parse_polar_acc_csv(path: Path) -> PolarStream:
    return _read_stream_csv(path, ["x_mg", "y_mg", "z_mg"])


@dataclass
class PolarCheckpoints:
    path: Path
    device_ns: np.ndarray
    wall_ns: np.ndarray


def parse_polar_checkpoints_csv(path: Path) -> PolarCheckpoints:
    path = Path(path)
    df = pd.read_csv(path)
    missing = {"device_ns", "wall_ns"} - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing expected column(s) {missing}")
    return PolarCheckpoints(
        path=path,
        device_ns=df["device_ns"].to_numpy(dtype=np.int64),
        wall_ns=df["wall_ns"].to_numpy(dtype=np.int64),
    )


@dataclass
class PolarSyncQC:
    n_checkpoints: int
    rate_ppm: float | None
    max_residual_ms: float | None
    mean_residual_ms: float | None


def refit_full_session(checkpoints: PolarCheckpoints) -> tuple[LinearClockFitResult, PolarSyncQC]:
    fit = fit_linear_clock(checkpoints.device_ns, checkpoints.wall_ns)
    if fit.n_checkpoints >= 2:
        resid = residuals_ns(fit, checkpoints.device_ns, checkpoints.wall_ns)
        resid_ms = np.abs(resid) / 1e6
        qc = PolarSyncQC(
            n_checkpoints=fit.n_checkpoints,
            rate_ppm=fit.rate_ppm(),
            max_residual_ms=float(resid_ms.max()),
            mean_residual_ms=float(resid_ms.mean()),
        )
    else:
        qc = PolarSyncQC(n_checkpoints=fit.n_checkpoints, rate_ppm=None, max_residual_ms=None, mean_residual_ms=None)
    return fit, qc


def wall_utc_seconds(stream: PolarStream, fit: LinearClockFitResult) -> np.ndarray:
    return fit.predict_wall_ns(stream.device_ns) / 1e9
