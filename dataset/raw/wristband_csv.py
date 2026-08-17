"""Parser for the wristband's six per-modality CSVs plus its sync file.

Files (all plain-text CSV, `<modality>-S%06d.csv`):
    ppg-S*.csv   200 Hz  device_us,red,ir,green
    imu-S*.csv   200 Hz  device_us,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z
    gsr-S*.csv   200 Hz  device_us,gsr
    mag-S*.csv   100 Hz  device_us,mag_x,mag_y,mag_z
    mlx-S*.csv     1 Hz  device_us,object_temp,ambient_temp
    bme-S*.csv     1 Hz  device_us,temp,hum,pres,gas
    meta-S*.csv          sync_index,computer_epoch_ms,device_us  (0+ rows --
                          0 rows means this session was never synced, see
                          the P009 backward-anchor override in overrides.yaml)

Sync anchor (normal case, >=1 meta row): wall_s = computer_epoch_ms/1000 +
(device_us - sync_device_us)/1e6, using the LAST meta row as the anchor (a
session can in principle be re-synced mid-recording; only v1 in practice
ever produces one row, but this doesn't assume that).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

MODALITY_COLUMNS = {
    "ppg": ["device_us", "red", "ir", "green"],
    "imu": ["device_us", "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"],
    "gsr": ["device_us", "gsr"],
    "mag": ["device_us", "mag_x", "mag_y", "mag_z"],
    "mlx": ["device_us", "object_temp", "ambient_temp"],
    "bme": ["device_us", "temp", "hum", "pres", "gas"],
}
NATIVE_RATE_HZ = {"ppg": 200, "imu": 200, "gsr": 200, "mag": 100, "mlx": 1, "bme": 1}


@dataclass
class WristbandModality:
    path: Path
    modality: str
    device_us: np.ndarray
    data: pd.DataFrame  # value columns only (device_us excluded), same row order


def parse_wristband_modality_csv(path: Path, modality: str) -> WristbandModality:
    path = Path(path)
    expected_cols = MODALITY_COLUMNS[modality]
    df = pd.read_csv(path)
    missing = set(expected_cols) - set(df.columns)
    if missing:
        raise ValueError(f"{path}: missing expected column(s) {missing}; found {list(df.columns)}")
    device_us = df["device_us"].to_numpy(dtype=np.int64)
    data = df[[c for c in expected_cols if c != "device_us"]].reset_index(drop=True)
    return WristbandModality(path=path, modality=modality, device_us=device_us, data=data)


@dataclass
class WristbandSyncAnchor:
    path: Path
    sync_index: int
    computer_epoch_ms: int
    sync_device_us: int


def parse_wristband_meta_csv(path: Path) -> list[WristbandSyncAnchor]:
    """Returns an empty list if the file has a header but zero data rows --
    that's the documented "never synced" case (e.g. participant P009's
    session 2), not a parse error."""
    path = Path(path)
    df = pd.read_csv(path)
    return [
        WristbandSyncAnchor(
            path=path,
            sync_index=int(row.sync_index),
            computer_epoch_ms=int(row.computer_epoch_ms),
            sync_device_us=int(row.device_us),
        )
        for row in df.itertuples(index=False)
    ]


def wall_utc_seconds_from_anchor(device_us: np.ndarray, anchor: WristbandSyncAnchor) -> np.ndarray:
    return anchor.computer_epoch_ms / 1000.0 + (device_us.astype(np.float64) - anchor.sync_device_us) / 1e6
