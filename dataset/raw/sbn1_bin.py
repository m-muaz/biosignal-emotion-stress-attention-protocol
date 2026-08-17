"""Parser for the in-ear device's onboard-peripheral `.bin` files: PPG.bin,
IMU.bin, MLX90632.bin, BME680.bin. All four share one "SBN1" framing
(reverse-engineered from ADS1299_BLE-main-gaoteng's sensor_binary_format.h).

Header (packed, little-endian, 32 bytes):
    offset  size  field
    0       4     magic          uint32, 0x314E4253 ("SBN1")
    4       1     version        uint8
    5       1     sensor_type    uint8: 1=PPG, 2=BMI270(IMU), 3=MLX90632, 4=BME680
    6       2     header_size    uint16, = 32
    8       2     record_size    uint16, sensor-specific (see RECORD_DTYPES)
    10      2     batch_size     uint16, records per SD write batch
    12      2     sample_rate_hz uint16, the rate the firmware was configured with
    14      2     flags          uint16, bit0=RECORD_CRC8 (always set),
                                  bit1=TIME_OFFSET_VALID (set once CMD_SET_TIME
                                  has patched `reserved` below)
    16      8     boot_uptime_s  float64
    24      8     reserved       all-zero until sync; after sync, holds
                                 time_offset_s (float64) = unix_time - uptime_s

Records (all packed, little-endian, all end in a 1-byte CRC-8 over the
preceding bytes):
    PPG        (25B): sample(u32) uptime_s(f8) red(u32) ir(u32) green(u32) crc8
    IMU/BMI270 (37B): sample(u32) uptime_s(f8) acc_x/y/z_mg(f4x3) gyro_x/y/z_dps(f4x3) crc8
    MLX90632   (23B): sample(u32) uptime_s(f8) status(u16) ambient_c(f4) object_c(f4) crc8
    BME680     (24B): sample(u32) uptime_s(f8) pressure_hpa(f4) humidity_percent(f4)
                       status(u8) measurement_index(u8) in_range(u8) crc8

Sync anchor: wall_s = record.uptime_s + header.time_offset_s (only valid once
TIME_OFFSET_VALID is set -- see docs/Dataset_Sync_Design.md §1). This is the
same additive offset the sibling ADS1299 file on the same device/boot gets,
just precomputed into one field instead of stored as two.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = 0x314E4253  # "SBN1"
TIME_OFFSET_VALID_FLAG = 0x0002

SENSOR_TYPES = {1: "ppg", 2: "imu", 3: "mlx90632", 4: "bme680"}

HEADER_DTYPE = np.dtype(
    [
        ("magic", "<u4"),
        ("version", "u1"),
        ("sensor_type", "u1"),
        ("header_size", "<u2"),
        ("record_size", "<u2"),
        ("batch_size", "<u2"),
        ("sample_rate_hz", "<u2"),
        ("flags", "<u2"),
        ("boot_uptime_s", "<f8"),
        ("reserved", "u1", (8,)),
    ]
)
assert HEADER_DTYPE.itemsize == 32

RECORD_DTYPES = {
    "ppg": np.dtype(
        [("sample", "<u4"), ("uptime_s", "<f8"), ("red", "<u4"), ("ir", "<u4"), ("green", "<u4"), ("crc8", "u1")]
    ),
    "imu": np.dtype(
        [
            ("sample", "<u4"),
            ("uptime_s", "<f8"),
            ("acc_x_mg", "<f4"),
            ("acc_y_mg", "<f4"),
            ("acc_z_mg", "<f4"),
            ("gyro_x_dps", "<f4"),
            ("gyro_y_dps", "<f4"),
            ("gyro_z_dps", "<f4"),
            ("crc8", "u1"),
        ]
    ),
    "mlx90632": np.dtype(
        [("sample", "<u4"), ("uptime_s", "<f8"), ("status", "<u2"), ("ambient_c", "<f4"), ("object_c", "<f4"), ("crc8", "u1")]
    ),
    "bme680": np.dtype(
        [
            ("sample", "<u4"),
            ("uptime_s", "<f8"),
            ("pressure_hpa", "<f4"),
            ("humidity_percent", "<f4"),
            ("status", "u1"),
            ("measurement_index", "u1"),
            ("in_range", "u1"),
            ("crc8", "u1"),
        ]
    ),
}
for _name, _dt in RECORD_DTYPES.items():
    assert _dt.itemsize in (25, 37, 23, 24), f"{_name}: unexpected itemsize {_dt.itemsize}"


@dataclass
class Sbn1File:
    path: Path
    sensor_type: str
    sample_rate_hz: int
    batch_size: int
    boot_uptime_s: float
    time_offset_s: float | None  # None until CMD_SET_TIME sync -- see `synced`
    synced: bool
    n_records: int
    fields: dict  # field_name -> np.ndarray, e.g. {"uptime_s": ..., "red": ..., ...}

    def wall_utc_seconds(self) -> np.ndarray:
        if not self.synced:
            raise ValueError(
                f"{self.path}: this boot never received CMD_SET_TIME "
                f"(TIME_OFFSET_VALID flag not set) -- no wall-clock mapping "
                f"exists, this file should have been excluded by the session resolver."
            )
        return self.fields["uptime_s"] + self.time_offset_s


def parse_sbn1_bin(path: Path) -> Sbn1File:
    path = Path(path)
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=HEADER_DTYPE, count=1)
        if header.size == 0:
            raise ValueError(f"{path}: file too small to contain a header")
        header = header[0]

        if int(header["magic"]) != MAGIC:
            raise ValueError(
                f"{path}: bad magic 0x{int(header['magic']):08x} (expected 0x{MAGIC:08x} / \"SBN1\")"
            )
        sensor_type = SENSOR_TYPES.get(int(header["sensor_type"]))
        if sensor_type is None:
            raise ValueError(f"{path}: unknown sensor_type byte {int(header['sensor_type'])}")

        record_dtype = RECORD_DTYPES[sensor_type]
        header_size = int(header["header_size"])
        if header_size != HEADER_DTYPE.itemsize:
            # Header claims a different size than we assumed -- trust the file
            # over our constant and seek to where it says records actually start.
            f.seek(header_size)
        records = np.fromfile(f, dtype=record_dtype)

    flags = int(header["flags"])
    synced = bool(flags & TIME_OFFSET_VALID_FLAG)
    time_offset_s = float(np.frombuffer(header["reserved"].tobytes(), dtype="<f8")[0]) if synced else None

    fields = {name: records[name].copy() for name in record_dtype.names if name not in ("sample", "crc8")}
    fields["sample"] = records["sample"].copy()

    return Sbn1File(
        path=path,
        sensor_type=sensor_type,
        sample_rate_hz=int(header["sample_rate_hz"]),
        batch_size=int(header["batch_size"]),
        boot_uptime_s=float(header["boot_uptime_s"]),
        time_offset_s=time_offset_s,
        synced=synced,
        n_records=int(records.size),
        fields=fields,
    )
