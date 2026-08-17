"""Parser for the ear-EEG / in-ear-EEG ADS1299 `.bin`/`.BIN` files.

Format (reverse-engineered from ADS1299_BLE_main/main/types.h and the
gaoteng in-ear sibling, which uses the byte-identical struct layout):

File = [ADS1299_FileHeader, 36 bytes] + [ADS1299_Record, 45 bytes] * N

Header (packed, little-endian):
    offset  size  field
    0       4     magic              uint32, 0x41443132
                                      (spells "AD12" big-endian; since it's
                                      stored little-endian on disk, a raw
                                      hexdump shows the bytes as "21DA" --
                                      same value, just byte order)
    4       1     version            uint8, always 1 so far
    5       8     boot_uptime_s      float64, esp_timer seconds at file-open
    13      8     unix_time_at_sync  float64, 0.0 until CMD_SET_TIME arrives
    21      8     uptime_at_sync     float64, 0.0 until CMD_SET_TIME arrives
    29      7     reserved

Record (packed, little-endian):
    offset  size  field
    0       8     timestamp    float64, boot-relative seconds (esp_timer)
    8       4     counter      uint32, BLOCK counter -- repeats 20x per
                                20-sample transport block, NOT a per-sample
                                sequence number. Don't use it for ordering;
                                record order in the file already is sample
                                order.
    12      32    channels     int32[8], raw sign-extended 24-bit ADC codes
    44      1     crc8         CRC-8 (poly 0x07, non-reflected) over bytes 0-43

Sync anchor: wall_s = record.timestamp + (unix_time_at_sync - uptime_at_sync),
using the header's own two fields -- see docs/Dataset_Sync_Design.md §1.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAGIC = 0x41443132
HEADER_DTYPE = np.dtype(
    [
        ("magic", "<u4"),
        ("version", "u1"),
        ("boot_uptime_s", "<f8"),
        ("unix_time_at_sync", "<f8"),
        ("uptime_at_sync", "<f8"),
        ("reserved", "u1", (7,)),
    ]
)
RECORD_DTYPE = np.dtype(
    [
        ("timestamp", "<f8"),
        ("counter", "<u4"),
        ("channels", "<i4", (8,)),
        ("crc8", "u1"),
    ]
)
assert HEADER_DTYPE.itemsize == 36
assert RECORD_DTYPE.itemsize == 45

NUM_CHANNELS = 8


def _crc8_table(poly: int = 0x07) -> np.ndarray:
    table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFF if (crc & 0x80) else (crc << 1) & 0xFF
        table[i] = crc
    return table


_CRC8_TABLE = _crc8_table()


def _crc8_of_records(raw_bytes: np.ndarray, n_records: int, record_size: int, payload_size: int) -> np.ndarray:
    """Vectorized CRC-8 over the first `payload_size` bytes of each fixed-size
    record. `raw_bytes` is the flat uint8 view of the whole records region.
    Loops over byte-position-within-record (a small, fixed count), not over
    records, so this stays fast even for multi-million-record files."""
    rows = raw_bytes.reshape(n_records, record_size)
    crc = np.zeros(n_records, dtype=np.uint8)
    for col in range(payload_size):
        crc = _CRC8_TABLE[crc ^ rows[:, col]]
    return crc


@dataclass
class Ads1299File:
    path: Path
    magic_ok: bool
    version: int
    boot_uptime_s: float
    unix_time_at_sync: float
    uptime_at_sync: float
    synced: bool  # unix_time_at_sync != 0.0 -- this boot actually got CMD_SET_TIME
    n_records: int
    timestamp: np.ndarray  # boot-relative seconds, shape (n_records,)
    channels: np.ndarray  # raw ADC codes, shape (n_records, 8)
    counter: np.ndarray  # block counter, shape (n_records,)
    crc_ok_fraction: float | None  # None unless validate_crc=True was passed

    @property
    def sync_offset_s(self) -> float:
        """wall_s = record.timestamp + sync_offset_s. Only meaningful if
        `synced` is True -- callers must check that first."""
        return self.unix_time_at_sync - self.uptime_at_sync

    def wall_utc_seconds(self) -> np.ndarray:
        if not self.synced:
            raise ValueError(
                f"{self.path}: this boot never received CMD_SET_TIME "
                f"(unix_time_at_sync=0.0) -- no wall-clock mapping exists, "
                f"this file should have been excluded by the session resolver."
            )
        return self.timestamp + self.sync_offset_s


@dataclass
class Ads1299HeaderPeek:
    path: Path
    magic_ok: bool
    unix_time_at_sync: float
    uptime_at_sync: float
    synced: bool
    file_size_bytes: int

    @property
    def sync_offset_s(self) -> float:
        return self.unix_time_at_sync - self.uptime_at_sync


def peek_ads1299_header(path: Path) -> Ads1299HeaderPeek:
    """Reads only the 36-byte header -- cheap enough to call on every restart
    candidate (some of which are 30+ MB) before deciding which one to fully parse."""
    path = Path(path)
    file_size = path.stat().st_size
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=HEADER_DTYPE, count=1)
    if header.size == 0:
        return Ads1299HeaderPeek(path, magic_ok=False, unix_time_at_sync=0.0, uptime_at_sync=0.0, synced=False, file_size_bytes=file_size)
    header = header[0]
    magic_ok = int(header["magic"]) == MAGIC
    unix_time_at_sync = float(header["unix_time_at_sync"]) if magic_ok else 0.0
    uptime_at_sync = float(header["uptime_at_sync"]) if magic_ok else 0.0
    return Ads1299HeaderPeek(
        path=path,
        magic_ok=magic_ok,
        unix_time_at_sync=unix_time_at_sync,
        uptime_at_sync=uptime_at_sync,
        synced=magic_ok and unix_time_at_sync != 0.0,
        file_size_bytes=file_size,
    )


def parse_ads1299_bin(path: Path, validate_crc: bool = False) -> Ads1299File:
    path = Path(path)
    with open(path, "rb") as f:
        header = np.fromfile(f, dtype=HEADER_DTYPE, count=1)
        if header.size == 0:
            raise ValueError(f"{path}: file too small to contain a header")
        header = header[0]
        records = np.fromfile(f, dtype=RECORD_DTYPE)

    magic_ok = int(header["magic"]) == MAGIC
    if not magic_ok:
        raise ValueError(
            f"{path}: bad magic 0x{int(header['magic']):08x} (expected "
            f"0x{MAGIC:08x} / \"AD12\") -- not a recognized ADS1299 file, "
            f"or the file is truncated/corrupt."
        )

    unix_time_at_sync = float(header["unix_time_at_sync"])
    uptime_at_sync = float(header["uptime_at_sync"])
    synced = unix_time_at_sync != 0.0

    crc_ok_fraction = None
    if validate_crc and records.size:
        raw = records.view("u1").reshape(records.size, RECORD_DTYPE.itemsize)
        computed = _crc8_of_records(raw.ravel(), records.size, RECORD_DTYPE.itemsize, RECORD_DTYPE.itemsize - 1)
        crc_ok_fraction = float(np.mean(computed == records["crc8"]))

    return Ads1299File(
        path=path,
        magic_ok=magic_ok,
        version=int(header["version"]),
        boot_uptime_s=float(header["boot_uptime_s"]),
        unix_time_at_sync=unix_time_at_sync,
        uptime_at_sync=uptime_at_sync,
        synced=synced,
        n_records=int(records.size),
        timestamp=records["timestamp"].copy(),
        channels=records["channels"].copy(),
        counter=records["counter"].copy(),
        crc_ok_fraction=crc_ok_fraction,
    )
