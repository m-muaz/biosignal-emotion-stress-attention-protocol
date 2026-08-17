"""Event-windowed dataset over the canonical synced Parquet output.

Per the design decision (2026-08-17): streams stay at native rate in the
canonical Parquet files; this is where per-window resampling to a common
target rate happens, on the fly, only for the window being fetched --
nothing is resampled up front at dataset-build time.

`torch` is NOT required to import this module or to build/inspect windows
-- only `EventWindowDataset.__getitem__(..., to_tensor=True)` (or wrapping
this in a real `torch.utils.data.DataLoader`) needs it installed. Duck-types
the `Dataset` protocol (`__len__`/`__getitem__`) rather than subclassing
`torch.utils.data.Dataset`, specifically so nothing here needs torch at
import time.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dataset.windows import Window, extract_all_windows, load_events_parquet

# stream name -> its value columns in the canonical parquet (wall_utc_s is
# always column 0, added by build_dataset.py, and excluded from this list).
STREAM_VALUE_COLUMNS = {
    "ear_eeg_out.ads1299": [f"ch{i}" for i in range(1, 9)],
    "ear_eeg_in.ads1299": [f"ch{i}" for i in range(1, 9)],
    "ear_eeg_in.ppg": ["red", "ir", "green"],
    "ear_eeg_in.imu": ["acc_x_mg", "acc_y_mg", "acc_z_mg", "gyro_x_dps", "gyro_y_dps", "gyro_z_dps"],
    "ear_eeg_in.mlx90632": ["ambient_c", "object_c"],
    "ear_eeg_in.bme680": ["pressure_hpa", "humidity_percent"],
    "wristband.ppg": ["red", "ir", "green"],
    "wristband.imu": ["accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"],
    "wristband.gsr": ["gsr"],
    "wristband.mag": ["mag_x", "mag_y", "mag_z"],
    "wristband.mlx": ["object_temp", "ambient_temp"],
    "wristband.bme": ["temp", "hum", "pres", "gas"],
    "polar_h10.ecg": ["ecg_uV"],
    "polar_h10.acc": ["x_mg", "y_mg", "z_mg"],
}


def _slice_and_resample(df: pd.DataFrame, t_start: float, t_end: float, value_cols: list[str], target_hz: float | None):
    """Returns (grid_seconds, values) where grid_seconds is relative to
    t_start (0 = window start). `target_hz=None` returns the stream's own
    samples as-is (irregular spacing, but with an explicit time axis so
    downstream code can still align it); otherwise linearly interpolates
    onto a fixed-step grid at target_hz. Returns None if the window has no
    samples in this stream at all (e.g. a device that opted out, or a gap)."""
    wall = df["wall_utc_s"].to_numpy()
    mask = (wall >= t_start) & (wall <= t_end)
    if not mask.any():
        return None
    sub_wall = wall[mask]
    sub_vals = df.loc[mask, value_cols].to_numpy(dtype=np.float64)

    if target_hz is None:
        return sub_wall - t_start, sub_vals

    n = max(2, int(round((t_end - t_start) * target_hz)))
    grid = np.linspace(0.0, t_end - t_start, n)
    if sub_wall.size < 2:
        # can't interpolate from a single sample -- hold it constant across the window
        out = np.repeat(sub_vals, n, axis=0) if sub_vals.size else np.full((n, len(value_cols)), np.nan)
        return grid, out
    src_t = sub_wall - t_start
    out = np.empty((n, len(value_cols)), dtype=np.float64)
    for i in range(len(value_cols)):
        out[:, i] = np.interp(grid, src_t, sub_vals[:, i])
    return grid, out


def load_window_biosignals(
    processed_dir: Path,
    participant_id: str,
    window: Window,
    streams: list[str] | None = None,
    target_hz: float | None = None,
    _stream_cache: dict | None = None,
) -> dict:
    """Fetches `{stream_name: {"t_rel_s": ..., "values": ..., "columns": ...}}`
    for one Window, reading straight from the canonical Parquet -- the same
    slicing/resampling `EventWindowDataset.__getitem__` uses internally,
    exposed standalone for ad-hoc inspection/plotting (see
    notebooks/dataset_walkthrough.ipynb). `t_rel_s` is seconds relative to
    `window.t_start` (0 = window start); only streams with >=1 sample in
    the window are included.

    `_stream_cache` is an optional {(participant_id, stream): DataFrame|None}
    dict to reuse across many calls for the same participant (avoids
    re-reading a multi-MB Parquet file per window) -- EventWindowDataset
    passes its own cache in; standalone notebook/script callers can omit it
    (each call just reads fresh) or pass a plain `{}` and reuse it themselves."""
    streams = streams or list(STREAM_VALUE_COLUMNS)
    out = {}
    for stream in streams:
        if _stream_cache is not None and (participant_id, stream) in _stream_cache:
            df = _stream_cache[(participant_id, stream)]
        else:
            path = Path(processed_dir) / participant_id / f"{stream}.parquet"
            df = pd.read_parquet(path) if path.exists() else None
            if _stream_cache is not None:
                _stream_cache[(participant_id, stream)] = df
        if df is None:
            continue
        sliced = _slice_and_resample(df, window.t_start, window.t_end, STREAM_VALUE_COLUMNS[stream], target_hz)
        if sliced is None:
            continue
        t_rel, values = sliced
        out[stream] = {"t_rel_s": t_rel, "values": values, "columns": STREAM_VALUE_COLUMNS[stream]}
    return out


class EventWindowDataset:
    """Duck-types the PyTorch `Dataset` protocol (`__len__`/`__getitem__`)
    over event-defined windows (dataset/windows.py) across one or more
    participants' canonical Parquet output (dataset/build_dataset.py).

    Usage (no torch needed):
        ds = EventWindowDataset(processed_dir, ["P001", "P007"], tasks="emotion", target_hz=250.0)
        sample = ds[0]   # dict of numpy arrays + metadata

        # every task/window kind at once, including auto-detected gaps:
        ds_all = EventWindowDataset(processed_dir, ["P001", "P007"])
        # just baselines across every task, for a resting-vs-task contrast:
        ds_baseline = EventWindowDataset(processed_dir, ["P001", "P007"], window_types=["baseline"])

    Usage with a real torch DataLoader:
        from torch.utils.data import DataLoader
        loader = DataLoader(ds, batch_size=4, collate_fn=lambda b: b)  # custom collate needed --
                                                                        # windows have variable length/streams
    """

    def __init__(
        self,
        processed_dir: Path,
        participant_ids: list[str],
        tasks: str | list[str] | None = None,
        window_types: list[str] | None = None,
        target_hz: float | None = None,
        streams: list[str] | None = None,
        min_gap_s: float = 1.0,
    ):
        """`tasks`: a task name, a list of task names, or None for every
        task extract_all_windows knows about (including "transition" gap
        windows -- pass tasks=["emotion", ...] explicitly to exclude those).
        `window_types`: further filter to e.g. ["trial"] only, or None for
        all window kinds (trial/block/baseline/self_report/questionnaire/transition).
        `min_gap_s`: passed through to extract_all_windows's gap detection."""
        self.processed_dir = Path(processed_dir)
        self.tasks = [tasks] if isinstance(tasks, str) else tasks
        self.window_types = window_types
        self.target_hz = target_hz
        self.streams = streams or list(STREAM_VALUE_COLUMNS)
        self._stream_cache: dict[tuple[str, str], pd.DataFrame | None] = {}

        self.windows: list[tuple[str, "Window"]] = []
        for pid in participant_ids:
            events_path = self.processed_dir / pid / "events.parquet"
            if not events_path.exists():
                continue
            events = load_events_parquet(events_path)
            for w in extract_all_windows(events, min_gap_s=min_gap_s):
                if self.tasks is not None and w.task not in self.tasks:
                    continue
                if self.window_types is not None and w.window_type not in self.window_types:
                    continue
                self.windows.append((pid, w))

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int, to_tensor: bool = False) -> dict:
        participant_id, w = self.windows[idx]
        biosignals = load_window_biosignals(
            self.processed_dir, participant_id, w, streams=self.streams, target_hz=self.target_hz, _stream_cache=self._stream_cache
        )

        if to_tensor:
            import torch  # lazy: only touched if the caller actually asks for tensors

            for stream, entry in biosignals.items():
                entry["values"] = torch.from_numpy(entry["values"]).float()
                entry["t_rel_s"] = torch.from_numpy(entry["t_rel_s"]).float()

        return {
            "participant_id": participant_id,
            "task": w.task,
            "window_type": w.window_type,
            "t_start": w.t_start,
            "t_end": w.t_end,
            "block_index": w.block_index,
            "trial_index": w.trial_index,
            "condition_label": w.condition_label,
            "stimulus": w.stimulus,
            "labels": w.labels,
            "meta": w.meta,
            "biosignals": biosignals,
        }
