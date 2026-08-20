"""Fixed-shape `.npy` export on top of the event-windowed dataset
(`dataset/windows.py`, `dataset/torch_dataset.py`).

Each task window (e.g. one video clip's [t_start, t_end]) is cut into
fixed-length, non-overlapping epochs (default 1.0s) before resampling.
floor(duration / epoch_s) epochs come out of a window; any leftover
shorter than one epoch is dropped, never padded. Every epoch becomes its
own row and inherits its parent window's label, so epochs from
different-length clips/participants concatenate with no padding needed.

Per (task, window_kind) "file key" x stream, writes an {X.npy, y.npy} pair:
    X.shape == (N, C, samples_per_epoch)   float32   -- N = total epochs
    y.shape == (N,)                        int64
plus `<file_prefix>_meta.csv` (same row order) with every raw
participant-response field on the parent window, plus each row's own
epoch index/bounds for grouping epochs back into their source window.

See docs/Dataset_Sync_Design.md §8-9 for the label-scheme design writeup.

Usage:
    python -m dataset.export_npy --processed-dir <out_dir> --export-dir <out_dir>\\npy_export

    from dataset.export_npy import export_all
    manifest = export_all(processed_dir, export_dir)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataset.torch_dataset import STREAM_VALUE_COLUMNS
from dataset.windows import Window, extract_all_windows, load_events_parquet

SAMPLES_PER_SEGMENT = 200  # fallback for any stream not in DEFAULT_SAMPLES_PER_SEGMENT
DEFAULT_EEG_STREAM = "ear_eeg_out.ads1299"  # mandatory device -- stays the unsuffixed/primary stream

WRISTBAND_STREAMS = ["wristband.ppg", "wristband.imu", "wristband.gsr", "wristband.mag", "wristband.mlx", "wristband.bme"]
IN_EAR_EEG_STREAM = "ear_eeg_in.ads1299"  # optional 2nd EEG device
IN_EAR_AUX_STREAMS = ["ear_eeg_in.ppg", "ear_eeg_in.imu", "ear_eeg_in.mlx90632", "ear_eeg_in.bme680"]
POLAR_STREAMS = ["polar_h10.ecg", "polar_h10.acc"]

# samples/epoch per stream, default = round(native Hz) so nothing is
# meaningfully up/down-sampled; override via samples_per_segment={...}.
DEFAULT_SAMPLES_PER_SEGMENT = {
    "ear_eeg_out.ads1299": 200,
    "ear_eeg_in.ads1299": 200,
    "ear_eeg_in.ppg": 100,
    "ear_eeg_in.imu": 100,
    "ear_eeg_in.mlx90632": 1,
    "ear_eeg_in.bme680": 1,
    "wristband.ppg": 200,
    "wristband.imu": 200,
    "wristband.gsr": 200,
    "wristband.mag": 100,
    "wristband.mlx": 1,
    "wristband.bme": 1,
    "polar_h10.ecg": 130,
    "polar_h10.acc": 200,
}

# Filename suffix per stream (<key>_<suffix>_X.npy); primary EEG stays unsuffixed.
STREAM_SHORT_NAME = {
    DEFAULT_EEG_STREAM: None,
    "ear_eeg_in.ads1299": "eegin",
    "ear_eeg_in.ppg": "inear_ppg",
    "ear_eeg_in.imu": "inear_imu",
    "ear_eeg_in.mlx90632": "inear_mlx",
    "ear_eeg_in.bme680": "inear_bme",
    "wristband.ppg": "ppg",
    "wristband.imu": "imu",
    "wristband.gsr": "gsr",
    "wristband.mag": "mag",
    "wristband.mlx": "mlx",
    "wristband.bme": "bme",
    "polar_h10.ecg": "ecg",
    "polar_h10.acc": "polar_acc",
}

DEFAULT_STREAMS = [DEFAULT_EEG_STREAM, IN_EAR_EEG_STREAM] + IN_EAR_AUX_STREAMS + WRISTBAND_STREAMS + POLAR_STREAMS


def _file_prefix(key: str, stream: str) -> str:
    suffix = STREAM_SHORT_NAME.get(stream, stream.replace(".", "_"))
    return key if suffix is None else f"{key}_{suffix}"


# Epoch length (seconds) per file key -- override via epoch_seconds={...} /
# --epoch-seconds key=N,... . At the 1.0s default, windows shorter than 1s
# (sart_trial/stroop_trial/the lapse keys) floor to 0 epochs and get
# dropped (tracked in the manifest as too_short_for_one_epoch).
DEFAULT_EPOCH_SECONDS = {
    "emotion_trial": 1.0,
    "emotion_baseline": 1.0,
    "math_trial": 1.0,
    "math_baseline": 1.0,
    "highway_trial": 1.0,
    "highway_baseline": 1.0,
    "stroop_trial": 1.0,
    "stroop_baseline": 1.0,
    "schulte_trial": 1.0,
    "schulte_baseline": 1.0,
    "sart_trial": 1.0,
    "sart_baseline": 1.0,
    "sart_lapse_trial": 1.0,
    "stroop_lapse_trial": 1.0,
}


def _epoch_bounds(t_start: float, t_end: float, epoch_s: float) -> tuple[int, float]:
    """(n_epochs, t_end_effective) cutting [t_start, t_end] into
    floor(duration / epoch_s) non-overlapping epochs; leftover dropped."""
    n_epochs = int((t_end - t_start) // epoch_s)
    return n_epochs, t_start + n_epochs * epoch_s


# 0 = baseline/rest in every key, so concatenating <key>_baseline's y with
# <key>_trial's y reproduces the rest(0)/condition(1..) convention.
LABEL_MAPS = {
    "emotion_trial": {0: "baseline/rest", 1: "negative", 2: "neutral", 3: "positive"},
    "emotion_baseline": {0: "baseline/rest"},
    "math_trial": {0: "baseline/rest", 1: "tier_1", 2: "tier_2", 3: "tier_3"},
    "math_baseline": {0: "baseline/rest"},
    "highway_trial": {0: "baseline/rest", 1: "tier_1", 2: "tier_2", 3: "tier_3"},
    "highway_baseline": {0: "baseline/rest"},
    "stroop_trial": {0: "baseline/rest", 1: "incongruent", 2: "congruent"},
    "stroop_baseline": {0: "baseline/rest"},
    "schulte_trial": {0: "baseline/rest", 1: "slower_than_cohort_median", 2: "faster_than_cohort_median"},
    "schulte_baseline": {0: "baseline/rest"},
    "sart_trial": {0: "baseline/rest", 1: "go", 2: "no_go"},
    "sart_baseline": {0: "baseline/rest"},
    "sart_lapse_trial": {0: "attentive_correct_withhold", 1: "lapse_commission_error"},
    "stroop_lapse_trial": {0: "attentive_correct_response", 1: "lapse_incorrect_response"},
}


def _label_baseline(w: Window):
    return 0


def _label_emotion_trial(w: Window):
    return {"negative": 1, "neutral": 2, "positive": 3}.get(w.condition_label)


def _label_tier_block(w: Window):
    """Shared by math/highway blocks -- tier (1/2/3) lives in meta["tier"]."""
    return {1: 1, 2: 2, 3: 3}.get(w.meta.get("tier"))


def _label_stroop_trial(w: Window):
    cong = w.stimulus.get("congruent")
    return None if cong is None else (2 if cong else 1)


def _label_sart_trial(w: Window):
    is_omit = w.stimulus.get("is_omit")
    return None if is_omit is None else (2 if is_omit else 1)


def _make_schulte_label_fn(median_completion_s: float):
    """Median-split of completion_time_sec vs. a cohort-wide threshold
    (Schulte has no natural difficulty condition to label with instead).

    Caveat: completion_time_sec IS t_end - t_start for the window, so this
    label is a direct function of its own duration -- and n_epochs_in_window
    in meta.csv now makes that duration a literal, adjacent column. Not
    changed without discussion; `misclicks` (duration-independent) is the
    cleaner alternative if this is revisited."""

    def _label(w: Window):
        ct = w.labels.get("completion_time_sec")
        if ct is None:
            return None
        return 2 if ct < median_completion_s else 1

    return _label


def _label_lapse(w: Window):
    return w.meta.get("lapse_label")  # set by the lapse-window builders below


BASE_SPECS = [
    dict(key="emotion_trial", task="emotion", window_type="trial", label_fn=_label_emotion_trial),
    dict(key="emotion_baseline", task="emotion", window_type="baseline", label_fn=_label_baseline),
    dict(key="math_trial", task="stress", window_type="block", label_fn=_label_tier_block),
    dict(key="math_baseline", task="stress", window_type="baseline", label_fn=_label_baseline),
    dict(key="highway_trial", task="attention_highway", window_type="block", label_fn=_label_tier_block),
    dict(key="highway_baseline", task="attention_highway", window_type="baseline", label_fn=_label_baseline),
    dict(key="stroop_trial", task="attention_focus_stroop", window_type="trial", label_fn=_label_stroop_trial),
    dict(key="stroop_baseline", task="attention_focus_stroop", window_type="baseline", label_fn=_label_baseline),
    dict(key="schulte_trial", task="attention_focus_schulte", window_type="block", label_fn=None),  # filled in by _build_specs
    dict(key="schulte_baseline", task="attention_focus_schulte", window_type="baseline", label_fn=_label_baseline),
    dict(key="sart_trial", task="sart", window_type="trial", label_fn=_label_sart_trial),
    dict(key="sart_baseline", task="sart", window_type="baseline", label_fn=_label_baseline),
    dict(key="sart_lapse_trial", task="sart", window_type="lapse", label_fn=_label_lapse),
    dict(key="stroop_lapse_trial", task="attention_focus_stroop", window_type="lapse", label_fn=_label_lapse),
]

# Attentional-lapse windows: truncated to end BEFORE the response so a
# lapse classifier can't just read the motor/error-related signal that
# defines its own label. See docs/Dataset_Sync_Design.md §9.
LAPSE_RESPONSE_BUFFER_S = 0.15  # excludes premotor readiness-potential + the keypress itself
MIN_LAPSE_WINDOW_S = 0.2  # drop a trial if truncation leaves less pre-response signal than this


def _build_sart_lapse_windows(windows: list[Window]) -> list[Window]:
    """Commission-error lapse (Robertson et al. 1997): no-go trials only.
    SART's `response` event fires at a fixed ~1.16s trial-slot boundary
    regardless of the real keypress time, so the real keypress instant is
    `trial_start + rt`, not the event's own timestamp -- truncate from
    there. Trials with no keypress keep their full span (nothing to
    truncate)."""
    out = []
    for w in windows:
        if w.task != "sart" or w.window_type != "trial" or not w.stimulus.get("is_omit"):
            continue
        responded, rt = w.labels.get("responded"), w.labels.get("rt")
        if responded:
            if rt is None:
                continue
            t_end = w.t_start + rt - LAPSE_RESPONSE_BUFFER_S
            if t_end - w.t_start < MIN_LAPSE_WINDOW_S:
                continue
            label, condition_label = 1, "commission_error"
        else:
            t_end = w.t_end
            label, condition_label = 0, "correct_withhold"
        out.append(
            Window(
                task="sart",
                window_type="lapse",
                t_start=w.t_start,
                t_end=t_end,
                block_index=w.block_index,
                trial_index=w.trial_index,
                condition_label=condition_label,
                stimulus=w.stimulus,
                labels=w.labels,
                meta={**w.meta, "lapse_label": label},
            )
        )
    return out


def _build_stroop_lapse_windows(windows: list[Window]) -> list[Window]:
    """Error-based lapse: any incorrect response, any congruency. Unlike
    SART, Stroop's `response` event IS the real click instant, so truncate
    directly from `rt_ms`."""
    out = []
    for w in windows:
        if w.task != "attention_focus_stroop" or w.window_type != "trial":
            continue
        rt_ms, correct = w.labels.get("rt_ms"), w.labels.get("correct")
        if rt_ms is None or correct is None:
            continue
        t_end = w.t_start + (rt_ms / 1000.0) - LAPSE_RESPONSE_BUFFER_S
        if t_end - w.t_start < MIN_LAPSE_WINDOW_S:
            continue
        label = 0 if correct else 1
        out.append(
            Window(
                task="attention_focus_stroop",
                window_type="lapse",
                t_start=w.t_start,
                t_end=t_end,
                block_index=w.block_index,
                trial_index=w.trial_index,
                condition_label=("error" if label else "correct"),
                stimulus=w.stimulus,
                labels=w.labels,
                meta={**w.meta, "lapse_label": label},
            )
        )
    return out


def _segment_resample(df: pd.DataFrame, t_start: float, t_end: float, value_cols: list[str], n_segments: int, samples_per_segment: int = SAMPLES_PER_SEGMENT):
    """Divides [t_start, t_end] into n_segments equal-width segments,
    linearly interpolates onto one (n_segments * samples_per_segment)-point
    grid, reshapes to (len(value_cols), n_segments, samples_per_segment).
    Returns None if the window has zero samples of this stream at all."""
    wall = df["wall_utc_s"].to_numpy()
    mask = (wall >= t_start) & (wall <= t_end)
    if not mask.any():
        return None
    sub_wall = wall[mask] - t_start
    sub_vals = df.loc[mask, value_cols].to_numpy(dtype=np.float64)

    n_total = n_segments * samples_per_segment
    grid = np.linspace(0.0, t_end - t_start, n_total)
    if sub_wall.size < 2:
        flat = np.repeat(sub_vals, n_total, axis=0) if sub_vals.size else np.full((n_total, len(value_cols)), np.nan)
    else:
        flat = np.empty((n_total, len(value_cols)), dtype=np.float64)
        for i in range(len(value_cols)):
            flat[:, i] = np.interp(grid, sub_wall, sub_vals[:, i])

    return flat.T.reshape(len(value_cols), n_segments, samples_per_segment).astype(np.float32)


def _build_specs(schulte_median_s: float | None) -> list[dict]:
    specs = []
    for spec in BASE_SPECS:
        spec = dict(spec)
        if spec["key"] == "schulte_trial":
            spec["label_fn"] = _make_schulte_label_fn(schulte_median_s) if schulte_median_s is not None else (lambda w: None)
        specs.append(spec)
    return specs


def _gather_windows(processed_dir: Path, participant_ids: list[str]) -> dict[str, list[Window]]:
    """Every participant's windows plus derived lapse windows (window_type="lapse")."""
    out = {}
    for pid in participant_ids:
        events_path = processed_dir / pid / "events.parquet"
        if not events_path.exists():
            continue
        events = load_events_parquet(events_path)
        windows = extract_all_windows(events)
        windows = windows + _build_sart_lapse_windows(windows) + _build_stroop_lapse_windows(windows)
        out[pid] = windows
    return out


# t_start/t_end = this epoch's own bounds; window_t_start/window_t_end =
# the parent window's (e.g. whole clip's) span, for grouping epochs back.
META_COLUMNS = [
    "participant_id", "task", "window_type", "block_index", "trial_index", "condition_label",
    "window_t_start", "window_t_end", "window_duration_s",
    "epoch_index", "n_epochs_in_window", "t_start", "t_end",
    "label", "stimulus_json", "labels_json", "meta_json",
]


def _epoch_meta_row(pid: str, w: Window, label: int, epoch_index: int, n_epochs: int, epoch_t_start: float, epoch_t_end: float) -> dict:
    return {
        "participant_id": pid,
        "task": w.task,
        "window_type": w.window_type,
        "block_index": w.block_index,
        "trial_index": w.trial_index,
        "condition_label": w.condition_label,
        "window_t_start": w.t_start,
        "window_t_end": w.t_end,
        "window_duration_s": w.t_end - w.t_start,
        "epoch_index": epoch_index,
        "n_epochs_in_window": n_epochs,
        "t_start": epoch_t_start,
        "t_end": epoch_t_end,
        "label": label,
        "stimulus_json": json.dumps(w.stimulus),
        "labels_json": json.dumps(w.labels),
        "meta_json": json.dumps(w.meta),
    }


def export_all(
    processed_dir: Path,
    export_dir: Path,
    participant_ids: list[str] | None = None,
    streams: list[str] | None = None,
    samples_per_segment: dict[str, int] | None = None,
    epoch_seconds: dict[str, float] | None = None,
    pool: bool = True,
    emotion_only: bool = False,
) -> dict:
    """Writes <export_dir>/<participant_id>/<file_prefix>_{X,y,meta} per
    participant x stream x file key, plus (if pool) pooled/<file_prefix>_*
    concatenated across participants. file_prefix is <key> for the primary
    EEG stream, <key>_<suffix> otherwise.

    emotion_only=True restricts file keys to emotion_trial/emotion_baseline
    (for handing off just the video emotion task before the stress/
    attention label scheme, see PROGRESS.md, is finalized).

    Returns the manifest dict, also written to export_dir/export_manifest.json."""
    processed_dir = Path(processed_dir)
    export_dir = Path(export_dir)
    if participant_ids is None:
        participant_ids = sorted(p.name for p in processed_dir.iterdir() if p.is_dir() and (p / "events.parquet").exists())

    streams = list(streams) if streams is not None else list(DEFAULT_STREAMS)
    samples_per_segment = {**DEFAULT_SAMPLES_PER_SEGMENT, **(samples_per_segment or {})}
    epoch_seconds = {**DEFAULT_EPOCH_SECONDS, **(epoch_seconds or {})}

    windows_by_pid = _gather_windows(processed_dir, participant_ids)

    if emotion_only:
        schulte_median_s = None
    else:
        schulte_durations = [
            w.labels.get("completion_time_sec")
            for windows in windows_by_pid.values()
            for w in windows
            if w.task == "attention_focus_schulte" and w.window_type == "block" and w.labels.get("completion_time_sec") is not None
        ]
        schulte_median_s = float(np.median(schulte_durations)) if schulte_durations else None
    specs = _build_specs(schulte_median_s)
    if emotion_only:
        specs = [s for s in specs if s["key"].startswith("emotion_")]

    file_prefixes = {spec["key"]: {stream: _file_prefix(spec["key"], stream) for stream in streams} for spec in specs}

    manifest = {
        "streams": streams,
        "file_keys": [s["key"] for s in specs],  # sanity_check_npy.py reads this, not BASE_SPECS
        "emotion_only": emotion_only,
        "channels": {s: STREAM_VALUE_COLUMNS[s] for s in streams},
        "samples_per_segment": samples_per_segment,
        "epoch_seconds": epoch_seconds,
        "schulte_median_completion_s": schulte_median_s,
        "lapse_response_buffer_s": LAPSE_RESPONSE_BUFFER_S,
        "min_lapse_window_s": MIN_LAPSE_WINDOW_S,
        "label_maps": LABEL_MAPS,
        "participants": {},
        "pooled": {},
    }

    pooled = {file_prefixes[spec["key"]][stream]: {"X": [], "y": [], "pid": [], "meta": []} for spec in specs for stream in streams}

    for pid in participant_ids:
        windows = windows_by_pid.get(pid, [])
        pdir = export_dir / pid
        pdir.mkdir(parents=True, exist_ok=True)
        participant_report: dict = {}
        if pid not in windows_by_pid:
            participant_report["_warning"] = "no events.parquet found -- every file key gets 0 windows"

        stream_dfs = {}
        for stream in streams:
            path = processed_dir / pid / f"{stream}.parquet"
            stream_dfs[stream] = pd.read_parquet(path) if path.exists() else None

        for spec in specs:
            key = spec["key"]
            epoch_s = epoch_seconds[key]
            matched = [w for w in windows if w.task == spec["task"] and w.window_type == spec["window_type"]]

            # Label + epoch-count planning is stream-independent -- signal
            # availability (below) is checked per stream separately.
            planned, n_label_unresolvable, n_too_short = [], 0, 0
            for w in matched:
                label = spec["label_fn"](w)
                if label is None:
                    n_label_unresolvable += 1
                    continue
                n_epochs, t_end_eff = _epoch_bounds(w.t_start, w.t_end, epoch_s)
                if n_epochs < 1:
                    n_too_short += 1
                    continue
                planned.append((w, label, n_epochs, t_end_eff))

            for stream in streams:
                prefix = file_prefixes[key][stream]
                df = stream_dfs[stream]
                if df is None:
                    participant_report[prefix] = {"error": f"{stream}.parquet not found"}
                    continue

                value_cols = STREAM_VALUE_COLUMNS[stream]
                sps = samples_per_segment.get(stream, SAMPLES_PER_SEGMENT)

                # Each epoch is resampled independently (n_segments=1 over
                # just that epoch's own bounds), NOT as one joint multi-
                # segment call over the whole window then sliced -- a joint
                # linspace across N epochs doesn't decompose into N
                # independent per-epoch grids (endpoint alignment differs
                # by a sub-point amount), which used to make this silently
                # diverge from sanity_check_npy.py's per-row recompute.
                # Per-epoch calls also mean a signal gap only drops the
                # epochs actually missing data, not the whole window.
                X_list, y_list, meta_rows = [], [], []
                n_epochs_no_signal = 0
                for w, label, n_epochs, t_end_eff in planned:
                    for e in range(n_epochs):
                        epoch_t_start = w.t_start + e * epoch_s
                        epoch_t_end = epoch_t_start + epoch_s
                        arr = _segment_resample(df, epoch_t_start, epoch_t_end, value_cols, 1, samples_per_segment=sps)
                        if arr is None:
                            n_epochs_no_signal += 1
                            continue
                        X_list.append(arr[:, 0, :])
                        y_list.append(label)
                        meta_rows.append(_epoch_meta_row(pid, w, label, e, n_epochs, epoch_t_start, epoch_t_end))

                X = np.stack(X_list, axis=0) if X_list else np.zeros((0, len(value_cols), sps), dtype=np.float32)
                y = np.array(y_list, dtype=np.int64) if y_list else np.zeros((0,), dtype=np.int64)
                meta_df = pd.DataFrame(meta_rows, columns=META_COLUMNS)

                np.save(pdir / f"{prefix}_X.npy", X)
                np.save(pdir / f"{prefix}_y.npy", y)
                # %.17g = float64 round-trip precision; pandas' default
                # to_csv/read_csv float handling silently loses the last
                # bit of a ~1.79e9 UTC timestamp otherwise, breaking the
                # bit-exact recompute check in sanity_check_npy.py. Always
                # read this file back with float_precision="round_trip".
                meta_df.to_csv(pdir / f"{prefix}_meta.csv", index=False, float_format="%.17g")

                participant_report[prefix] = {
                    "n_matched_windows": len(matched),
                    "n_windows_planned": len(planned),  # passed label + min-duration filtering
                    "n_epochs_written": len(y_list),
                    "dropped": {
                        "label_unresolvable": n_label_unresolvable,
                        "too_short_for_one_epoch": n_too_short,
                        "no_signal_epochs": n_epochs_no_signal,
                    },
                    "shape": list(X.shape),
                }

                if pool:
                    pooled[prefix]["X"].append(X)
                    pooled[prefix]["y"].append(y)
                    pooled[prefix]["pid"].extend([pid] * len(y_list))
                    pooled[prefix]["meta"].append(meta_df)

        manifest["participants"][pid] = participant_report

    if pool:
        pool_dir = export_dir / "pooled"
        pool_dir.mkdir(parents=True, exist_ok=True)
        for spec in specs:
            key = spec["key"]
            for stream in streams:
                prefix = file_prefixes[key][stream]
                sps = samples_per_segment.get(stream, SAMPLES_PER_SEGMENT)
                value_cols = STREAM_VALUE_COLUMNS[stream]

                chunks = pooled[prefix]["X"]
                X = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, len(value_cols), sps), dtype=np.float32)
                y = np.concatenate(pooled[prefix]["y"], axis=0) if pooled[prefix]["y"] else np.zeros((0,), dtype=np.int64)
                pid_arr = np.array(pooled[prefix]["pid"])
                meta_df = pd.concat(pooled[prefix]["meta"], ignore_index=True) if pooled[prefix]["meta"] else pd.DataFrame(columns=META_COLUMNS)

                np.save(pool_dir / f"{prefix}_X.npy", X)
                np.save(pool_dir / f"{prefix}_y.npy", y)
                np.save(pool_dir / f"{prefix}_participant_ids.npy", pid_arr)
                meta_df.to_csv(pool_dir / f"{prefix}_meta.csv", index=False, float_format="%.17g")

                labels_u, counts_u = (np.unique(y, return_counts=True) if len(y) else (np.array([]), np.array([])))
                manifest["pooled"][prefix] = {
                    "n_total": int(len(y)),
                    "shape": list(X.shape),
                    "label_counts": {int(lbl): int(cnt) for lbl, cnt in zip(labels_u, counts_u)},
                }

    manifest_path = export_dir / "export_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    return manifest


def _parse_kv_int_arg(raw: str | None) -> dict[str, int]:
    if not raw:
        return {}
    return {k.strip(): int(v) for k, v in (part.split("=") for part in raw.split(","))}


def _parse_kv_float_arg(raw: str | None) -> dict[str, float]:
    if not raw:
        return {}
    return {k.strip(): float(v) for k, v in (part.split("=") for part in raw.split(","))}


def main():
    parser = argparse.ArgumentParser(description="Export fixed-shape (N, C, samples_per_epoch) biosignal + label .npy files -- N = total epochs across all matched windows.")
    parser.add_argument("--processed-dir", type=str, required=True, help="dataset/build_dataset.py output dir")
    parser.add_argument("--export-dir", type=str, required=True, help="Where to write <participant_id>/<file_prefix>_{X,y,meta} + pooled/")
    parser.add_argument("--participants", nargs="*", default=None, help="Subset of participant IDs (default: all under --processed-dir)")
    parser.add_argument("--streams", nargs="*", default=None, help=f"Streams to export (default: every device: {DEFAULT_STREAMS})")
    parser.add_argument("--samples-per-segment", type=str, default=None, help='Per-stream samples-per-epoch override, e.g. "wristband.gsr=50"')
    parser.add_argument("--epoch-seconds", type=str, default=None, help='Per-key epoch length override (default 1.0), e.g. "emotion_trial=2,stroop_trial=0.5"')
    parser.add_argument("--no-pool", action="store_true", help="Skip writing pooled/ (all-participants-concatenated)")
    parser.add_argument("--emotion-only", action="store_true", help="Export only emotion_trial/emotion_baseline")
    args = parser.parse_args()

    manifest = export_all(
        Path(args.processed_dir),
        Path(args.export_dir),
        participant_ids=args.participants,
        streams=args.streams,
        samples_per_segment=_parse_kv_int_arg(args.samples_per_segment),
        epoch_seconds=_parse_kv_float_arg(args.epoch_seconds),
        pool=not args.no_pool,
        emotion_only=args.emotion_only,
    )
    n_ok = sum(1 for r in manifest["participants"].values() if "error" not in r)
    n_err = len(manifest["participants"]) - n_ok
    print(f"Exported {n_ok} participant(s), {n_err} skipped (see export_manifest.json). Pooled totals:")
    for prefix, info in sorted(manifest["pooled"].items()):
        print(f"  {prefix:24s} shape={tuple(info['shape'])}  label_counts={info['label_counts']}")


if __name__ == "__main__":
    main()
