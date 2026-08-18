"""Fixed-shape `.npy` export on top of the event-windowed dataset
(`dataset/windows.py`, `dataset/torch_dataset.py`), for a downstream
collaborator's own DL/ML loader.

Produces, per (task, window_kind) "file key" x stream, an `{X.npy, y.npy}`
pair:
    X.shape == (B, C, T, samples_per_segment)   float32
    y.shape == (B,)                             int64
plus a `<file_prefix>_meta.csv` (same row order as X/y) carrying every raw
participant-response field, so nothing is lost even though y is a small int.

`C`/`samples_per_segment` depend on the stream (8ch/200 for the default EEG
stream, fewer channels and a stream-appropriate rate for wristband
modalities); `T` (segment count) is a property of the file key (task
duration), shared across every stream exported for that key.

See docs/Dataset_Sync_Design.md §8 for the full design writeup (why these
label/segment-count choices, what they mirror from Mumtaz2016/FACED/
MentalArithmetic/PhysioNet-MI-style benchmark datasets) and §9 for the
attentional-lapse windows and multi-stream export added 2026-08-18.

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

SAMPLES_PER_SEGMENT = 200  # default/fallback for any stream not listed in DEFAULT_SAMPLES_PER_SEGMENT
DEFAULT_EEG_STREAM = "ear_eeg_out.ads1299"  # mandatory device, present for every participant;
# ear_eeg_in.ads1299 is optional/opt-outable and frequently missing -- see
# docs/Dataset_Sync_Design.md. Pass it via `streams=` to export that instead
# (participants who opted out simply get an "error" entry in the manifest
# for that stream's files, never silently skipped).

# ── which streams to export, and at what rate ────────────────────────────
# Wristband modalities have very different native rates/channel counts than
# EEG (see docs/Dataset_Sync_Design.md §1), so forcing all of them through
# EEG's 200-samples/segment default would either fabricate data (upsampling
# 1Hz temperature/env sensors to 200 points) or pointlessly oversample.
# `samples_per_segment` is configurable per stream -- these defaults are
# each stream's own native rate (round(native_hz)), so by default nothing
# is meaningfully up/down-sampled; override via `samples_per_segment={...}`
# / `--samples-per-segment stream=N,...` for a specific target Hz instead.
WRISTBAND_STREAMS = ["wristband.ppg", "wristband.imu", "wristband.gsr", "wristband.mag", "wristband.mlx", "wristband.bme"]

DEFAULT_SAMPLES_PER_SEGMENT = {
    "ear_eeg_out.ads1299": 200,  # native 250Hz, downsampled to 200 (the shape the collaborator asked for)
    "ear_eeg_in.ads1299": 200,
    "wristband.ppg": 200,  # native 200Hz
    "wristband.imu": 200,  # native 200Hz
    "wristband.gsr": 200,  # native 200Hz
    "wristband.mag": 100,  # native 100Hz
    "wristband.mlx": 1,  # native 1Hz -- upsampling to 200 would fabricate data, not resample it
    "wristband.bme": 1,  # native 1Hz
}

# Short suffix for non-default streams' filenames (<key>_<suffix>_X.npy).
# The primary EEG stream stays unsuffixed (<key>_X.npy) for backward
# compatibility with the original single-stream export.
STREAM_SHORT_NAME = {
    DEFAULT_EEG_STREAM: None,
    "wristband.ppg": "ppg",
    "wristband.imu": "imu",
    "wristband.gsr": "gsr",
    "wristband.mag": "mag",
    "wristband.mlx": "mlx",
    "wristband.bme": "bme",
}


def _file_prefix(key: str, stream: str) -> str:
    suffix = STREAM_SHORT_NAME.get(stream, stream.replace(".", "_"))
    return key if suffix is None else f"{key}_{suffix}"


# ── segment-count (T) defaults ───────────────────────────────────────────
# Computed 2026-08-17 as round(median real duration in seconds) per
# (task, window_type) across all 12 real participants (P001-P012) --
# real-data-derived, not guessed. One segment ~= 1 real second at this
# default, i.e. each segment's grid is close to a straight resample of that
# segment's native samples -- not heavily over/under-sampled. Override per
# key via `segments={...}` / `--segments key=N,...` if a different T is
# needed; T is otherwise a free parameter -- every window in
# [t_start, t_end] is divided into exactly T equal-width segments
# regardless of its own real duration, each linearly interpolated onto its
# own samples_per_segment-point grid, so real seconds/segment varies by
# window/file but every file's X.npy still stacks into one rectangular
# (B, C, T, samples_per_segment) array.
DEFAULT_SEGMENTS = {
    "emotion_trial": 68,
    "emotion_baseline": 15,
    "math_trial": 60,
    "math_baseline": 10,
    "highway_trial": 61,
    "highway_baseline": 5,
    "stroop_trial": 1,
    "stroop_baseline": 10,
    "schulte_trial": 26,
    "schulte_baseline": 10,
    "sart_trial": 1,
    "sart_baseline": 30,
    # Lapse windows are truncated sub-spans of stroop_trial/sart_trial (see
    # _build_*_lapse_windows below) -- still short single-epoch spans, same
    # T=1 as their parent trial windows.
    "sart_lapse_trial": 1,
    "stroop_lapse_trial": 1,
}

# ── label encoding ────────────────────────────────────────────────────────
# 0 is reserved for baseline/rest in EVERY file key, so concatenating a
# task's own "<key>_baseline" y with its "<key>_trial" y reproduces the
# rest(0)/condition(1..) convention used by e.g. MentalArithmetic/
# PhysioNet-MI, without the caller having to renumber anything.
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
    # No baseline sibling for the lapse specs -- they're inherently binary
    # attentive/lapse, not a task-vs-rest contrast; use sart_baseline/
    # stroop_baseline (same task) as the rest reference if one is needed.
    "sart_lapse_trial": {0: "attentive_correct_withhold", 1: "lapse_commission_error"},
    "stroop_lapse_trial": {0: "attentive_correct_response", 1: "lapse_incorrect_response"},
}


def _label_baseline(w: Window):
    return 0


def _label_emotion_trial(w: Window):
    return {"negative": 1, "neutral": 2, "positive": 3}.get(w.condition_label)


def _label_tier_block(w: Window):
    """Shared by math (stress/raindrop) and highway blocks -- both store
    the tier (1/2/3) in `meta["tier"]`."""
    return {1: 1, 2: 2, 3: 3}.get(w.meta.get("tier"))


def _label_stroop_trial(w: Window):
    cong = w.stimulus.get("congruent")
    return None if cong is None else (2 if cong else 1)


def _label_sart_trial(w: Window):
    is_omit = w.stimulus.get("is_omit")
    return None if is_omit is None else (2 if is_omit else 1)


def _make_schulte_label_fn(median_completion_s: float):
    """Schulte has no natural multi-class condition (single 5x5 grid, no
    difficulty tiers) -- fall back to a performance-based median split
    (faster/slower than the cohort-wide median completion time), in the
    spirit of Mumtaz2016 MentalArithmetic's "good/bad counter" framing.
    The threshold is a single cohort-wide constant (computed once up front,
    same for every participant/file) so per-participant labels stay
    comparable -- NOT an independent per-participant median.

    Caveat (flagged 2026-08-18, not changed without asking first): this
    label is a direct function of the window's OWN real duration
    (completion_time_sec IS t_end - t_start for this window) -- the
    sharpest form of circularity of any label here. Resampling to a fixed
    T=26 segments means the network never sees the literal duration number
    as a feature, but subtle "how much real time got compressed into this
    grid" correlates could still leak in. `misclicks` (a count, independent
    of duration) in the same labels_json is a cleaner alternative if this
    label is ever tightened."""

    def _label(w: Window):
        ct = w.labels.get("completion_time_sec")
        if ct is None:
            return None
        return 2 if ct < median_completion_s else 1

    return _label


def _label_lapse(w: Window):
    """Shared by both lapse specs below -- the lapse builders already
    compute the 0/1 label and stash it in `meta["lapse_label"]`."""
    return w.meta.get("lapse_label")


# key, task, window_type, label_fn (schulte_trial's is filled in at export time,
# once the cohort-wide median completion time is known)
BASE_SPECS = [
    dict(key="emotion_trial", task="emotion", window_type="trial", label_fn=_label_emotion_trial),
    dict(key="emotion_baseline", task="emotion", window_type="baseline", label_fn=_label_baseline),
    dict(key="math_trial", task="stress", window_type="block", label_fn=_label_tier_block),
    dict(key="math_baseline", task="stress", window_type="baseline", label_fn=_label_baseline),
    dict(key="highway_trial", task="attention_highway", window_type="block", label_fn=_label_tier_block),
    dict(key="highway_baseline", task="attention_highway", window_type="baseline", label_fn=_label_baseline),
    dict(key="stroop_trial", task="attention_focus_stroop", window_type="trial", label_fn=_label_stroop_trial),
    dict(key="stroop_baseline", task="attention_focus_stroop", window_type="baseline", label_fn=_label_baseline),
    dict(key="schulte_trial", task="attention_focus_schulte", window_type="block", label_fn=None),
    dict(key="schulte_baseline", task="attention_focus_schulte", window_type="baseline", label_fn=_label_baseline),
    dict(key="sart_trial", task="sart", window_type="trial", label_fn=_label_sart_trial),
    dict(key="sart_baseline", task="sart", window_type="baseline", label_fn=_label_baseline),
    dict(key="sart_lapse_trial", task="sart", window_type="lapse", label_fn=_label_lapse),
    dict(key="stroop_lapse_trial", task="attention_focus_stroop", window_type="lapse", label_fn=_label_lapse),
]

# ── attentional-lapse windows (added 2026-08-18) ─────────────────────────
# Both truncate the epoch to end BEFORE the participant's actual response,
# so a "did they lapse" classifier can't just read the motor
# response/error-related-negativity that defines its own label -- see
# docs/Dataset_Sync_Design.md §9 for the full reasoning (circularity/
# reverse-causation discussion) and the real-data check that motivated the
# specific truncation math per task.
LAPSE_RESPONSE_BUFFER_S = 0.15  # excludes premotor readiness-potential + the keypress itself (conservative; movement-prep onset is commonly reported ~200ms+ before an overt response)
MIN_LAPSE_WINDOW_S = 0.2  # drop a trial if truncation would leave less than this much pre-response signal


def _build_sart_lapse_windows(windows: list[Window]) -> list[Window]:
    """Commission-error attentional lapse (Robertson et al., 1997's SART
    definition): scoped to NO-GO trials only (is_omit==True) -- responding
    on one is the field's standard operational definition of a lapse in
    sustained attention; correctly withholding is "attentive".

    Truncation: SART's own `response` event fires at a ~1.157-1.16s fixed
    trial-slot boundary REGARDLESS of when the actual keypress happened
    (confirmed against real P007 data -- the gap from trial_start to the
    response event stays ~1.157-1.16s across observed `rt` from 0.41s to
    1.06s). The real keypress instant is `trial_start + rt`, not the
    `response` event's own timestamp -- so for trials where the participant
    DID respond (the commission-error/lapse case), the window is truncated
    to `t_start + rt - LAPSE_RESPONSE_BUFFER_S`. Trials with no keypress
    (correctly withheld) keep their full original span -- no motor event
    occurred at all, so there's nothing to truncate away."""
    out = []
    for w in windows:
        if w.task != "sart" or w.window_type != "trial":
            continue
        if not w.stimulus.get("is_omit"):
            continue  # scope: no-go trials only, per the commission-error definition
        responded = w.labels.get("responded")
        rt = w.labels.get("rt")
        if responded:
            if rt is None:
                continue
            t_end = w.t_start + rt - LAPSE_RESPONSE_BUFFER_S
            if t_end - w.t_start < MIN_LAPSE_WINDOW_S:
                continue  # responded too fast to leave a usable pre-response window
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
    """Error-based attentional lapse for Stroop: any incorrect response
    (regardless of congruency) is treated as a lapse of attentional/
    cognitive control -- an error on an easy congruent trial is if
    anything a stronger tell than an incongruent-trial error.

    Truncation: unlike SART, Stroop's `response` event IS the actual click
    instant (confirmed against real P007 data -- its timestamp matches
    `rt_ms` to within measurement noise, unlike SART's fixed-slot
    behavior). Still truncated to `t_start + rt_ms/1000 - LAPSE_RESPONSE_BUFFER_S`
    so the epoch excludes the click/motor response itself, not just
    whatever followed it."""
    out = []
    for w in windows:
        if w.task != "attention_focus_stroop" or w.window_type != "trial":
            continue
        rt_ms = w.labels.get("rt_ms")
        correct = w.labels.get("correct")
        if rt_ms is None or correct is None:
            continue
        t_end = w.t_start + (rt_ms / 1000.0) - LAPSE_RESPONSE_BUFFER_S
        if t_end - w.t_start < MIN_LAPSE_WINDOW_S:
            continue  # responded too fast to leave a usable pre-response window
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
    """Divides [t_start, t_end] into `n_segments` equal-width segments and
    linearly interpolates the whole window's samples onto one
    `n_segments * samples_per_segment`-point grid (equivalent to resampling
    each segment separately onto its own `samples_per_segment`-point grid,
    since the segments are contiguous, non-overlapping, and equal-width),
    then reshapes to `(len(value_cols), n_segments, samples_per_segment)`.

    Same on-the-fly linear-interpolation approach as
    `dataset/torch_dataset.py`'s `_slice_and_resample` -- just reshaped into
    segments instead of one flat time axis. Returns None if the window has
    zero samples of this stream at all (e.g. participant opted out of this
    device, or a genuine sensor gap)."""
    wall = df["wall_utc_s"].to_numpy()
    mask = (wall >= t_start) & (wall <= t_end)
    if not mask.any():
        return None
    sub_wall = wall[mask] - t_start
    sub_vals = df.loc[mask, value_cols].to_numpy(dtype=np.float64)

    n_total = n_segments * samples_per_segment
    grid = np.linspace(0.0, t_end - t_start, n_total)
    if sub_wall.size < 2:
        # can't interpolate from a single sample -- hold it constant across the window
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
    """Every participant's windows, PLUS the derived attentional-lapse
    windows appended on top (window_type="lapse", not something
    extract_all_windows itself knows how to produce)."""
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


META_COLUMNS = [
    "participant_id", "task", "window_type", "block_index", "trial_index", "condition_label",
    "t_start", "t_end", "duration_s", "label", "stimulus_json", "labels_json", "meta_json",
]


def _window_meta_row(pid: str, w: Window, label: int) -> dict:
    return {
        "participant_id": pid,
        "task": w.task,
        "window_type": w.window_type,
        "block_index": w.block_index,
        "trial_index": w.trial_index,
        "condition_label": w.condition_label,
        "t_start": w.t_start,
        "t_end": w.t_end,
        "duration_s": w.t_end - w.t_start,
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
    segments: dict[str, int] | None = None,
    pool: bool = True,
) -> dict:
    """Writes `<export_dir>/<participant_id>/<file_prefix>_{X,y,meta}.{npy,npy,csv}`
    for every participant x stream x file key, plus (if `pool`)
    `<export_dir>/pooled/<file_prefix>_*` concatenated across every
    participant (with a companion `<file_prefix>_participant_ids.npy` so
    the pooled arrays stay traceable/splittable). `file_prefix` is `<key>`
    for the primary EEG stream (backward-compatible, unsuffixed) and
    `<key>_<stream_suffix>` for every other stream (e.g. `emotion_trial_ppg`).
    Returns the manifest dict also written to `export_dir/export_manifest.json`."""
    processed_dir = Path(processed_dir)
    export_dir = Path(export_dir)
    if participant_ids is None:
        participant_ids = sorted(p.name for p in processed_dir.iterdir() if p.is_dir() and (p / "events.parquet").exists())

    streams = list(streams) if streams is not None else [DEFAULT_EEG_STREAM] + WRISTBAND_STREAMS
    samples_per_segment = {**DEFAULT_SAMPLES_PER_SEGMENT, **(samples_per_segment or {})}
    segments = {**DEFAULT_SEGMENTS, **(segments or {})}

    windows_by_pid = _gather_windows(processed_dir, participant_ids)

    # Schulte's label needs a cohort-wide median completion time, computed
    # once up front across every participant before any labels are assigned.
    schulte_durations = [
        w.labels.get("completion_time_sec")
        for windows in windows_by_pid.values()
        for w in windows
        if w.task == "attention_focus_schulte" and w.window_type == "block" and w.labels.get("completion_time_sec") is not None
    ]
    schulte_median_s = float(np.median(schulte_durations)) if schulte_durations else None
    specs = _build_specs(schulte_median_s)

    file_prefixes = {spec["key"]: {stream: _file_prefix(spec["key"], stream) for stream in streams} for spec in specs}

    manifest = {
        "streams": streams,
        "channels": {s: STREAM_VALUE_COLUMNS[s] for s in streams},
        "samples_per_segment": samples_per_segment,
        "segments": segments,
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
            participant_report["_warning"] = "no events.parquet found for this participant -- every file key gets 0 windows"

        stream_dfs = {}
        for stream in streams:
            path = processed_dir / pid / f"{stream}.parquet"
            stream_dfs[stream] = pd.read_parquet(path) if path.exists() else None

        for spec in specs:
            key = spec["key"]
            T = segments[key]
            matched = [w for w in windows if w.task == spec["task"] and w.window_type == spec["window_type"]]

            # Label resolution is stream-independent -- do it once per
            # (participant, file key), then just check signal availability
            # per stream below (e.g. wristband could have a gap where EEG
            # doesn't, or vice versa).
            labeled, n_label_unresolvable = [], 0
            for w in matched:
                label = spec["label_fn"](w)
                if label is None:
                    n_label_unresolvable += 1
                else:
                    labeled.append((w, label))

            for stream in streams:
                prefix = file_prefixes[key][stream]
                df = stream_dfs[stream]
                if df is None:
                    participant_report[prefix] = {"error": f"{stream}.parquet not found -- skipped for this participant/stream"}
                    continue

                value_cols = STREAM_VALUE_COLUMNS[stream]
                sps = samples_per_segment.get(stream, SAMPLES_PER_SEGMENT)

                X_list, y_list, meta_rows = [], [], []
                n_no_signal = 0
                for w, label in labeled:
                    arr = _segment_resample(df, w.t_start, w.t_end, value_cols, T, samples_per_segment=sps)
                    if arr is None:
                        n_no_signal += 1
                        continue
                    X_list.append(arr)
                    y_list.append(label)
                    meta_rows.append(_window_meta_row(pid, w, label))

                X = np.stack(X_list, axis=0) if X_list else np.zeros((0, len(value_cols), T, sps), dtype=np.float32)
                y = np.array(y_list, dtype=np.int64) if y_list else np.zeros((0,), dtype=np.int64)
                meta_df = pd.DataFrame(meta_rows, columns=META_COLUMNS)

                np.save(pdir / f"{prefix}_X.npy", X)
                np.save(pdir / f"{prefix}_y.npy", y)
                # float_format="%.17g" -- pandas' default to_csv float
                # formatting loses precision on t_start/t_end (confirmed: last
                # digit of a ~1.79e9 UTC timestamp silently rounds off), which
                # shifts the resample grid enough to make a recomputed row
                # fail to bit-match its stored .npy row. %.17g is float64's
                # full round-trip precision. Read it back with
                # pd.read_csv(..., float_precision="round_trip") -- pandas'
                # DEFAULT csv float PARSER also loses precision, separately.
                meta_df.to_csv(pdir / f"{prefix}_meta.csv", index=False, float_format="%.17g")

                participant_report[prefix] = {
                    "n_matched_windows": len(matched),
                    "n_written": len(y_list),
                    "dropped": {"no_signal_samples_in_window": n_no_signal, "label_unresolvable": n_label_unresolvable},
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
            T = segments[key]
            for stream in streams:
                prefix = file_prefixes[key][stream]
                sps = samples_per_segment.get(stream, SAMPLES_PER_SEGMENT)
                value_cols = STREAM_VALUE_COLUMNS[stream]

                chunks = pooled[prefix]["X"]
                X = np.concatenate(chunks, axis=0) if chunks else np.zeros((0, len(value_cols), T, sps), dtype=np.float32)
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
    out = {}
    for part in raw.split(","):
        k, v = part.split("=")
        out[k.strip()] = int(v)
    return out


def main():
    parser = argparse.ArgumentParser(description="Export fixed-shape (B, C, T, samples_per_segment) biosignal + label .npy files from the canonical synced dataset.")
    parser.add_argument("--processed-dir", type=str, required=True, help="dataset/build_dataset.py output dir")
    parser.add_argument("--export-dir", type=str, required=True, help="Where to write <participant_id>/<file_prefix>_{X,y,meta} + pooled/")
    parser.add_argument("--participants", nargs="*", default=None, help="Subset of participant IDs (default: every one found under --processed-dir)")
    parser.add_argument("--streams", nargs="*", default=None, help=f"Which streams to export (default: EEG + all wristband modalities: {[DEFAULT_EEG_STREAM] + WRISTBAND_STREAMS})")
    parser.add_argument("--samples-per-segment", type=str, default=None, help='Per-stream samples/segment override, e.g. "wristband.gsr=50"')
    parser.add_argument("--segments", type=str, default=None, help='Per-key T override, e.g. "emotion_trial=10,stroop_trial=3"')
    parser.add_argument("--no-pool", action="store_true", help="Skip writing the pooled/ (all-participants-concatenated) files")
    args = parser.parse_args()

    manifest = export_all(
        Path(args.processed_dir),
        Path(args.export_dir),
        participant_ids=args.participants,
        streams=args.streams,
        samples_per_segment=_parse_kv_int_arg(args.samples_per_segment),
        segments=_parse_kv_int_arg(args.segments),
        pool=not args.no_pool,
    )
    n_ok = sum(1 for r in manifest["participants"].values() if "error" not in r)
    n_err = len(manifest["participants"]) - n_ok
    print(f"Exported {n_ok} participant(s), {n_err} skipped (see export_manifest.json). Pooled totals:")
    for prefix, info in sorted(manifest["pooled"].items()):
        print(f"  {prefix:24s} shape={tuple(info['shape'])}  label_counts={info['label_counts']}")


if __name__ == "__main__":
    main()
