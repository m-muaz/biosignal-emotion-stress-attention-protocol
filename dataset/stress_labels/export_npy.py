"""Phase 4 of plans/stress_task.md: turns a Dataset-Version sample table
(dataset/stress_labels/versions.py) into fixed-shape EEG `.npy` epochs,
mirroring dataset/export_npy.py's conventions for the emotion task:

    X.shape == (N, C, samples_per_epoch)   float32
    y.shape == (N,)                        int64 (model_target, may be -1/NaN-coded if non-numeric)

plus a `<version_key>_meta.csv` (same row order) carrying every cond_*/
beh_*/subj_*/derived_*/meta_* label column from the version's sample table
untouched (plan §48: never collapse into one `label`), plus this row's own
epoch bounds/index, plus every per-channel EEG QC metric from
dataset/eeg_qc.py.

EEG-only: only `ear_eeg_out.ads1299` (mandatory out-ear device) is
converted (ADC->uV), filtered, and QC-scored here, via
dataset/eeg_preprocess.py. Nothing is auto-excluded for failing QC --
QC columns are written for every epoch; filtering on them is a downstream,
config-driven choice (per project convention agreed 2026-08-24).

Window bounds per sample-table kind (plan §5-13):
    block-level (S0/S1 versions):        block_start / block_end
    raindrop trial (S4):                 trial_start / response_timestamp
    highway obstacle (S5):                spawn_timestamp / resolve_timestamp
    highway self-report (S6):             prompt_timestamp / response_timestamp
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataset.eeg_preprocess import CHANNELS, NATIVE_FS, convert_and_filter
from dataset.eeg_qc import kurtosis_reference_from_epochs, score_epoch
from dataset.export_npy import _epoch_bounds
from dataset.stress_labels.versions import VERSION_BUILDERS, load_all_canonical_tables

DEFAULT_EPOCH_S = 1.0
SAMPLES_PER_EPOCH = 200  # matches dataset/export_npy.py's EEG convention (native 250Hz, resampled to 200/epoch)

# Window-bound column pair per sample-table kind, keyed by which columns are
# present on that version's sample rows (versions differ in table shape).
WINDOW_BOUND_CANDIDATES = [
    ("block_start", "block_end"),
    ("trial_start", "response_timestamp"),
    ("spawn_timestamp", "resolve_timestamp"),
    ("prompt_timestamp", "response_timestamp"),
]

META_LABEL_PREFIXES = ("cond_", "beh_", "subj_", "derived_", "meta_")

QC_SUMMARY_COLUMNS = [
    "qc_n_checks_passed_min_over_channels",
    "qc_n_checks_total",
    "qc_clip_frac_max",
    "qc_dc_drift_uv_max",
    "qc_emg_hf_ratio_max",
    "qc_line_noise_ratio_max",
    "qc_snr_db_min",
]


def _window_bounds_col_pair(samples: pd.DataFrame) -> tuple[str, str]:
    for start_col, end_col in WINDOW_BOUND_CANDIDATES:
        if start_col in samples.columns and end_col in samples.columns:
            return start_col, end_col
    raise ValueError(f"no recognized window-bound column pair found among {list(samples.columns)}")


def _label_columns(samples: pd.DataFrame) -> list[str]:
    return [c for c in samples.columns if c.startswith(META_LABEL_PREFIXES) or c in ("model_target", "subjective_labels_available", "label_resolution")]


def _qc_summary_row(epoch_qc: dict) -> dict:
    channels = epoch_qc["channels"]
    return {
        "qc_n_checks_passed_min_over_channels": epoch_qc["n_checks_passed_min_over_channels"],
        "qc_n_checks_total": epoch_qc["n_checks_total"],
        "qc_clip_frac_max": max(c["clip_frac"] for c in channels),
        "qc_dc_drift_uv_max": max(c["dc_drift_uv"] for c in channels),
        "qc_emg_hf_ratio_max": max(c["emg_hf_ratio"] for c in channels if not np.isnan(c["emg_hf_ratio"])) if channels else float("nan"),
        "qc_line_noise_ratio_max": max(c["line_noise_ratio"] for c in channels if not np.isnan(c["line_noise_ratio"])) if channels else float("nan"),
        "qc_snr_db_min": min(c["snr_db"] for c in channels) if channels else float("nan"),
        "qc_channels_json": json.dumps(channels),
    }


def _slice_uv(df: pd.DataFrame, t_start: float, t_end: float) -> np.ndarray | None:
    """Returns (n_channels, n_samples) at native resolution within
    [t_start, t_end], or None if fewer than 2 samples are present."""
    wall = df["wall_utc_s"].to_numpy()
    mask = (wall >= t_start) & (wall <= t_end)
    if mask.sum() < 2:
        return None
    return df.loc[mask, CHANNELS].to_numpy(dtype=np.float64).T, wall[mask]


def _resample_epoch(vals: np.ndarray, src_wall: np.ndarray, t_start: float, t_end: float, n_samples: int) -> np.ndarray:
    grid = np.linspace(0.0, t_end - t_start, n_samples)
    src_t = src_wall - t_start
    out = np.empty((vals.shape[0], n_samples), dtype=np.float64)
    for c in range(vals.shape[0]):
        out[c] = np.interp(grid, src_t, vals[c])
    return out


def build_eeg_epochs_for_version(
    processed_dir: Path,
    version_key: str,
    samples: pd.DataFrame,
    epoch_s: float = DEFAULT_EPOCH_S,
    samples_per_epoch: int = SAMPLES_PER_EPOCH,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Cuts every sample row's window into epoch_s-second epochs of the
    out-ear EEG stream, converted to uV and filtered (once per participant,
    continuous, before epoching -- see dataset/eeg_preprocess.py). Returns
    (X of shape (N, 8, samples_per_epoch), meta_df aligned row-for-row).

    Two passes per participant: the kurtosis-outlier QC check needs a
    reference population (mean/std of kurtosis) to z-score against, and
    that reference should be built from this participant's OWN epochs
    (electrode contact quality is participant-specific) -- so every epoch
    must be built once before any of them can be scored. See
    dataset/eeg_qc.py::kurtosis_reference_from_epochs."""
    if samples.empty:
        return np.zeros((0, len(CHANNELS), samples_per_epoch), dtype=np.float32), pd.DataFrame()

    start_col, end_col = _window_bounds_col_pair(samples)
    label_cols = _label_columns(samples)

    X_list, meta_rows = [], []
    for pid, pid_samples in samples.groupby("participant_id"):
        eeg_path = Path(processed_dir) / pid / "ear_eeg_out.ads1299.parquet"
        if not eeg_path.exists():
            continue
        raw_df = pd.read_parquet(eeg_path)
        uv_raw, uv_filt = convert_and_filter(raw_df, CHANNELS, NATIVE_FS)

        # Pass 1: build every epoch for this participant, deferring QC scoring.
        pending = []  # (raw_epoch, filt_epoch, partial_meta_row)
        for _, row in pid_samples.iterrows():
            t_start, t_end = float(row[start_col]), float(row[end_col])
            n_epochs, _ = _epoch_bounds(t_start, t_end, epoch_s)
            for e in range(n_epochs):
                epoch_t0, epoch_t1 = t_start + e * epoch_s, t_start + (e + 1) * epoch_s

                raw_slice = _slice_uv(uv_raw, epoch_t0, epoch_t1)
                filt_slice = _slice_uv(uv_filt, epoch_t0, epoch_t1)
                if raw_slice is None or filt_slice is None:
                    continue
                raw_vals, raw_wall = raw_slice
                filt_vals, filt_wall = filt_slice

                raw_epoch = _resample_epoch(raw_vals, raw_wall, epoch_t0, epoch_t1, samples_per_epoch)
                filt_epoch = _resample_epoch(filt_vals, filt_wall, epoch_t0, epoch_t1, samples_per_epoch)

                meta_row = {
                    "participant_id": pid,
                    "version": version_key,
                    "window_t_start": t_start,
                    "window_t_end": t_end,
                    "epoch_index": e,
                    "n_epochs_in_window": n_epochs,
                    "t_start": epoch_t0,
                    "t_end": epoch_t1,
                }
                for col in label_cols:
                    meta_row[col] = row[col]
                pending.append((raw_epoch, filt_epoch, meta_row))

        kurtosis_ref = kurtosis_reference_from_epochs([filt_epoch for _, filt_epoch, _ in pending])

        # Pass 2: score every epoch against this participant's own reference.
        for raw_epoch, filt_epoch, meta_row in pending:
            epoch_qc = score_epoch(raw_epoch, filt_epoch, fs=samples_per_epoch / epoch_s, kurtosis_ref=kurtosis_ref)
            X_list.append(filt_epoch.astype(np.float32))
            meta_row.update(_qc_summary_row(epoch_qc))
            meta_rows.append(meta_row)

    X = np.stack(X_list, axis=0) if X_list else np.zeros((0, len(CHANNELS), samples_per_epoch), dtype=np.float32)
    meta_df = pd.DataFrame(meta_rows)
    return X, meta_df


def export_version(
    processed_dir: Path,
    export_dir: Path,
    version_key: str,
    participant_ids: list[str] | None = None,
    epoch_s: float = DEFAULT_EPOCH_S,
    samples_per_epoch: int = SAMPLES_PER_EPOCH,
    version_kwargs: dict | None = None,
) -> dict:
    """Builds one Dataset Version's EEG epochs and writes
    <export_dir>/<version_key>_{X,y,meta}.{npy,npy,csv} (pooled across all
    participants -- per-participant splitting is Phase 5, done downstream
    via dataset/stress_labels/splits.py on the meta table's participant_id
    column, not re-derived here)."""
    processed_dir, export_dir = Path(processed_dir), Path(export_dir)
    if version_key not in VERSION_BUILDERS:
        raise KeyError(f"unknown version_key {version_key!r}; available: {list(VERSION_BUILDERS)}")

    if participant_ids is None:
        participant_ids = sorted(p.name for p in processed_dir.iterdir() if p.is_dir() and (p / "events.parquet").exists())

    tables = load_all_canonical_tables(processed_dir, participant_ids)
    samples = VERSION_BUILDERS[version_key](tables, **(version_kwargs or {}))

    X, meta_df = build_eeg_epochs_for_version(processed_dir, version_key, samples, epoch_s, samples_per_epoch)

    target_col = "model_target" if "model_target" in meta_df.columns else None
    if target_col is not None and not meta_df.empty and pd.api.types.is_numeric_dtype(meta_df[target_col].dropna()):
        y = meta_df[target_col].fillna(-1).to_numpy(dtype=np.int64)
    else:
        y = np.full(len(meta_df), -1, dtype=np.int64)  # non-numeric/continuous targets stay in meta.csv only

    export_dir.mkdir(parents=True, exist_ok=True)
    np.save(export_dir / f"{version_key}_X.npy", X)
    np.save(export_dir / f"{version_key}_y.npy", y)
    meta_df.to_csv(export_dir / f"{version_key}_meta.csv", index=False, float_format="%.17g")

    return {
        "version_key": version_key,
        "n_epochs": int(X.shape[0]),
        "shape": list(X.shape),
        "participants": sorted(meta_df["participant_id"].unique().tolist()) if not meta_df.empty else [],
        "epoch_seconds": epoch_s,
        "samples_per_epoch": samples_per_epoch,
    }


def export_config(processed_dir: Path, export_dir: Path, config_name: str, epoch_s: float = DEFAULT_EPOCH_S, samples_per_epoch: int = SAMPLES_PER_EPOCH) -> dict:
    """Same as export_version, but reads version_key + exclude_highway_pre_fix
    from a configs/stress_labels/<config_name>.yaml file (Phase 6) instead of
    passing them directly -- lets a colleague export a variant's EEG epochs
    without touching Python."""
    from dataset.stress_labels.config import load_version_config

    config = load_version_config(config_name)
    if config.version_builder not in VERSION_BUILDERS:
        raise KeyError(f"config {config_name!r} references unknown version_builder {config.version_builder!r}; available: {list(VERSION_BUILDERS)}")
    version_kwargs = {"exclude_highway_pre_fix": config.exclude_highway_pre_fix} if "highway" in config.version_builder else {}
    return export_version(processed_dir, export_dir, config.version_builder, epoch_s=epoch_s, samples_per_epoch=samples_per_epoch, version_kwargs=version_kwargs)


def main():
    import argparse

    from dataset.stress_labels.config import discover_configs

    parser = argparse.ArgumentParser(description="Export one config-driven stress-task Dataset Version to fixed-shape EEG .npy epochs (Phase 4).")
    parser.add_argument("--processed-dir", type=str, required=True)
    parser.add_argument("--export-dir", type=str, required=True)
    parser.add_argument("--config", type=str, nargs="+", default=None, choices=discover_configs(), help="one or more config names under configs/stress_labels/")
    parser.add_argument("--all", action="store_true", help="export every available config")
    parser.add_argument("--epoch-seconds", type=float, default=DEFAULT_EPOCH_S)
    parser.add_argument("--samples-per-epoch", type=int, default=SAMPLES_PER_EPOCH)
    args = parser.parse_args()

    if args.all:
        config_names = discover_configs()
    elif args.config:
        config_names = args.config
    else:
        print("No --config specified (and --all not passed) -- nothing to export.\nAvailable configs:")
        for name in discover_configs():
            print(f"  {name}")
        raise SystemExit(1)

    for name in config_names:
        report = export_config(Path(args.processed_dir), Path(args.export_dir), name, epoch_s=args.epoch_seconds, samples_per_epoch=args.samples_per_epoch)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
