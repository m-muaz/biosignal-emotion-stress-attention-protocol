"""Phase 4 of plans/attention_task.md: turns one config-driven Dataset
Version (dataset.attention_labels.config + .versions) into fixed-shape EEG
`.npy` epochs, reusing dataset/stress_labels/export_npy.py's exact
conventions (which themselves mirror dataset/export_npy.py's emotion-task
export):

    X.shape == (N, C, samples_per_epoch)   float32
    y.shape == (N,)                        int64 (model_target, -1-coded if absent/non-numeric)

plus a `<version_key>_meta.csv` (same row order) carrying every cond_*/
beh_*/derived_*/meta_* label column untouched, this row's own epoch
bounds/index, and every per-channel EEG QC metric from dataset/eeg_qc.py.

What's new here vs. the stress-task exporter: window [t_start, t_end]
bounds are NOT already columns on the sample table for every version (SART/
Stroop trial rows only carry a single `stimulus_timestamp`; Schulte clicks
only a `click_timestamp`) -- they're computed from the version's YAML
config (`windowing.mode` + `pre_sec`/`post_sec`, or `start_col`/`end_col`
for block mode), per plan §38-41. Prospective/pre-click windows are
asserted (not just documented) to end at-or-before the event they predict
(Rule 23, plan §67).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataset.attention_labels.config import DatasetVersionConfig, WindowSpec, load_version_config
from dataset.attention_labels.versions import VERSION_BUILDERS, load_all_canonical_tables
from dataset.eeg_preprocess import CHANNELS, NATIVE_FS, convert_and_filter
from dataset.eeg_qc import kurtosis_reference_from_epochs, score_epoch
from dataset.export_npy import _epoch_bounds
from dataset.stress_labels.splits import ParticipantSplit, split_by_participant

DEFAULT_EPOCH_S = 1.0
SAMPLES_PER_EPOCH = 200  # matches dataset/export_npy.py's EEG convention (native 250Hz, resampled to 200/epoch)

META_LABEL_PREFIXES = ("cond_", "beh_", "derived_", "meta_")
PROSPECTIVE_MODES = {"prospective_pre_stimulus", "pre_click"}  # window must end at/before its anchor event

QC_SUMMARY_COLUMNS = [
    "qc_n_checks_passed_min_over_channels",
    "qc_n_checks_total",
    "qc_clip_frac_max",
    "qc_dc_drift_uv_max",
    "qc_emg_hf_ratio_max",
    "qc_line_noise_ratio_max",
    "qc_snr_db_min",
]


def _label_columns(samples: pd.DataFrame) -> list[str]:
    return [c for c in samples.columns if c.startswith(META_LABEL_PREFIXES) or c == "model_target"]


def compute_window_bounds(row: pd.Series, spec: WindowSpec) -> tuple[float, float]:
    """(t_start, t_end) for one sample row, per the version's WindowSpec.
    Raises if a prospective/pre-click window would leak past its anchor
    event (Rule 23) -- callers should catch/skip rather than silently
    clamp, since a leak here means the config itself is misconfigured."""
    if spec.mode == "block":
        return float(row[spec.start_col]), float(row[spec.end_col])

    anchor = float(row[spec.anchor_col])
    t_start, t_end = anchor - spec.pre_sec, anchor + spec.post_sec
    if spec.mode in PROSPECTIVE_MODES and t_end > anchor + 1e-9:
        raise ValueError(f"window_end {t_end} exceeds anchor {anchor} for prospective mode {spec.mode!r} -- check post_sec in config")
    return t_start, t_end


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


def _slice_uv(df: pd.DataFrame, t_start: float, t_end: float) -> tuple[np.ndarray, np.ndarray] | None:
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
    window_spec: WindowSpec,
    epoch_s: float = DEFAULT_EPOCH_S,
    samples_per_epoch: int = SAMPLES_PER_EPOCH,
) -> tuple[np.ndarray, pd.DataFrame, int]:
    """Cuts every sample row's window (computed from `window_spec`, not
    pre-existing start/end columns -- see compute_window_bounds) into
    epoch_s-second epochs of the out-ear EEG stream. Returns (X of shape
    (N, 8, samples_per_epoch), meta_df aligned row-for-row,
    skipped_leakage_count).

    Two passes per participant, same reasoning as dataset/stress_labels/
    export_npy.py: the kurtosis-outlier QC check needs THIS participant's
    own epoch population as its reference (electrode contact quality is
    participant-specific), so every epoch must be built once before any of
    them can be scored."""
    if samples.empty:
        return np.zeros((0, len(CHANNELS), samples_per_epoch), dtype=np.float32), pd.DataFrame(), 0

    label_cols = _label_columns(samples)
    skipped_leakage = 0

    X_list, meta_rows = [], []
    for pid, pid_samples in samples.groupby("participant_id"):
        eeg_path = Path(processed_dir) / pid / "ear_eeg_out.ads1299.parquet"
        if not eeg_path.exists():
            continue
        raw_df = pd.read_parquet(eeg_path)
        uv_raw, uv_filt = convert_and_filter(raw_df, CHANNELS, NATIVE_FS)

        pending = []  # (raw_epoch, filt_epoch, partial_meta_row)
        for _, row in pid_samples.iterrows():
            try:
                t_start, t_end = compute_window_bounds(row, window_spec)
            except ValueError:
                skipped_leakage += 1
                continue
            if t_end <= t_start:
                continue

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
                    "window_mode": window_spec.mode,
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

        for raw_epoch, filt_epoch, meta_row in pending:
            epoch_qc = score_epoch(raw_epoch, filt_epoch, fs=samples_per_epoch / epoch_s, kurtosis_ref=kurtosis_ref)
            X_list.append(filt_epoch.astype(np.float32))
            meta_row.update(_qc_summary_row(epoch_qc))
            meta_rows.append(meta_row)

    X = np.stack(X_list, axis=0) if X_list else np.zeros((0, len(CHANNELS), samples_per_epoch), dtype=np.float32)
    meta_df = pd.DataFrame(meta_rows)
    return X, meta_df, skipped_leakage


def export_version(
    processed_dir: Path,
    export_dir: Path,
    config_name: str,
    participant_ids: list[str] | None = None,
    epoch_s: float = DEFAULT_EPOCH_S,
    samples_per_epoch: int = SAMPLES_PER_EPOCH,
    seed: int = 42,
) -> dict:
    """Builds one Dataset Version's EEG epochs from its YAML config
    (configs/attention_labels/<config_name>.yaml) and writes
    <export_dir>/<config_name>_{X,y,meta}.{npy,npy,csv} (pooled across all
    participants -- Phase 5 per-participant splitting is done downstream on
    the meta table's participant_id column via dataset/stress_labels/
    splits.py, not re-derived here, EXCEPT for versions whose label itself
    needs a train-only fit -- e.g. a4_stroop_rt_residual -- where the same
    split is threaded through so the fit and the exported windows agree)."""
    processed_dir, export_dir = Path(processed_dir), Path(export_dir)
    config = load_version_config(config_name)
    if config.version_builder not in VERSION_BUILDERS:
        raise KeyError(f"config {config_name!r} references unknown version_builder {config.version_builder!r}; available: {list(VERSION_BUILDERS)}")

    if participant_ids is None:
        participant_ids = sorted(p.name for p in processed_dir.iterdir() if p.is_dir() and (p / "attention_blocks.parquet").exists())

    tables = load_all_canonical_tables(processed_dir, participant_ids)
    builder = VERSION_BUILDERS[config.version_builder]

    kwargs = {}
    if config.history:
        if "rt_trials" in config.history:
            kwargs["history_n"] = config.history["rt_trials"]
        if "minimum_valid_trials" in config.history:
            kwargs["minimum_history"] = config.history["minimum_valid_trials"]
    if config.version_builder == "a4_stroop_rt_residual":
        split = split_by_participant(participant_ids, seed=config.split_seed)
        kwargs["split"] = split
    if config.version_builder == "a6_schulte_click":
        kwargs["require_high_confidence"] = config.requirements.get("click_reconstruction_valid", True)

    samples = builder(tables, **kwargs)

    X, meta_df, skipped_leakage = build_eeg_epochs_for_version(processed_dir, config.name, samples, config.windowing, epoch_s, samples_per_epoch)

    target_col = "model_target" if "model_target" in meta_df.columns else None
    if target_col is not None and not meta_df.empty and pd.api.types.is_numeric_dtype(meta_df[target_col].dropna()):
        y = meta_df[target_col].fillna(-1).to_numpy(dtype=np.int64)
    else:
        y = np.full(len(meta_df), -1, dtype=np.int64)  # non-numeric/continuous targets stay in meta.csv only

    export_dir.mkdir(parents=True, exist_ok=True)
    np.save(export_dir / f"{config.name}_X.npy", X)
    np.save(export_dir / f"{config.name}_y.npy", y)
    meta_df.to_csv(export_dir / f"{config.name}_meta.csv", index=False, float_format="%.17g")

    return {
        "config_name": config.name,
        "n_samples_before_windowing": len(samples),
        "n_epochs": int(X.shape[0]),
        "shape": list(X.shape),
        "skipped_leakage_windows": skipped_leakage,
        "participants": sorted(meta_df["participant_id"].unique().tolist()) if not meta_df.empty else [],
        "epoch_seconds": epoch_s,
        "samples_per_epoch": samples_per_epoch,
        "window_mode": config.windowing.mode,
    }


def main():
    import argparse

    from dataset.attention_labels.config import discover_configs

    parser = argparse.ArgumentParser(description="Export one config-driven attention Dataset Version to fixed-shape EEG .npy epochs (Phase 4).")
    parser.add_argument("--processed-dir", type=str, required=True)
    parser.add_argument("--export-dir", type=str, required=True)
    parser.add_argument("--config", type=str, nargs="+", default=None, choices=discover_configs(), help="one or more config names under configs/attention_labels/")
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
        report = export_version(Path(args.processed_dir), Path(args.export_dir), name, epoch_s=args.epoch_seconds, samples_per_epoch=args.samples_per_epoch)
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
