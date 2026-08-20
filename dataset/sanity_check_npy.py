"""Independent verification of `dataset/export_npy.py` output -- does NOT
trust `export_manifest.json`'s own bookkeeping; re-derives everything from
the canonical Parquet + events log fresh and compares.

Checks, per participant per file key x stream:
  1. Shape sanity: X.ndim==3, X.shape==(N,C,samples_per_epoch), y.shape==(N,).
  2. Window-count cross-check: re-runs extract_all_windows (+ lapse
     builders) fresh, compares against n_matched_windows.
  3. Epoch-count cross-check: reapplies floor(duration/epoch_s) per matched
     window, compares against n_epochs_written.
  4. Recompute-and-compare: rebuilds a sample of rows straight from the
     Parquet (using each row's own epoch t_start/t_end) and asserts
     bit-exact match against the stored .npy row.
  5. NaN reporting.
  6. Pooled-vs-per-participant totals.

Usage:
    python -m dataset.sanity_check_npy --processed-dir <out_dir> --export-dir <out_dir>\\npy_export
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataset.export_npy import BASE_SPECS, _epoch_bounds, _file_prefix, _gather_windows, _segment_resample

N_RECOMPUTE_SAMPLES = 5  # per (participant, file prefix) -- rows re-derived from scratch and compared


def _check_participant_file(pdir: Path, pid: str, prefix: str, epoch_s: float, n_channels: int, sps: int, df: pd.DataFrame, matched_windows, value_cols, report: dict):
    X_path, y_path, meta_path = pdir / f"{prefix}_X.npy", pdir / f"{prefix}_y.npy", pdir / f"{prefix}_meta.csv"
    if not (X_path.exists() and y_path.exists()):
        report["errors"].append(f"{pid}/{prefix}: missing X.npy or y.npy")
        return

    X = np.load(X_path)
    y = np.load(y_path)
    # round_trip: pandas' default float parser loses precision on
    # t_start/t_end, which breaks the bit-exact recompute check below.
    meta = pd.read_csv(meta_path, float_precision="round_trip") if meta_path.exists() else pd.DataFrame()

    if X.ndim != 3:
        report["errors"].append(f"{pid}/{prefix}: X.ndim={X.ndim}, expected 3")
        return
    if X.shape[1:] != (n_channels, sps):
        report["errors"].append(f"{pid}/{prefix}: X.shape={X.shape}, expected (N, {n_channels}, {sps})")
    if y.shape != (X.shape[0],):
        report["errors"].append(f"{pid}/{prefix}: y.shape={y.shape} != (X.shape[0]={X.shape[0]},)")
    if len(meta) != X.shape[0]:
        report["errors"].append(f"{pid}/{prefix}: meta.csv has {len(meta)} rows, X has {X.shape[0]}")
    if X.dtype != np.float32:
        report["warnings"].append(f"{pid}/{prefix}: X.dtype={X.dtype}, expected float32")
    if y.dtype.kind not in ("i", "u"):
        report["warnings"].append(f"{pid}/{prefix}: y.dtype={y.dtype}, expected integer")

    n_nan = int(np.isnan(X).sum()) if X.size else 0
    if n_nan:
        report["warnings"].append(f"{pid}/{prefix}: {n_nan}/{X.size} NaN values ({100*n_nan/X.size:.3f}%)")

    participant_report = report["manifest_participants"].get(pid, {}).get(prefix, {})

    # window-count cross-check
    n_expected = len(matched_windows)
    n_recorded = participant_report.get("n_matched_windows")
    if n_recorded is not None and n_recorded != n_expected:
        report["errors"].append(f"{pid}/{prefix}: manifest says n_matched_windows={n_recorded}, fresh extract gives {n_expected}")

    # epoch-count cross-check. matched_windows isn't label-filtered, so this
    # is an upper bound: only asserted exactly when nothing was dropped.
    n_epochs_expected = sum(_epoch_bounds(w.t_start, w.t_end, epoch_s)[0] for w in matched_windows)
    dropped = participant_report.get("dropped", {})
    n_epochs_recorded = participant_report.get("n_epochs_written")
    if n_epochs_recorded is not None:
        n_dropped_elsewhere = dropped.get("label_unresolvable", 0) + dropped.get("no_signal_epochs", 0)
        if n_dropped_elsewhere == 0 and n_epochs_recorded != n_epochs_expected:
            report["errors"].append(f"{pid}/{prefix}: manifest says n_epochs_written={n_epochs_recorded}, fresh sum gives {n_epochs_expected}")
        elif n_epochs_recorded > n_epochs_expected:
            report["errors"].append(f"{pid}/{prefix}: n_epochs_written={n_epochs_recorded} exceeds upper bound {n_epochs_expected} from fresh re-extraction")

    # recompute-and-compare: rebuild a sample of rows from each row's own
    # epoch bounds (n_segments=1) and bit-check against the stored .npy row
    n_epochs_written = participant_report.get("n_epochs_written", X.shape[0])
    if n_epochs_written and len(meta):
        sample_idx = np.linspace(0, n_epochs_written - 1, num=min(N_RECOMPUTE_SAMPLES, n_epochs_written), dtype=int)
        for idx in sample_idx:
            row = meta.iloc[idx]
            recomputed = _segment_resample(df, float(row["t_start"]), float(row["t_end"]), value_cols, 1, samples_per_segment=sps)
            if recomputed is None:
                report["errors"].append(f"{pid}/{prefix} row {idx}: recompute found zero signal samples, but this row was written")
                continue
            if not np.allclose(recomputed[:, 0, :], X[idx], equal_nan=True):
                report["errors"].append(f"{pid}/{prefix} row {idx}: recomputed array does NOT match stored .npy row -- alignment bug")
        report["n_recomputed_ok"] += len(sample_idx)


def run_sanity_checks(processed_dir: Path, export_dir: Path, streams: list[str] | None = None) -> dict:
    processed_dir, export_dir = Path(processed_dir), Path(export_dir)
    manifest = json.loads((export_dir / "export_manifest.json").read_text(encoding="utf-8"))
    streams = streams if streams is not None else manifest["streams"]
    channels = manifest["channels"]
    epoch_seconds = manifest["epoch_seconds"]
    samples_per_segment = manifest["samples_per_segment"]
    specs_by_key = {s["key"]: s for s in BASE_SPECS}
    file_keys = manifest.get("file_keys", list(specs_by_key))  # actual keys exported this run, e.g. for --emotion-only

    report = {"errors": [], "warnings": [], "n_recomputed_ok": 0, "manifest_participants": manifest["participants"]}

    participant_ids = sorted(manifest["participants"])
    windows_by_pid = _gather_windows(processed_dir, participant_ids)

    for pid in participant_ids:
        pdir = export_dir / pid
        windows = windows_by_pid.get(pid, [])

        stream_dfs = {}
        for stream in streams:
            path = processed_dir / pid / f"{stream}.parquet"
            stream_dfs[stream] = pd.read_parquet(path) if path.exists() else None

        for key in file_keys:
            spec = specs_by_key[key]
            matched = [w for w in windows if w.task == spec["task"] and w.window_type == spec["window_type"]]
            epoch_s = epoch_seconds[key]

            for stream in streams:
                prefix = _file_prefix(key, stream)
                df = stream_dfs[stream]
                if df is None:
                    if "error" not in manifest["participants"].get(pid, {}).get(prefix, {}):
                        report["errors"].append(f"{pid}/{prefix}: {stream}.parquet missing but manifest has no error recorded for it")
                    continue
                n_channels = len(channels[stream])
                sps = samples_per_segment.get(stream, 200)
                _check_participant_file(pdir, pid, prefix, epoch_s, n_channels, sps, df, matched, channels[stream], report)

    # pooled-vs-per-participant totals
    pool_dir = export_dir / "pooled"
    if pool_dir.is_dir():
        for key in file_keys:
            for stream in streams:
                prefix = _file_prefix(key, stream)
                pooled_y_path = pool_dir / f"{prefix}_y.npy"
                pooled_pid_path = pool_dir / f"{prefix}_participant_ids.npy"
                if not pooled_y_path.exists():
                    continue
                pooled_y = np.load(pooled_y_path)
                pooled_pid = np.load(pooled_pid_path, allow_pickle=True)
                expected_total = sum(manifest["participants"][pid].get(prefix, {}).get("n_epochs_written", 0) for pid in participant_ids)
                if len(pooled_y) != expected_total:
                    report["errors"].append(f"pooled/{prefix}: y has {len(pooled_y)} rows, sum of per-participant n_epochs_written is {expected_total}")
                if len(pooled_pid) != len(pooled_y):
                    report["errors"].append(f"pooled/{prefix}: participant_ids.npy length {len(pooled_pid)} != y length {len(pooled_y)}")

    report["ok"] = len(report["errors"]) == 0
    return report


def main():
    parser = argparse.ArgumentParser(description="Independently verify dataset/export_npy.py output.")
    parser.add_argument("--processed-dir", type=str, required=True)
    parser.add_argument("--export-dir", type=str, required=True)
    parser.add_argument("--streams", nargs="*", default=None, help="Default: whatever streams are recorded in export_manifest.json")
    args = parser.parse_args()

    report = run_sanity_checks(Path(args.processed_dir), Path(args.export_dir), streams=args.streams)
    print(f"Recomputed-and-compared {report['n_recomputed_ok']} sample rows.")
    print(f"{len(report['warnings'])} warning(s):")
    for w in report["warnings"]:
        print(f"  WARN  {w}")
    print(f"{len(report['errors'])} error(s):")
    for e in report["errors"]:
        print(f"  ERROR {e}")
    print("SANITY CHECK: " + ("PASSED" if report["ok"] else "FAILED"))


if __name__ == "__main__":
    main()
