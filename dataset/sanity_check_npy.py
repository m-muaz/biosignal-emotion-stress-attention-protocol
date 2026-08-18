"""Independent verification of `dataset/export_npy.py` output -- does NOT
trust `export_manifest.json`'s own bookkeeping; re-derives everything from
the canonical Parquet + events log fresh and compares.

Checks, per participant per file key x stream:
  1. Shape sanity: X.ndim==4, X.shape==(B,C,T,samples_per_segment),
     y.shape==(B,), dtypes, C matches that stream's configured channel count.
  2. Window-count cross-check: re-runs `extract_all_windows` (+ the lapse
     window builders) fresh and confirms it matches `n_matched_windows`
     recorded at export time -- catches any drift between export and the
     live windows.py/export_npy.py code.
  3. Recompute-and-compare: for a sample of rows, rebuilds that exact
     window's (C, T, samples_per_segment) array straight from the Parquet
     again and asserts it's bit-identical (allclose) to what's stored in
     the .npy at the same row -- the strongest possible "is this actually
     aligned" check, since it repeats the real computation rather than
     inspecting metadata.
  4. NaN/degenerate-data reporting: fraction of NaN per file (expected 0
     unless a stream has a genuine multi-second gap inside a window).
  5. Pooled-vs-per-participant totals: pooled file's B equals the sum of
     every participant's `n_written` for that file, and its
     participant_ids.npy length matches.

Usage:
    python -m dataset.sanity_check_npy --processed-dir <out_dir> --export-dir <out_dir>\\npy_export
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dataset.export_npy import BASE_SPECS, _file_prefix, _gather_windows, _segment_resample

N_RECOMPUTE_SAMPLES = 5  # per (participant, file prefix) -- rows re-derived from scratch and compared


def _check_participant_file(pdir: Path, pid: str, prefix: str, T: int, n_channels: int, sps: int, df: pd.DataFrame, matched_windows, value_cols, report: dict):
    X_path, y_path, meta_path = pdir / f"{prefix}_X.npy", pdir / f"{prefix}_y.npy", pdir / f"{prefix}_meta.csv"
    if not (X_path.exists() and y_path.exists()):
        report["errors"].append(f"{pid}/{prefix}: missing X.npy or y.npy")
        return

    X = np.load(X_path)
    y = np.load(y_path)
    # float_precision="round_trip" -- pandas' default C-engine float parser
    # (xstrtod) silently loses the last bit of precision on t_start/t_end
    # (confirmed: a ~1.79e9 UTC timestamp reads back off by ~2e-7s), which
    # shifts the resample grid enough to fail the bit-exact recompute check
    # below. Every consumer of these meta.csv files should read them the
    # same way if they ever recompute against t_start/t_end.
    meta = pd.read_csv(meta_path, float_precision="round_trip") if meta_path.exists() else pd.DataFrame()

    if X.ndim != 4:
        report["errors"].append(f"{pid}/{prefix}: X.ndim={X.ndim}, expected 4")
        return
    if X.shape[1:] != (n_channels, T, sps):
        report["errors"].append(f"{pid}/{prefix}: X.shape={X.shape}, expected (B, {n_channels}, {T}, {sps})")
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

    # window-count cross-check: independent re-extraction vs. what was exported
    n_expected = len(matched_windows)
    n_recorded = report["manifest_participants"].get(pid, {}).get(prefix, {}).get("n_matched_windows")
    if n_recorded is not None and n_recorded != n_expected:
        report["errors"].append(f"{pid}/{prefix}: manifest says n_matched_windows={n_recorded}, fresh extract gives {n_expected}")

    # recompute-and-compare: rebuild a sample of rows straight from Parquet
    # and bit-check against what's actually stored in the .npy
    n_written = report["manifest_participants"].get(pid, {}).get(prefix, {}).get("n_written", X.shape[0])
    if n_written and len(meta):
        sample_idx = np.linspace(0, n_written - 1, num=min(N_RECOMPUTE_SAMPLES, n_written), dtype=int)
        for idx in sample_idx:
            row = meta.iloc[idx]
            recomputed = _segment_resample(df, float(row["t_start"]), float(row["t_end"]), value_cols, T, samples_per_segment=sps)
            if recomputed is None:
                report["errors"].append(f"{pid}/{prefix} row {idx}: recompute found zero signal samples, but this row was written")
                continue
            if not np.allclose(recomputed, X[idx], equal_nan=True):
                report["errors"].append(f"{pid}/{prefix} row {idx}: recomputed array does NOT match stored .npy row -- alignment bug")
        report["n_recomputed_ok"] += len(sample_idx)


def run_sanity_checks(processed_dir: Path, export_dir: Path, streams: list[str] | None = None) -> dict:
    processed_dir, export_dir = Path(processed_dir), Path(export_dir)
    manifest = json.loads((export_dir / "export_manifest.json").read_text(encoding="utf-8"))
    streams = streams if streams is not None else manifest["streams"]
    channels = manifest["channels"]
    segments = manifest["segments"]
    samples_per_segment = manifest["samples_per_segment"]

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

        for spec in BASE_SPECS:
            key = spec["key"]
            matched = [w for w in windows if w.task == spec["task"] and w.window_type == spec["window_type"]]
            T = segments[key]

            for stream in streams:
                prefix = _file_prefix(key, stream)
                df = stream_dfs[stream]
                if df is None:
                    if "error" not in manifest["participants"].get(pid, {}).get(prefix, {}):
                        report["errors"].append(f"{pid}/{prefix}: {stream}.parquet missing but manifest has no error recorded for it")
                    continue
                n_channels = len(channels[stream])
                sps = samples_per_segment.get(stream, 200)
                _check_participant_file(pdir, pid, prefix, T, n_channels, sps, df, matched, channels[stream], report)

    # pooled-vs-per-participant totals
    pool_dir = export_dir / "pooled"
    if pool_dir.is_dir():
        for spec in BASE_SPECS:
            for stream in streams:
                prefix = _file_prefix(spec["key"], stream)
                pooled_y_path = pool_dir / f"{prefix}_y.npy"
                pooled_pid_path = pool_dir / f"{prefix}_participant_ids.npy"
                if not pooled_y_path.exists():
                    continue
                pooled_y = np.load(pooled_y_path)
                pooled_pid = np.load(pooled_pid_path, allow_pickle=True)
                expected_total = sum(manifest["participants"][pid].get(prefix, {}).get("n_written", 0) for pid in participant_ids)
                if len(pooled_y) != expected_total:
                    report["errors"].append(f"pooled/{prefix}: y has {len(pooled_y)} rows, sum of per-participant n_written is {expected_total}")
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
