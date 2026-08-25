"""Part (b): which contamination factors (drift, line noise, HF/EMG,
headroom) drive same-ear spatial SNR, and how does SNR behave per task.

Reuses analysis/eeg_channel_snr.py's windowed same-ear SNR and
analysis/eeg_channel_integrity.py's windowed drift/hf/line-noise/headroom
factors, joined on (channel, window_idx) within each task section, at the
same WINDOW_S so the two are directly comparable window-for-window.

Two outputs:
  eeg_snr_by_task.csv          -- per pid x channel x task: SNR + factor
                                   medians (answers "how does SNR behave
                                   per task").
  eeg_snr_factor_regression.csv -- per pid x channel (all tasks pooled) and
                                   per channel (all participants pooled):
                                   correlation + standardized-regression
                                   beta of each factor against SNR, plus R^2
                                   (answers "which factor matters most").
"""
import numpy as np
import pandas as pd
from pathlib import Path

from dataset.windows import extract_all_windows, load_events_parquet
from analysis.eeg_continuous_preprocess import (
    CHANNELS, NATIVE_FS, PARTICIPANTS, PROCESSED_DIR, UV_PER_CODE,
    resample_uniform, filter_continuous_uv,
)
from analysis.eeg_channel_snr import windowed_snr_db
from analysis.eeg_channel_integrity import compute_window_metrics, WINDOW_S

QC_DIR = Path(__file__).parents[1] / "data" / "qc"
FACTORS = ["drift_ratio_db", "hf_ratio_db", "line_noise_60_db", "headroom_fraction"]
TASKS_EXCLUDED = {"transition"}  # auto-detected gaps, not a real task


def load_task_section_uv(pid: str, task: str):
    events = load_events_parquet(PROCESSED_DIR / pid / "events.parquet")
    windows = [w for w in extract_all_windows(events) if w.task == task]
    if not windows:
        return None
    t0, t1 = min(w.t_start for w in windows), max(w.t_end for w in windows)
    df = pd.read_parquet(PROCESSED_DIR / pid / "ear_eeg_out.ads1299.parquet")
    sec = df[(df.wall_utc_s >= t0) & (df.wall_utc_s <= t1)].copy()
    sec[CHANNELS] = sec[CHANNELS].astype(np.float64) * UV_PER_CODE
    return sec


def participant_tasks(pid: str) -> list[str]:
    events = load_events_parquet(PROCESSED_DIR / pid / "events.parquet")
    tasks = {w.task for w in extract_all_windows(events)} - TASKS_EXCLUDED
    return sorted(tasks)


def snr_and_factors_for_task(pid: str, task: str) -> pd.DataFrame | None:
    sec = load_task_section_uv(pid, task)
    if sec is None or len(sec) < WINDOW_S * NATIVE_FS:
        return None
    raw_uniform = resample_uniform(sec, NATIVE_FS)
    filt = filter_continuous_uv(raw_uniform)
    raw_vals, filt_vals = raw_uniform[CHANNELS].to_numpy(), filt[CHANNELS].to_numpy()

    snr = windowed_snr_db(filt_vals, NATIVE_FS, WINDOW_S)[["channel", "window_idx", "snr_db"]]
    factors = compute_window_metrics(pid, raw_vals, filt_vals, NATIVE_FS, WINDOW_S)[["channel", "window_idx"] + FACTORS]
    merged = snr.merge(factors, on=["channel", "window_idx"], how="inner")
    merged.insert(0, "pid", pid)
    merged.insert(1, "task", task)
    return merged


def standardized_regression(df: pd.DataFrame, target: str, factors: list[str]) -> dict | None:
    """OLS on z-scored target/factors -- betas are comparable across factors
    despite their different units (dB vs. fraction)."""
    d = df[[target] + factors].dropna()
    stds = d.std()
    if len(d) < 10 or (stds == 0).any():
        return None
    z = (d - d.mean()) / stds
    X = np.column_stack([z[f].to_numpy() for f in factors] + [np.ones(len(z))])
    y = z[target].to_numpy()
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    r2 = 1.0 - np.sum((y - y_hat) ** 2) / np.sum((y - y.mean()) ** 2)
    out = {f"beta_{f}": b for f, b in zip(factors, beta[:-1])}
    out.update({"r2": float(r2), "n_windows": len(d)})
    return out


def factor_summary(df: pd.DataFrame, target: str, factors: list[str]) -> dict:
    row = {f"corr_{f}": float(df[target].corr(df[f])) for f in factors}
    reg = standardized_regression(df, target, factors)
    if reg:
        row.update(reg)
    else:
        row.update({f"beta_{f}": np.nan for f in factors} | {"r2": np.nan, "n_windows": len(df)})
    return row


def main():
    QC_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for pid in PARTICIPANTS:
        for task in participant_tasks(pid):
            df = snr_and_factors_for_task(pid, task)
            if df is not None and not df.empty:
                all_rows.append(df)
        print(f"{pid}: done")
    long_df = pd.concat(all_rows, ignore_index=True)

    # --- how does SNR behave per task (per participant, per channel) ---
    by_task = long_df.groupby(["pid", "channel", "task"]).agg(
        n_windows=("snr_db", "count"),
        snr_db_median=("snr_db", "median"),
        snr_db_p10=("snr_db", lambda s: s.quantile(0.10)),
        snr_db_p90=("snr_db", lambda s: s.quantile(0.90)),
        drift_ratio_db_median=("drift_ratio_db", "median"),
        hf_ratio_db_median=("hf_ratio_db", "median"),
        line_noise_60_db_median=("line_noise_60_db", "median"),
        headroom_fraction_median=("headroom_fraction", "median"),
    ).reset_index()
    by_task_path = QC_DIR / "eeg_snr_by_task.csv"
    by_task.to_csv(by_task_path, index=False)
    print(f"\nWrote {by_task_path}  ({len(by_task):,} rows)")

    # --- which factor moves SNR the most (all tasks pooled) ---
    reg_rows = []
    for (pid, ch), g in long_df.groupby(["pid", "channel"]):
        reg_rows.append({"scope": "participant_channel", "pid": pid, "channel": ch, **factor_summary(g, "snr_db", FACTORS)})
    for ch, g in long_df.groupby("channel"):
        reg_rows.append({"scope": "cohort_pooled_channel", "pid": "ALL", "channel": ch, **factor_summary(g, "snr_db", FACTORS)})
    regression = pd.DataFrame(reg_rows)
    reg_path = QC_DIR / "eeg_snr_factor_regression.csv"
    regression.to_csv(reg_path, index=False)
    print(f"Wrote {reg_path}  ({len(regression):,} rows)")

    print("\n=== Cohort-pooled per-channel: which factor best predicts same-ear SNR (standardized beta, all tasks pooled) ===")
    cohort = regression[regression.scope == "cohort_pooled_channel"].set_index("channel")
    print(cohort[[f"beta_{f}" for f in FACTORS] + ["r2", "n_windows"]].round(3).to_string())

    print("\n=== Per-task cohort-median SNR (across participants x channels) ===")
    print(by_task.groupby("task")["snr_db_median"].median().sort_values().round(1).to_string())


if __name__ == "__main__":
    main()
