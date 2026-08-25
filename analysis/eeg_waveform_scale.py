"""Part (a): per-participant, whole-session filtered-waveform amplitude at
multiple timescales -- a first-look sense-check before any cohort claims.

For each participant x channel, at window sizes 1/2/5/10s, computes
peak-to-peak and RMS of the FILTERED (60Hz notch + 1-40Hz bandpass) signal,
plus the fraction of windows flagged abnormal: either a filtered ptp over
ABNORMAL_PTP_UV (physiological EEG is ~10-100uV; matches the 500uV
convention already used in eeg_continuous_preprocess.py), or any exact
raw-ADC-rail clip inside the window.

Deliberately NOT rolled up into a cohort number here -- two CSVs, both
per-participant: full per-window rows (for distribution plots) and a
per-participant x channel x window-size summary (for bar/heatmap plots).
"""
import numpy as np
import pandas as pd
from pathlib import Path

from analysis.eeg_continuous_preprocess import (
    CHANNELS, NATIVE_FS, PARTICIPANTS, PROCESSED_DIR, FULL_SCALE_CODE, UV_PER_CODE,
    resample_uniform, filter_continuous_uv,
)

QC_DIR = Path(__file__).parents[1] / "data" / "qc"
WINDOW_SIZES_S = [1.0, 2.0, 5.0, 10.0]
ADC_FULL_SCALE_UV = FULL_SCALE_CODE * UV_PER_CODE
CLIP_LEVEL = 0.999
ABNORMAL_PTP_UV = 500.0


def load_whole_session_uv(pid: str) -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / pid / "ear_eeg_out.ads1299.parquet").copy()
    df[CHANNELS] = df[CHANNELS].astype(np.float64) * UV_PER_CODE
    return df


def windowed_ptp_rms(raw: np.ndarray, filt: np.ndarray, fs: float, window_s: float) -> pd.DataFrame:
    """raw, filt: (T, 8) uV on the same uniform grid. Vectorized reshape,
    not a per-window Python loop -- whole-session data is ~250-900k samples."""
    win_len = int(window_s * fs)
    n_win = raw.shape[0] // win_len
    r = raw[:n_win * win_len].reshape(n_win, win_len, 8)
    f = filt[:n_win * win_len].reshape(n_win, win_len, 8)

    ptp_uv = f.max(axis=1) - f.min(axis=1)                                  # (n_win, 8)
    rms_uv = np.sqrt((f ** 2).mean(axis=1))                                 # (n_win, 8)
    raw_clip_fraction = (np.abs(r) >= CLIP_LEVEL * ADC_FULL_SCALE_UV).mean(axis=1)

    return pd.DataFrame({
        "window_idx": np.repeat(np.arange(n_win), 8),
        "t_start_s": np.repeat(np.arange(n_win) * window_s, 8),
        "channel": np.tile(CHANNELS, n_win),
        "ptp_uv": ptp_uv.ravel(),
        "rms_uv": rms_uv.ravel(),
        "raw_clip_fraction": raw_clip_fraction.ravel(),
    })


def main():
    QC_DIR.mkdir(parents=True, exist_ok=True)
    all_windows = []

    for pid in PARTICIPANTS:
        sec = load_whole_session_uv(pid)
        raw_uniform = resample_uniform(sec, NATIVE_FS)
        filt = filter_continuous_uv(raw_uniform)
        raw_vals, filt_vals = raw_uniform[CHANNELS].to_numpy(), filt[CHANNELS].to_numpy()

        for window_s in WINDOW_SIZES_S:
            df = windowed_ptp_rms(raw_vals, filt_vals, NATIVE_FS, window_s)
            df.insert(0, "pid", pid)
            df.insert(1, "window_s", window_s)
            all_windows.append(df)
        print(f"{pid}: {raw_vals.shape[0] / NATIVE_FS:.0f}s session processed")

    windows = pd.concat(all_windows, ignore_index=True)
    windows["abnormal_ptp"] = windows.ptp_uv > ABNORMAL_PTP_UV
    windows["abnormal_clip"] = windows.raw_clip_fraction > 0
    windows["abnormal_any"] = windows.abnormal_ptp | windows.abnormal_clip

    windows_path = QC_DIR / "eeg_waveform_scale_windows.csv"
    windows.to_csv(windows_path, index=False)
    print(f"\nWrote {windows_path}  ({len(windows):,} rows)")

    summary = windows.groupby(["pid", "channel", "window_s"]).agg(
        n_windows=("ptp_uv", "count"),
        ptp_uv_median=("ptp_uv", "median"),
        ptp_uv_p95=("ptp_uv", lambda s: s.quantile(0.95)),
        rms_uv_median=("rms_uv", "median"),
        rms_uv_p95=("rms_uv", lambda s: s.quantile(0.95)),
        frac_abnormal_ptp=("abnormal_ptp", "mean"),
        frac_abnormal_clip=("abnormal_clip", "mean"),
        frac_abnormal_any=("abnormal_any", "mean"),
    ).reset_index()

    summary_path = QC_DIR / "eeg_waveform_scale_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}  ({len(summary):,} rows, per-participant -- no cohort rollup)")

    print("\n=== Per-participant summary at the 2s window size (first few rows, for a sanity look) ===")
    print(summary[summary.window_s == 2.0].head(16).to_string(index=False))


if __name__ == "__main__":
    main()
