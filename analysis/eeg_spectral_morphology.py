"""Part (c): 1/f aperiodic spectral fit and the full 8x8 channel
correlation matrix (plan Priorities 9 and 6.1), computed on the whole
session at WINDOW_S windows -- each TWICE: once over all windows, once
restricted to windows that pass the hard-fail QC mask from
eeg_channel_integrity.py (not clipped/near-rail/flat/low-headroom), so the
"all" vs "clean" comparison shows how much artifacts distort these numbers.

aperiodic fit: log10(PSD) = intercept - exponent * log10(f), fit on the raw
(pre-filter) per-window PSD over APERIODIC_FIT_BAND via linear regression.

correlation: per-window 8x8 Pearson corr of the filtered signal, averaged
across windows in Fisher-z space, per participant x channel-pair.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal as sps

from analysis.eeg_continuous_preprocess import (
    CHANNELS, NATIVE_FS, PARTICIPANTS, PROCESSED_DIR, UV_PER_CODE,
    resample_uniform, filter_continuous_uv,
)
from analysis.eeg_channel_integrity import compute_window_metrics, WINDOW_S, MIRROR_PAIRS

QC_DIR = Path(__file__).parents[1] / "data" / "qc"
APERIODIC_FIT_BAND = (2.0, 40.0)  # avoids sub-delta drift and stays within the 1-40Hz passband
LEFT, RIGHT = CHANNELS[:4], CHANNELS[4:]
MIRROR_SET = {frozenset(p) for p in MIRROR_PAIRS}


def load_whole_session_uv(pid: str) -> pd.DataFrame:
    df = pd.read_parquet(PROCESSED_DIR / pid / "ear_eeg_out.ads1299.parquet").copy()
    df[CHANNELS] = df[CHANNELS].astype(np.float64) * UV_PER_CODE
    return df


def pair_relationship(a: str, b: str) -> str:
    if frozenset((a, b)) in MIRROR_SET:
        return "mirror"
    if a in LEFT and b in LEFT:
        return "within_left"
    if a in RIGHT and b in RIGHT:
        return "within_right"
    return "cross_ear"


def aperiodic_fit(freqs: np.ndarray, psd: np.ndarray) -> tuple[float, float]:
    """A near-flatline/all-zero window (dead or fully clipped channel) has
    ~0 PSD everywhere -- no meaningful log-log slope, so return NaN rather
    than fail; those windows are excluded from the median downstream."""
    mask = (freqs >= APERIODIC_FIT_BAND[0]) & (freqs <= APERIODIC_FIT_BAND[1]) & (psd > 0)
    if mask.sum() < 3:
        return np.nan, np.nan
    logf, logp = np.log10(freqs[mask]), np.log10(psd[mask])
    slope, intercept = np.polyfit(logf, logp, 1)
    pred = slope * logf + intercept
    r2 = 1.0 - np.sum((logp - pred) ** 2) / np.sum((logp - logp.mean()) ** 2)
    return -slope, r2  # PSD ~ f^-exponent -> exponent = -slope


def per_pid_windows(pid: str, raw: np.ndarray, filt: np.ndarray, fs: float, window_s: float):
    """Yields (window_idx, r_win (win_len,8), f_win (win_len,8)) plus a
    (n_win, 8) boolean clean mask built from eeg_channel_integrity's
    hard-fail flags."""
    win_metrics = compute_window_metrics(pid, raw, filt, fs, window_s)
    win_metrics["hard_bad"] = win_metrics[["bad_clipping", "bad_near_rail", "bad_flatline", "bad_offset"]].any(axis=1)
    clean = ~win_metrics.pivot(index="window_idx", columns="channel", values="hard_bad")[CHANNELS].to_numpy()

    win_len = int(window_s * fs)
    n_win = raw.shape[0] // win_len
    for wi in range(n_win):
        yield wi, raw[wi * win_len:(wi + 1) * win_len], filt[wi * win_len:(wi + 1) * win_len], clean[wi]


def main():
    QC_DIR.mkdir(parents=True, exist_ok=True)
    aperiodic_rows, corr_rows = [], []

    for pid in PARTICIPANTS:
        sec = load_whole_session_uv(pid)
        raw_uniform = resample_uniform(sec, NATIVE_FS)
        filt = filter_continuous_uv(raw_uniform)
        raw_vals, filt_vals = raw_uniform[CHANNELS].to_numpy(), filt[CHANNELS].to_numpy()
        nperseg = min(int(NATIVE_FS * 4), int(WINDOW_S * NATIVE_FS))

        for wi, r_win, f_win, clean_ch in per_pid_windows(pid, raw_vals, filt_vals, NATIVE_FS, WINDOW_S):
            for ci, ch in enumerate(CHANNELS):
                freqs, psd = sps.welch(r_win[:, ci], fs=NATIVE_FS, nperseg=nperseg)
                exponent, r2 = aperiodic_fit(freqs, psd)
                aperiodic_rows.append({"pid": pid, "channel": ch, "window_idx": wi,
                                        "aperiodic_exponent": exponent, "fit_r2": r2, "clean": bool(clean_ch[ci])})

            corr = np.corrcoef(f_win, rowvar=False)
            corr_z = np.arctanh(np.clip(corr, -0.999999, 0.999999))
            for i in range(8):
                for j in range(i + 1, 8):
                    corr_rows.append({"pid": pid, "channel_i": CHANNELS[i], "channel_j": CHANNELS[j],
                                       "window_idx": wi, "corr_z": corr_z[i, j],
                                       "both_clean": bool(clean_ch[i] and clean_ch[j])})
        print(f"{pid}: done")

    aperiodic_long = pd.DataFrame(aperiodic_rows)
    corr_long = pd.DataFrame(corr_rows)

    # --- aperiodic exponent: all windows vs. artifact-clean windows only ---
    def summarize_aperiodic(df, scope_name):
        g = df.groupby(["pid", "channel"]).agg(
            n_windows=("aperiodic_exponent", "count"),
            aperiodic_exponent_median=("aperiodic_exponent", "median"),
            aperiodic_exponent_p25=("aperiodic_exponent", lambda s: s.quantile(0.25)),
            aperiodic_exponent_p75=("aperiodic_exponent", lambda s: s.quantile(0.75)),
            fit_r2_median=("fit_r2", "median"),
        ).reset_index()
        g.insert(2, "scope", scope_name)
        return g

    aperiodic_report = pd.concat([
        summarize_aperiodic(aperiodic_long, "all_windows"),
        summarize_aperiodic(aperiodic_long[aperiodic_long.clean], "clean_windows_only"),
    ], ignore_index=True)
    aperiodic_path = QC_DIR / "eeg_aperiodic_fit.csv"
    aperiodic_report.to_csv(aperiodic_path, index=False)
    print(f"\nWrote {aperiodic_path}  ({len(aperiodic_report):,} rows)")

    # --- correlation matrix: all windows vs. artifact-clean windows only ---
    def summarize_corr(df, scope_name):
        g = df.groupby(["pid", "channel_i", "channel_j"]).agg(
            n_windows=("corr_z", "count"), corr_z_mean=("corr_z", "mean"),
        ).reset_index()
        g["corr"] = np.tanh(g["corr_z_mean"])
        g["relationship"] = [pair_relationship(a, b) for a, b in zip(g.channel_i, g.channel_j)]
        g.insert(3, "scope", scope_name)
        return g.drop(columns="corr_z_mean")

    corr_report = pd.concat([
        summarize_corr(corr_long, "all_windows"),
        summarize_corr(corr_long[corr_long.both_clean], "clean_windows_only"),
    ], ignore_index=True)
    corr_path = QC_DIR / "eeg_channel_correlation_matrix.csv"
    corr_report.to_csv(corr_path, index=False)
    print(f"Wrote {corr_path}  ({len(corr_report):,} rows)")

    print("\n=== Cohort-median aperiodic exponent by channel, all vs. clean windows ===")
    piv = aperiodic_report.groupby(["channel", "scope"])["aperiodic_exponent_median"].median().unstack()
    print(piv.reindex(CHANNELS).round(2).to_string())

    print("\n=== Cohort-median mirror-pair correlation, all vs. clean windows ===")
    mirror = corr_report[corr_report.relationship == "mirror"]
    print(mirror.groupby(["channel_i", "channel_j", "scope"])["corr"].median().unstack().round(2).to_string())


if __name__ == "__main__":
    main()
