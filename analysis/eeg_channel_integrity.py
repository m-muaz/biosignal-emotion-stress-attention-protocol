"""Raw-ADC hardware integrity, robust amplitude, spectral contamination, and
spatial-correlation QC for the out-ear EEG.

Implements Priorities 2 (raw integrity), 4 (robust amplitude), 5 (spectral
contamination), 6 (spatial correlation), and 13 (composite usable-window
report) of plans/out_ear_eeg_qc_analysis.md. Extends, not replaces,
analysis/eeg_channel_snr.py -- run both.

Everything below is computed per participant x channel x 10s window on the
same time grid analysis/eeg_continuous_preprocess.py already builds:
`raw` = resample_uniform() output (raw uV, pre-filter, uniform 250Hz grid);
`filt` = filter_continuous_uv(raw) (60Hz notch + 1-40Hz bandpass).

Threshold strategy (see plan): exact-rail / flatline / headroom checks are
absolute (ADC-rail-derived) hard-fail flags. Drift / HF / line-noise are
informational cohort-relative robust-z flags -- a channel is only flagged
if it's an outlier relative to the rest of the cohort, not against an
absolute EEG-textbook number.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal as sps

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from analysis.eeg_continuous_preprocess import (
    CHANNELS, NATIVE_FS, PARTICIPANTS, FULL_SCALE_CODE, UV_PER_CODE,
    load_emotion_section_uv, resample_uniform, filter_continuous_uv,
)
from analysis.eeg_channel_snr import longest_anomalous_run
from analysis.plot_style import INK_PRIMARY, INK_SECONDARY, AXIS, CATEGORICAL, style_axes

PLOTS_DIR = Path(__file__).parents[1] / "data" / "qc_plots" / "channel_integrity"
QC_DIR = Path(__file__).parents[1] / "data" / "qc"

WINDOW_S = 10.0
ADC_FULL_SCALE_UV = FULL_SCALE_CODE * UV_PER_CODE  # differential full-scale, both rails
CLIP_LEVEL = 0.999  # matches eeg_continuous_preprocess.CLIP_MARGIN convention
RAIL_LEVELS = {"rail90_fraction": 0.90, "rail95_fraction": 0.95, "rail99_fraction": 0.99, "rail999_fraction": 0.999}

EAR_GROUPS = {"left": [0, 1, 2, 3], "right": [4, 5, 6, 7]}
MIRROR_PAIRS = [("ch1", "ch5"), ("ch2", "ch6"), ("ch3", "ch7"), ("ch4", "ch8")]

DRIFT_BAND, EEG_BAND = (0.1, 1.0), (1.0, 40.0)
HF_LOW_BAND, HF_BAND = (1.0, 20.0), (20.0, 45.0)
LINE_SIGNAL_BAND, LINE_BG_BANDS = (59.0, 61.0), [(55.0, 58.0), (62.0, 65.0)]

# Hard-fail thresholds -- physically grounded, not cohort-relative.
NEAR_RAIL_BAD_FRAC = 0.01   # >1% of a window pinned above the 99% rail
MIN_HEADROOM = 0.10         # window's median sits within 10% of full scale
FLAT_MAD_UV = 0.5           # near-flatline: robust sigma (MAD*1.4826) below this many uV
# Informational cohort-relative threshold (Priority 5/9's spectral flags).
ROBUST_Z_BAD = 3.0


def band_power(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    """Integrated (not mean) PSD over [lo, hi]: power = sum(PSD[band]) * df."""
    mask = (freqs >= lo) & (freqs <= hi)
    if not mask.any():
        return np.nan
    return float(psd[mask].sum() * (freqs[1] - freqs[0]))


def robust_z(s: pd.Series) -> pd.Series:
    med = s.median()
    mad = 1.4826 * (s - med).abs().median()
    return (s - med) / mad if mad > 0 else pd.Series(0.0, index=s.index)


def window_correlation_summary(f_win: np.ndarray) -> dict:
    """f_win: (win_len, 8) filtered signal. Whole 8x8 corr matrix reduced to
    within/cross-ear and mirrored-pair scalars (Priority 6)."""
    corr = np.corrcoef(f_win, rowvar=False)
    left, right = EAR_GROUPS["left"], EAR_GROUPS["right"]
    within = [corr[i, j] for grp in (left, right) for gi, i in enumerate(grp) for j in grp[gi + 1:]]
    cross = [corr[i, j] for i in left for j in right]
    out = {
        "mean_within_ear_corr": float(np.mean(within)),
        "mean_cross_ear_corr": float(np.mean(cross)),
        "mean_all_channel_corr": float(np.mean(corr[np.triu_indices(8, k=1)])),
    }
    for a, b in MIRROR_PAIRS:
        out[f"mirror_corr_{a}_{b}"] = float(corr[CHANNELS.index(a), CHANNELS.index(b)])
    return out


def compute_window_metrics(pid: str, raw: np.ndarray, filt: np.ndarray, fs: float, window_s: float) -> pd.DataFrame:
    """raw, filt: (T, 8) uV, same uniform time grid. One row per channel x window."""
    win_len = int(window_s * fs)
    n_win = raw.shape[0] // win_len
    nperseg = min(int(fs * 4), win_len)
    rows = []
    for wi in range(n_win):
        r_win, f_win = raw[wi * win_len:(wi + 1) * win_len], filt[wi * win_len:(wi + 1) * win_len]
        spatial = window_correlation_summary(f_win)
        for ci, ch in enumerate(CHANNELS):
            rc, fc = r_win[:, ci], f_win[:, ci]
            abs_frac = np.abs(rc) / ADC_FULL_SCALE_UV
            med = float(np.median(rc))
            mad = float(1.4826 * np.median(np.abs(rc - med)))
            headroom = 1.0 - abs(med) / ADC_FULL_SCALE_UV
            _, counts = np.unique(rc, return_counts=True)

            freqs, psd = sps.welch(rc, fs=fs, nperseg=nperseg)
            p_drift, p_eeg = band_power(freqs, psd, *DRIFT_BAND), band_power(freqs, psd, *EEG_BAND)
            p_hf_low, p_hf = band_power(freqs, psd, *HF_LOW_BAND), band_power(freqs, psd, *HF_BAND)
            p_line = band_power(freqs, psd, *LINE_SIGNAL_BAND)
            p_bg = float(np.mean([band_power(freqs, psd, lo, hi) for lo, hi in LINE_BG_BANDS]))

            row = {
                "pid": pid, "channel": ch, "window_idx": wi, "t_start_s": wi * window_s,
                "clip_fraction": float((abs_frac >= CLIP_LEVEL).mean()),
                "raw_median_uv": med, "raw_mad_uv": mad,
                "offset_fraction": 1.0 - headroom, "headroom_fraction": headroom,
                "unique_code_fraction": float(len(counts) / len(rc)),
                "dominant_code_fraction": float(counts.max() / len(rc)),
                "repeat_fraction": float((np.diff(rc) == 0).mean()),
                "filtered_mad_uv": float(1.4826 * np.median(np.abs(fc - np.median(fc)))),
                "filtered_rms_uv": float(np.sqrt(np.mean(fc ** 2))),
                "filtered_ptp_uv": float(np.ptp(fc)),
                "drift_ratio_db": 10 * np.log10(p_drift / p_eeg) if p_eeg > 0 else np.nan,
                "hf_ratio_db": 10 * np.log10(p_hf / p_hf_low) if p_hf_low > 0 else np.nan,
                "line_noise_60_db": 10 * np.log10(p_line / p_bg) if p_bg > 0 else np.nan,
                **spatial,
            }
            for name, lvl in RAIL_LEVELS.items():
                row[name] = float((abs_frac >= lvl).mean())
            row["near_flat"] = mad < FLAT_MAD_UV
            row["bad_clipping"] = row["clip_fraction"] > 0
            row["bad_near_rail"] = row["rail99_fraction"] > NEAR_RAIL_BAD_FRAC
            row["bad_flatline"] = row["near_flat"]
            row["bad_offset"] = headroom < MIN_HEADROOM
            rows.append(row)
    return pd.DataFrame(rows)


def build_long_df() -> pd.DataFrame:
    frames = []
    for pid in PARTICIPANTS:
        sec, windows = load_emotion_section_uv(pid)
        if sec is None:
            continue
        raw_uniform = resample_uniform(sec, NATIVE_FS)
        filt = filter_continuous_uv(raw_uniform)
        frames.append(compute_window_metrics(pid, raw_uniform[CHANNELS].to_numpy(), filt[CHANNELS].to_numpy(), NATIVE_FS, WINDOW_S))
    return pd.concat(frames, ignore_index=True)


def build_report(long_df: pd.DataFrame) -> pd.DataFrame:
    """Rolls windows up to one row per participant x channel -- the plan's
    main `eeg_channel_quality_report.csv` output."""
    long_df = long_df.copy()
    long_df["hard_bad"] = long_df[["bad_clipping", "bad_near_rail", "bad_flatline", "bad_offset"]].any(axis=1)
    long_df["usable"] = ~long_df["hard_bad"]

    rows = []
    for (pid, ch), g in long_df.groupby(["pid", "channel"]):
        g = g.sort_values("window_idx")
        rows.append({
            "pid": pid, "channel": ch, "n_windows": len(g),
            "usable_pct": float(g["usable"].mean() * 100),
            "clip_pct": float(g["bad_clipping"].mean() * 100),
            "near_rail_pct": float(g["bad_near_rail"].mean() * 100),
            "flat_pct": float(g["bad_flatline"].mean() * 100),
            "offset_bad_pct": float(g["bad_offset"].mean() * 100),
            "headroom_median": float(g["headroom_fraction"].median()),
            "filtered_rms_uv_median": float(g["filtered_rms_uv"].median()),
            "drift_ratio_db_median": float(g["drift_ratio_db"].median()),
            "hf_ratio_db_median": float(g["hf_ratio_db"].median()),
            "line_noise_60_db_median": float(g["line_noise_60_db"].median()),
            "mean_within_ear_corr_median": float(g["mean_within_ear_corr"].median()),
            "mean_cross_ear_corr_median": float(g["mean_cross_ear_corr"].median()),
            "longest_bad_run_s": longest_anomalous_run(g["hard_bad"].to_numpy()) * WINDOW_S,
        })
    report = pd.DataFrame(rows)

    # Informational, cohort-relative outlier flags (Priority 5/13) -- computed
    # across the whole cohort per metric, higher = worse for all three.
    for col, z_col, flag_col in [
        ("drift_ratio_db_median", "z_drift", "bad_drift"),
        ("hf_ratio_db_median", "z_hf", "bad_hf"),
        ("line_noise_60_db_median", "z_line_noise", "bad_line_noise"),
    ]:
        report[z_col] = robust_z(report[col])
        report[flag_col] = report[z_col] > ROBUST_Z_BAD
    return report


def plot_usable_heatmap(report: pd.DataFrame):
    pivot = report.pivot(index="pid", columns="channel", values="usable_pct")[CHANNELS]
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(pivot.to_numpy(), cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(pivot.columns))); ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index))); ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j]:.0f}", ha="center", va="center", fontsize=6, color=INK_PRIMARY)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Usable windows (%, hard-fail-free)", color=INK_SECONDARY)
    ax.set_title("Hard-fail-free window %: participant x channel", color=INK_PRIMARY, fontsize=11)
    fig.tight_layout()
    out = PLOTS_DIR / "usable_fraction_heatmap.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor()); plt.close(fig)
    print(f"Saved: {out}")


def plot_bad_reason_breakdown(report: pd.DataFrame):
    """Cohort-mean %-bad by failure reason, per channel -- tells you WHY a
    channel fails, not just that it does."""
    reasons = ["clip_pct", "near_rail_pct", "flat_pct", "offset_bad_pct"]
    labels = ["Exact clip", "Near-rail (>99%)", "Flatline", "Low headroom"]
    med = report.groupby("channel")[reasons].mean().reindex(CHANNELS)
    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(CHANNELS))
    for reason, label, color in zip(reasons, labels, CATEGORICAL):
        ax.bar(CHANNELS, med[reason], bottom=bottom, label=label, color=color)
        bottom += med[reason].to_numpy()
    ax.set_ylabel("Cohort-mean % of windows failed (hard reasons)", color=INK_SECONDARY)
    ax.set_title("Why windows fail, by channel (stacked)", color=INK_PRIMARY, fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    style_axes(ax)
    fig.tight_layout()
    out = PLOTS_DIR / "bad_reason_breakdown.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor()); plt.close(fig)
    print(f"Saved: {out}")


def plot_mean_correlation_matrix():
    """Cohort-mean full 8x8 channel correlation matrix (Priority 6.1) --
    directly shows whether ch6 tracks its ear-mates and its mirror ch2."""
    mats = []
    for pid in PARTICIPANTS:
        sec, _ = load_emotion_section_uv(pid)
        if sec is None:
            continue
        filt = filter_continuous_uv(resample_uniform(sec, NATIVE_FS))
        vals = filt[CHANNELS].to_numpy()
        win_len = int(WINDOW_S * NATIVE_FS)
        n_win = vals.shape[0] // win_len
        zs = [np.arctanh(np.clip(np.corrcoef(vals[wi * win_len:(wi + 1) * win_len], rowvar=False), -0.999999, 0.999999))
              for wi in range(n_win)]
        mats.append(np.tanh(np.mean(zs, axis=0)))
    mean_corr = np.mean(mats, axis=0)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mean_corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(8)); ax.set_xticklabels(CHANNELS)
    ax.set_yticks(range(8)); ax.set_yticklabels(CHANNELS)
    for i in range(8):
        for j in range(8):
            ax.text(j, i, f"{mean_corr[i, j]:.2f}", ha="center", va="center", fontsize=7,
                     color="white" if abs(mean_corr[i, j]) > 0.6 else INK_PRIMARY)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Mean correlation")
    ax.set_title("Cohort-mean filtered 8x8 channel correlation", color=INK_PRIMARY, fontsize=11)
    fig.tight_layout()
    out = PLOTS_DIR / "mean_correlation_matrix.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor()); plt.close(fig)
    print(f"Saved: {out}")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    QC_DIR.mkdir(parents=True, exist_ok=True)

    long_df = build_long_df()
    report = build_report(long_df)

    report_path = QC_DIR / "eeg_channel_quality_report.csv"
    report.to_csv(report_path, index=False)
    print(f"Wrote {report_path}  ({len(report)} participant x channel rows, window={WINDOW_S:.0f}s)")

    plot_usable_heatmap(report)
    plot_bad_reason_breakdown(report)
    plot_mean_correlation_matrix()

    print("\n=== Cohort median usable_pct by channel (worst first) ===")
    print(report.groupby("channel")["usable_pct"].median().reindex(CHANNELS).sort_values().round(1).to_string())

    print("\n=== Channels flagged as cohort-relative spectral outliers (informational, robust_z > 3) ===")
    outliers = report[report.bad_drift | report.bad_hf | report.bad_line_noise]
    cols = ["pid", "channel", "z_drift", "z_hf", "z_line_noise"]
    print(outliers[cols].round(2).to_string(index=False) if len(outliers) else "(none)")

    print("\n=== Cohort median same-position mirror correlation (ch_i vs ch_i+4) ===")
    for a, b in MIRROR_PAIRS:
        col = f"mirror_corr_{a}_{b}"
        print(f"{a}-{b}: {long_df[col].median():.2f}")


if __name__ == "__main__":
    main()
