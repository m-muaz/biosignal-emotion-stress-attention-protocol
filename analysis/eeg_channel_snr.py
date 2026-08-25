"""Per-channel SNR and same-ear anomaly detection for the out-ear EEG.

Rationale (per sensor_attachment.jpg): ch1-4 sit on the left ear, ch5-8 on
the right ear, close enough together that they should pick up a strongly
overlapping physiological signal. So for a channel, define:

    reference(t) = mean of its OTHER 3 same-ear channels at time t
    signal power = var(reference)                  -- the shared, presumably-real component
    noise power  = var(channel - reference)         -- this channel's private deviation from its ear-mates
    SNR_dB       = 10*log10(signal power / noise power)

A channel that isn't tracking its ear-mates (bad contact, cable noise,
local artifact) shows up as low SNR here, regardless of its own absolute
amplitude. Computed both over the whole emotion section (one number) and
in non-overlapping windows at two timescales (2s, 30s) to see whether a
channel is persistently bad or only misbehaves in short bursts.

A channel-window is flagged "anomalous" if its SNR is >6dB below the
median of its 3 ear-mates in that same window (relative, not absolute --
overall session quality varies a lot by participant, see
eeg_continuous_preprocess.py's clip-rate/amplitude findings).
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal as sps

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from analysis.eeg_continuous_preprocess import (
    CHANNELS, NATIVE_FS, PARTICIPANTS,
    load_emotion_section_uv, resample_uniform, filter_continuous_uv,
)
from analysis.plot_style import CHANNEL_COLOR, CATEGORICAL, DIVERGING_BLUE_RED, INK_PRIMARY, INK_SECONDARY, AXIS, STATUS_CRITICAL, style_axes

EAR_GROUPS = {"left": [0, 1, 2, 3], "right": [4, 5, 6, 7]}  # indices into CHANNELS
WINDOW_SIZES_S = {"fine": 2.0, "coarse": 30.0}
ANOMALY_MARGIN_DB = 6.0  # a channel is anomalous if its SNR trails its ear-median by more than this
PLOTS_DIR = Path(__file__).parents[1] / "data" / "qc_plots" / "channel_snr"
DIVERGING_CMAP = LinearSegmentedColormap.from_list("snr_diverging", [DIVERGING_BLUE_RED[2], DIVERGING_BLUE_RED[1], DIVERGING_BLUE_RED[0]])  # red(bad) -> gray(0dB) -> blue(good)

# Spectral SNR bands -- computed on the RAW (pre-filter) signal only. Running
# these on the already-filtered 1-40Hz signal would be circular: the bandpass
# already zeroed out everything outside that band, so any noise-band
# comparison would trivially show huge SNR regardless of actual data quality.
#
# DEPRECATED (see plans/out_ear_eeg_qc_analysis.md Priority 1): the
# 100-124Hz "noise floor" band is near the ADS1299's 250SPS Nyquist, where
# the ADC's own digital decimation filter attenuates the signal heavily.
# Low PSD there may reflect the ADC transfer function, not true electronic
# noise -- this metric has NOT been validated against a bench experiment
# (shorted inputs) and must NOT be used for channel rejection. Kept only
# as an experimental/legacy column; use analysis/eeg_spectral_quality.py's
# drift_ratio_db / hf_ratio_db / line_noise_60_db instead.
EEG_BAND = (1.0, 40.0)
NOISE_FLOOR_BAND_LEGACY = (100.0, 124.0)
CONTAMINATION_BANDS = [(0.1, 1.0), (40.0, 100.0)]  # sub-delta drift + EMG/movement artifact


def windowed_snr_db(vals: np.ndarray, fs: float, window_s: float) -> pd.DataFrame:
    """vals: (T, 8) filtered signal. Returns one row per (ear, channel, window)."""
    win_len = int(window_s * fs)
    n_win = vals.shape[0] // win_len
    rows = []
    for ear, idxs in EAR_GROUPS.items():
        for wi in range(n_win):
            seg = vals[wi * win_len:(wi + 1) * win_len, idxs]  # (win_len, 4)
            for j, ch_idx in enumerate(idxs):
                ref = seg[:, [k for k in range(4) if k != j]].mean(axis=1)
                noise = seg[:, j] - ref
                sig_p, noise_p = np.var(ref), np.var(noise)
                snr_db = 10 * np.log10(sig_p / noise_p) if noise_p > 0 else np.inf
                rows.append({"ear": ear, "channel": CHANNELS[ch_idx], "window_idx": wi, "t_start_s": wi * window_s, "snr_db": snr_db})
    return pd.DataFrame(rows)


def flag_anomalies(snr_df: pd.DataFrame) -> pd.DataFrame:
    """Per (ear, window), flags channels trailing their ear-mates' median SNR by > ANOMALY_MARGIN_DB."""
    ear_median = snr_df.groupby(["ear", "window_idx"])["snr_db"].transform("median")
    snr_df = snr_df.copy()
    snr_df["anomalous"] = snr_df["snr_db"] < (ear_median - ANOMALY_MARGIN_DB)
    return snr_df


def longest_anomalous_run(flags: np.ndarray) -> int:
    """Longest streak of consecutive True values (in window units)."""
    best = cur = 0
    for f in flags:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    return best


def band_power(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs <= hi)
    return float(psd[mask].mean()) if mask.any() else np.nan


def spectral_snr_db(raw_vals: np.ndarray, fs: float) -> pd.DataFrame:
    """Per-channel spectral SNR on the RAW signal (see module docstring for
    why not filtered). `_legacy` = EEG band vs the unvalidated 100-124Hz
    near-Nyquist band (deprecated, see NOISE_FLOOR_BAND_LEGACY above);
    `_broad` = EEG band vs (that band + known contamination bands) -- kept
    for continuity but should not drive channel-rejection decisions."""
    rows = []
    for i, ch in enumerate(CHANNELS):
        freqs, psd = sps.welch(raw_vals[:, i], fs=fs, nperseg=min(int(fs * 4), raw_vals.shape[0]))
        p_eeg = band_power(freqs, psd, *EEG_BAND)
        p_noise = band_power(freqs, psd, *NOISE_FLOOR_BAND_LEGACY)
        p_contam = sum(band_power(freqs, psd, lo, hi) for lo, hi in CONTAMINATION_BANDS)
        rows.append({
            "channel": ch,
            "spectral_highband_ratio_legacy_db": 10 * np.log10(p_eeg / p_noise) if p_noise > 0 else np.inf,
            "spectral_snr_broad_db": 10 * np.log10(p_eeg / (p_noise + p_contam)) if (p_noise + p_contam) > 0 else np.inf,
        })
    return pd.DataFrame(rows)


def summarize_participant(pid: str) -> pd.DataFrame:
    sec, windows = load_emotion_section_uv(pid)
    if sec is None:
        return pd.DataFrame()
    sec_uniform = resample_uniform(sec, NATIVE_FS)
    filt = filter_continuous_uv(sec_uniform)
    vals = filt[CHANNELS].to_numpy()

    section_s = vals.shape[0] / NATIVE_FS
    whole = windowed_snr_db(vals, NATIVE_FS, section_s)  # one "window" = entire section
    spectral = spectral_snr_db(sec_uniform[CHANNELS].to_numpy(), NATIVE_FS)

    rows = []
    for ch in CHANNELS:
        row = {
            "pid": pid, "channel": ch,
            "whole_section_snr_db": whole.loc[whole.channel == ch, "snr_db"].iloc[0],
            "spectral_highband_ratio_legacy_db": spectral.loc[spectral.channel == ch, "spectral_highband_ratio_legacy_db"].iloc[0],
            "spectral_snr_broad_db": spectral.loc[spectral.channel == ch, "spectral_snr_broad_db"].iloc[0],
        }
        for label, win_s in WINDOW_SIZES_S.items():
            snr_df = flag_anomalies(windowed_snr_db(vals, NATIVE_FS, win_s))
            ch_df = snr_df[snr_df.channel == ch].sort_values("window_idx")
            row[f"median_snr_db_{label}"] = float(ch_df["snr_db"].median())
            row[f"frac_anomalous_{label}"] = float(ch_df["anomalous"].mean())
            row[f"longest_anomalous_run_s_{label}"] = longest_anomalous_run(ch_df["anomalous"].to_numpy()) * win_s
        rows.append(row)
    return pd.DataFrame(rows)


def plot_snr_heatmap(report: pd.DataFrame):
    """Participant x channel whole-section SNR, diverging around 0dB
    (blue=good, red=bad -- 0dB means noise power equals shared-signal
    power with its ear-mates)."""
    pivot = report.pivot(index="pid", columns="channel", values="whole_section_snr_db")[CHANNELS]
    vmax = np.nanmax(np.abs(pivot.to_numpy()))
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(pivot.to_numpy(), cmap=DIVERGING_CMAP, vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iloc[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6, color=INK_PRIMARY)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Same-ear SNR (dB)", color=INK_SECONDARY)
    ax.set_title("Whole-section same-ear SNR: participant x channel", color=INK_PRIMARY, fontsize=11)
    fig.tight_layout()
    out = PLOTS_DIR / "snr_heatmap.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out}")


def plot_channel_summary_bar(report: pd.DataFrame):
    """Cohort median whole-section SNR per channel, worst-to-best, with a
    0dB reference line (below it: this channel's noise outweighs the
    shared signal its ear-mates agree on)."""
    med = report.groupby("channel")["whole_section_snr_db"].median().reindex(CHANNELS).sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(med.index, med.values, color=[CHANNEL_COLOR[ch] for ch in med.index])
    ax.axhline(0, color=AXIS, linewidth=1.2)
    ax.set_ylabel("Cohort median SNR (dB)", color=INK_SECONDARY)
    ax.set_title("Per-channel same-ear SNR across all participants", color=INK_PRIMARY, fontsize=11)
    style_axes(ax)
    fig.tight_layout()
    out = PLOTS_DIR / "channel_snr_summary.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out}")


def plot_snr_timeseries_example(pid: str, ear: str, window_s: float = 2.0):
    """SNR(t) for all 4 channels of one ear, one participant -- shows the
    timescale of misbehavior directly (persistent vs. short-burst dips),
    with the anomaly threshold band shaded per channel-window."""
    sec, _ = load_emotion_section_uv(pid)
    filt = filter_continuous_uv(resample_uniform(sec, NATIVE_FS))
    vals = filt[CHANNELS].to_numpy()
    snr_df = flag_anomalies(windowed_snr_db(vals, NATIVE_FS, window_s))
    ear_df = snr_df[snr_df.ear == ear]

    fig, ax = plt.subplots(figsize=(11, 5))
    for ch_idx in EAR_GROUPS[ear]:
        ch = CHANNELS[ch_idx]
        ch_df = ear_df[ear_df.channel == ch].sort_values("window_idx")
        ax.plot(ch_df.t_start_s, ch_df.snr_db, color=CHANNEL_COLOR[ch], linewidth=1.3, label=ch)
        anom = ch_df[ch_df.anomalous]
        ax.scatter(anom.t_start_s, anom.snr_db, color=STATUS_CRITICAL, s=10, zorder=5)
    ax.axhline(0, color=AXIS, linewidth=1.0, linestyle="--")
    ax.set_xlabel("Time into emotion section (s)", color=INK_SECONDARY)
    ax.set_ylabel(f"SNR (dB, {window_s:.0f}s windows)", color=INK_SECONDARY)
    ax.set_title(f"{pid}: {ear} ear -- per-window SNR (red dot = flagged anomalous)", color=INK_PRIMARY, fontsize=11)
    ax.legend(fontsize=8, frameon=False, ncol=4)
    style_axes(ax)
    fig.tight_layout()
    out = PLOTS_DIR / f"{pid}_{ear}_snr_timeseries.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out}")


def plot_snr_definitions_compare(report: pd.DataFrame):
    """Cohort median SNR per channel under all three definitions side by
    side: same-ear spatial (leave-one-out vs. ear-mates), spectral narrow
    (EEG band vs. pure hardware noise floor), spectral broad (EEG band vs.
    noise floor + drift/EMG contamination). A channel can fail one axis and
    pass another -- e.g. a channel that drifts in step with its ear-mates
    passes spatial SNR but fails spectral SNR."""
    defs = [
        ("whole_section_snr_db", "Same-ear spatial", CATEGORICAL[0]),
        ("spectral_highband_ratio_legacy_db", "Spectral vs. noise floor", CATEGORICAL[3]),
        ("spectral_snr_broad_db", "Spectral vs. noise+contamination", CATEGORICAL[5]),
    ]
    med = report.groupby("channel")[[d[0] for d in defs]].median().reindex(CHANNELS)
    x = np.arange(len(CHANNELS))
    width = 0.8 / len(defs)
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, (col, label, color) in enumerate(defs):
        ax.bar(x + i * width - 0.4 + width / 2, med[col], width=width, color=color, label=label)
    ax.axhline(0, color=AXIS, linewidth=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(CHANNELS)
    ax.set_ylabel("Cohort median SNR (dB)", color=INK_SECONDARY)
    ax.set_title("Per-channel SNR: same-ear spatial vs. spectral definitions", color=INK_PRIMARY, fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    style_axes(ax)
    fig.tight_layout()
    out = PLOTS_DIR / "snr_definitions_compare.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = [summarize_participant(pid) for pid in PARTICIPANTS]
    report = pd.concat([r for r in all_rows if not r.empty], ignore_index=True)
    plot_snr_heatmap(report)
    plot_channel_summary_bar(report)
    plot_snr_definitions_compare(report)

    worst = report.sort_values("frac_anomalous_coarse", ascending=False).iloc[0]
    plot_snr_timeseries_example(worst.pid, "left" if worst.channel in CHANNELS[:4] else "right")

    print("=== Whole-section same-ear SNR (dB) per channel -- sorted worst first ===")
    print(report.sort_values("whole_section_snr_db")[["pid", "channel", "whole_section_snr_db"]].to_string(index=False))

    print("\n=== Cohort median SNR by channel -- all three definitions ===")
    print(report.groupby("channel")[["whole_section_snr_db", "spectral_highband_ratio_legacy_db", "spectral_snr_broad_db"]]
          .median().reindex(CHANNELS).round(1).to_string())

    print("\n=== Channels persistently anomalous at the coarse (30s) timescale (frac_anomalous_coarse > 0.3) ===")
    persistent = report[report.frac_anomalous_coarse > 0.3].sort_values("frac_anomalous_coarse", ascending=False)
    print(persistent[["pid", "channel", "frac_anomalous_coarse", "frac_anomalous_fine", "longest_anomalous_run_s_fine"]].to_string(index=False) or "(none)")

    print("\n=== Channels only transiently anomalous (fine-scale flagged, but NOT coarse) -- short-burst artifacts ===")
    transient = report[(report.frac_anomalous_fine > 0.3) & (report.frac_anomalous_coarse <= 0.3)]
    print(transient[["pid", "channel", "frac_anomalous_fine", "frac_anomalous_coarse", "longest_anomalous_run_s_fine"]].to_string(index=False) or "(none)")

    report.to_csv("data/eeg_channel_snr_report.csv", index=False)
    print("\nWrote data/eeg_channel_snr_report.csv")


if __name__ == "__main__":
    main()
