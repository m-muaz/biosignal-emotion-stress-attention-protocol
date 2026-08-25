"""Standard EEG preprocessing on the raw, synced, continuous out-ear signal
(not the already-epoched npy export) for the emotion task section.

Filtering an already-cut 1s epoch with a 1Hz-highpass Butterworth (as the
earlier epoch-level script did) is not standard practice -- the filter's
settling time is comparable to the epoch length, so edge transients
dominate. The correct order is: convert -> filter the *continuous* signal
-> THEN cut into epochs.

ADC->uV conversion and filtering now live in dataset/eeg_preprocess.py
(shared with dataset/stress_labels/export_npy.py) -- re-imported here so
the analysis/eeg_*.py scripts that import CHANNELS/NATIVE_FS/UV_PER_CODE/
resample_uniform/filter_continuous_uv from this module keep working
unchanged.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal as sps
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset.windows import extract_all_windows, load_events_parquet
from dataset.export_npy import _epoch_bounds, _label_emotion_trial, _label_baseline
from dataset.eeg_preprocess import (
    CHANNELS,
    NATIVE_FS,
    VREF,
    GAIN,
    FULL_SCALE_CODE,
    UV_PER_CODE,
    CLIP_MARGIN,
    counts_to_uv,
    resample_uniform,
    filter_continuous_uv,
)
from analysis.plot_style import CATEGORICAL, CHANNEL_COLOR, LABEL_COLOR, LABEL_NAME, INK_PRIMARY, INK_SECONDARY, AXIS, style_axes

PROCESSED_DIR = Path(__file__).parents[1] / "data" / "processed"
PLOTS_DIR = Path(__file__).parents[1] / "data" / "qc_plots" / "continuous_preprocess"
EPOCH_FS = 200.0  # samples/epoch in the final X array -- matches dataset/export_npy.py's convention

EPOCH_S = 1.0
PARTICIPANTS = sorted(p.name for p in PROCESSED_DIR.iterdir() if (p / "events.parquet").exists())


def load_emotion_section_uv(pid: str):
    """Continuous [ch1..ch8] in uV for this participant's whole emotion
    task span (all emotion trial/baseline windows), plus those windows."""
    events = load_events_parquet(PROCESSED_DIR / pid / "events.parquet")
    windows = [w for w in extract_all_windows(events) if w.task == "emotion"]
    if not windows:
        return None, None
    t0, t1 = min(w.t_start for w in windows), max(w.t_end for w in windows)

    df = pd.read_parquet(PROCESSED_DIR / pid / "ear_eeg_out.ads1299.parquet")
    sec = df[(df.wall_utc_s >= t0) & (df.wall_utc_s <= t1)].copy()
    sec = counts_to_uv(sec, CHANNELS)
    return sec, windows


def epoch_section(sec_raw: pd.DataFrame, sec_filt: pd.DataFrame, windows) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cuts each emotion trial/baseline window into non-overlapping
    EPOCH_S-second epochs, resampled onto a fixed 200-sample grid --
    same convention as dataset/export_npy.py. Clipping is flagged from the
    *raw* (pre-filter) samples in each epoch: a zero-phase filter smears a
    railed plateau's ringing across neighboring samples and erases its
    exact-rail signature, so clip detection must happen before filtering."""
    clip_uv = FULL_SCALE_CODE * UV_PER_CODE * (1 - CLIP_MARGIN)
    raw_wall = sec_raw.wall_utc_s.to_numpy()
    raw_vals = sec_raw[CHANNELS].to_numpy()
    filt_wall = sec_filt.wall_utc_s.to_numpy()  # uniform NATIVE_FS grid, not raw_wall
    filt_vals = sec_filt[CHANNELS].to_numpy()
    X_list, y_list, clip_list = [], [], []
    for w in windows:
        label = _label_emotion_trial(w) if w.window_type == "trial" else _label_baseline(w)
        if label is None:
            continue
        n_epochs, _ = _epoch_bounds(w.t_start, w.t_end, EPOCH_S)
        for e in range(n_epochs):
            t0, t1 = w.t_start + e * EPOCH_S, w.t_start + (e + 1) * EPOCH_S
            raw_mask = (raw_wall >= t0) & (raw_wall <= t1)
            filt_mask = (filt_wall >= t0) & (filt_wall <= t1)
            if raw_mask.sum() < 2 or filt_mask.sum() < 2:
                continue
            clipped = bool((np.abs(raw_vals[raw_mask]) > clip_uv).any())
            grid = np.linspace(0.0, EPOCH_S, int(EPOCH_S * EPOCH_FS))
            epoch = np.stack([np.interp(grid, filt_wall[filt_mask] - t0, filt_vals[filt_mask, c]) for c in range(len(CHANNELS))])
            X_list.append(epoch.astype(np.float32))
            y_list.append(label)
            clip_list.append(clipped)
    X = np.stack(X_list) if X_list else np.zeros((0, len(CHANNELS), int(EPOCH_S * EPOCH_FS)), dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    clip = np.array(clip_list, dtype=bool)
    return X, y, clip


def plot_raw_vs_filtered(pid: str, sec_uniform: pd.DataFrame, filt: pd.DataFrame, seconds: float = 5.0):
    """Stacked 8-channel trace, raw (uniform-resampled, pre-filter) vs
    filtered -- same time base, so the notch+bandpass effect is a fair
    visual comparison."""
    n = int(seconds * NATIVE_FS)
    t = np.arange(n) / NATIVE_FS
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=False)
    for ax, df, title in [(axes[0], sec_uniform, "Raw (uV, pre-filter)"), (axes[1], filt, "Filtered (60Hz notch + 1-40Hz bandpass)")]:
        offset = 0.0
        for ch in CHANNELS:
            vals = df[ch].to_numpy()[:n]
            span = max(np.ptp(vals), 10.0)
            ax.plot(t, vals + offset, color=CHANNEL_COLOR[ch], linewidth=1.0, label=ch)
            offset += span * 1.3
        ax.set_title(title, color=INK_PRIMARY, fontsize=11)
        ax.set_xlabel("Time (s)", color=INK_SECONDARY)
        style_axes(ax)
    axes[0].legend(ncol=4, fontsize=7, loc="upper right", frameon=False)
    fig.suptitle(f"{pid}: raw vs. filtered trace ({seconds:.0f}s excerpt, emotion section)", color=INK_PRIMARY)
    fig.tight_layout()
    out = PLOTS_DIR / f"{pid}_raw_vs_filtered.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out}")


def plot_psd_by_label(freqs: np.ndarray, psd_all: np.ndarray, y_all: np.ndarray):
    """Mean PSD (averaged over channels and epochs) per emotion label."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ch_mean_psd = psd_all.mean(axis=1)  # (N_epochs, F) -- average over 8 channels
    for label in sorted(np.unique(y_all)):
        mean_psd = ch_mean_psd[y_all == label].mean(axis=0)
        ax.plot(freqs, mean_psd, color=LABEL_COLOR[label], linewidth=1.6, label=LABEL_NAME[label])
    ax.set_yscale("log")
    ax.set_xlim(0, 45)
    ax.set_xlabel("Frequency (Hz)", color=INK_SECONDARY)
    ax.set_ylabel("PSD (uV^2/Hz, log scale)", color=INK_SECONDARY)
    ax.set_title("Mean PSD by emotion label (channel- and epoch-averaged)", color=INK_PRIMARY, fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    style_axes(ax)
    fig.tight_layout()
    out = PLOTS_DIR / "psd_by_label.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out}")


def plot_bandpower_by_label(freqs: np.ndarray, psd_all: np.ndarray, y_all: np.ndarray, bands: list[tuple[float, float]], band_names: list[str]):
    """Grouped bar chart: mean band power per label, one group per band."""
    ch_mean_psd = psd_all.mean(axis=1)
    fig, ax = plt.subplots(figsize=(9, 5))
    labels = sorted(np.unique(y_all))
    width = 0.8 / len(labels)
    x = np.arange(len(bands))
    for i, label in enumerate(labels):
        means = []
        for lo, hi in bands:
            m = (freqs >= lo) & (freqs <= hi)
            means.append(ch_mean_psd[y_all == label][:, m].mean())
        ax.bar(x + i * width - 0.4 + width / 2, means, width=width, color=LABEL_COLOR[label], label=LABEL_NAME[label])
    ax.set_xticks(x)
    ax.set_xticklabels(band_names)
    ax.set_ylabel("Mean band power (uV^2/Hz)", color=INK_SECONDARY)
    ax.set_title("Band power by emotion label", color=INK_PRIMARY, fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    style_axes(ax)
    fig.tight_layout()
    out = PLOTS_DIR / "bandpower_by_label.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out}")


def plot_cohort_qc(clip_rates: dict, amp_stats: dict):
    """Per-participant clip rate and post-filter artifact-amplitude rate, sorted worst-first."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    pids_by_clip = sorted(clip_rates, key=lambda p: -clip_rates[p])
    axes[0].bar(pids_by_clip, [clip_rates[p] * 100 for p in pids_by_clip], color=CATEGORICAL[0])
    axes[0].set_ylabel("% epochs with a clipped raw sample", color=INK_SECONDARY)
    axes[0].set_title("Clipping rate by participant", color=INK_PRIMARY, fontsize=11)

    pids_by_amp = sorted(amp_stats, key=lambda p: -amp_stats[p]["frac_over_500uV"])
    axes[1].bar(pids_by_amp, [amp_stats[p]["frac_over_500uV"] * 100 for p in pids_by_amp], color=CATEGORICAL[1])
    axes[1].set_ylabel("% raw-clean epochs with peak >500uV", color=INK_SECONDARY)
    axes[1].set_title("Post-filter artifact rate by participant", color=INK_PRIMARY, fontsize=11)

    for ax in axes:
        ax.tick_params(axis="x", rotation=90)
        style_axes(ax)
    fig.tight_layout()
    out = PLOTS_DIR / "cohort_qc_summary.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out}")


def plot_classifier_result(acc: np.ndarray, base: np.ndarray, chance: float):
    fig, ax = plt.subplots(figsize=(5, 5))
    names = ["LogReg\n(bandpower)", "Majority\nbaseline"]
    means = [acc.mean(), base.mean()]
    stds = [acc.std(), base.std()]
    ax.bar(names, means, yerr=stds, capsize=4, color=[CATEGORICAL[0], CATEGORICAL[1]])
    ax.axhline(chance, color=AXIS, linestyle="--", linewidth=1.2)
    ax.text(1.4, chance, f"chance = {chance:.3f}", color=INK_SECONDARY, fontsize=8, va="bottom")
    ax.set_ylim(0, 1)
    ax.set_ylabel("3-class accuracy", color=INK_SECONDARY)
    ax.set_title("Emotion-valence sanity classifier vs. chance", color=INK_PRIMARY, fontsize=11)
    style_axes(ax)
    fig.tight_layout()
    out = PLOTS_DIR / "classifier_vs_chance.png"
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    all_X, all_y, all_pid, clip_rates, amp_stats = [], [], [], {}, {}
    all_X_with_baseline, all_y_with_baseline = [], []

    for pid in PARTICIPANTS:
        sec, windows = load_emotion_section_uv(pid)
        if sec is None:
            continue
        filt = filter_continuous_uv(resample_uniform(sec, NATIVE_FS))
        X, y, clipped = epoch_section(sec, filt, windows)
        if X.shape[0] == 0:
            continue

        clip_rates[pid] = float(clipped.mean())
        X_clean, y_clean = X[~clipped], y[~clipped]

        # amplitude among the raw-clean epochs: a filter run over the whole
        # continuous section rings across a clipped plateau, so even epochs
        # with no clipped *sample* can still carry huge filtered amplitude if
        # they sit near a clipped stretch -- this is a second, independent QC
        # signal from clip rate alone.
        amp_stats[pid] = {
            "n_clean": int(len(y_clean)),
            "median_abs_uV": float(np.median(np.abs(X_clean))) if len(X_clean) else float("nan"),
            "frac_over_500uV": float((np.abs(X_clean).max(axis=(1, 2)) > 500).mean()) if len(X_clean) else float("nan"),
        }
        all_X_with_baseline.append(X_clean)
        all_y_with_baseline.append(y_clean)

        trial_mask = y_clean != 0
        all_X.append(X_clean[trial_mask])
        all_y.append(y_clean[trial_mask])
        all_pid.extend([pid] * trial_mask.sum())

    print("=== Clipping rate (raw-sample rail hits) per participant ===")
    for pid, rate in sorted(clip_rates.items(), key=lambda kv: -kv[1]):
        print(f"{pid}: {rate:.1%} of epochs clipped")

    print("\n=== Post-filter amplitude among raw-clean epochs (physiological EEG ~10-100uV) ===")
    for pid, s in sorted(amp_stats.items(), key=lambda kv: -kv[1]["median_abs_uV"]):
        print(f"{pid}: median|amp|={s['median_abs_uV']:.1f}uV  frac_epochs>500uV={s['frac_over_500uV']:.2f}  n={s['n_clean']}")

    X_all, y_all, pid_all = np.concatenate(all_X), np.concatenate(all_y), np.array(all_pid)
    print(f"\n{X_all.shape[0]} clean emotion-trial epochs remain")
    print(f"Amplitude (uV): mean={X_all.mean():.2f}  std={X_all.std():.2f}  "
          f"min={X_all.min():.1f}  max={X_all.max():.1f}  (physiological EEG ~ +-100uV)")

    bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 45)]
    band_names = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]

    X_baseline_incl = np.concatenate(all_X_with_baseline)
    y_baseline_incl = np.concatenate(all_y_with_baseline)
    freqs_plot, psd_plot = sps.welch(X_baseline_incl, fs=EPOCH_FS, nperseg=X_baseline_incl.shape[-1], axis=-1)
    plot_psd_by_label(freqs_plot, psd_plot, y_baseline_incl)
    plot_bandpower_by_label(freqs_plot, psd_plot, y_baseline_incl, bands, band_names)
    plot_cohort_qc(clip_rates, amp_stats)

    best_pid, worst_pid = min(clip_rates, key=clip_rates.get), max(clip_rates, key=clip_rates.get)
    for pid in {best_pid, worst_pid}:
        sec, _ = load_emotion_section_uv(pid)
        sec_uniform = resample_uniform(sec, NATIVE_FS)
        plot_raw_vs_filtered(pid, sec_uniform, filter_continuous_uv(sec_uniform))

    freqs, psd = sps.welch(X_all, fs=EPOCH_FS, nperseg=X_all.shape[-1], axis=-1)
    feat = np.concatenate([psd[..., (freqs >= lo) & (freqs <= hi)].mean(axis=-1) for lo, hi in bands], axis=1)
    feat = StandardScaler().fit_transform(np.log1p(feat))

    gkf = GroupKFold(n_splits=5)
    acc = cross_val_score(LogisticRegression(max_iter=2000), feat, y_all, groups=pid_all, cv=gkf, scoring="accuracy")
    base = cross_val_score(DummyClassifier(strategy="most_frequent"), feat, y_all, groups=pid_all, cv=gkf, scoring="accuracy")
    print(f"\nLogReg accuracy:  {acc.mean():.3f} +/- {acc.std():.3f}")
    print(f"Majority baseline: {base.mean():.3f} +/- {base.std():.3f}  (chance = 0.333)")
    plot_classifier_result(acc, base, chance=1 / 3)


if __name__ == "__main__":
    main()
