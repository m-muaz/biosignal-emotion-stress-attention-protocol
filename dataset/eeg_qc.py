"""Per-epoch EEG signal-quality metrics.

Six literature-grounded heuristics, each producing a raw value plus a
boolean pass/fail against one fixed, documented threshold. No weighting or
composite score is computed -- only `n_checks_passed` / `n_checks_total`
(a simple count, mirroring FASTER's multi-criterion approach), per the
project's stance against arbitrary weighted composites (plans/stress_task.md
§32) applied here to signal QC as well. Nothing here drops an epoch; this
module only annotates. Downstream code decides whether/how to filter on it.

Metric sources:
  - clip_frac: rail-hit detection, same convention as
    analysis/eeg_continuous_preprocess.py's existing clip check.
  - dc_drift_uv: DC-offset / slow-drift heuristic (mean amplitude of the
    epoch, in uV) -- standard EEG QC practice (e.g. PREP pipeline's
    high-amplitude/drift channel criteria; Bigdely-Shamlo et al. 2015).
  - emg_hf_ratio: power(>20Hz) / power(1-40Hz) -- EMG/muscle contamination
    is spectrally dominant above ~20Hz (Goncharova et al. 2003, "EMG
    contamination of EEG: spectral and topographical characteristics").
  - line_noise_ratio: power in a narrow band around 60Hz vs. its
    immediate neighbors -- the PREP pipeline's line-noise detection
    approach (Bigdely-Shamlo et al. 2015).
  - snr_db: 10*log10(signal-band power / out-of-band residual power),
    the standard band-power SNR estimate used broadly in EEG QC
    (Scholarpedia: "Signal-to-noise ratio in neuroscience").
  - kurtosis_z: z-scored excess kurtosis vs. a cohort/participant
    baseline -- statistical outlier criterion from Delorme & Makeig 2004
    and used in FASTER (Nolan, Whelan & Reilly 2010).
"""
from __future__ import annotations

import numpy as np
from scipy import signal as sps
from scipy.stats import kurtosis

from dataset.eeg_preprocess import BANDPASS_HZ, CLIP_UV, LINE_FREQ_HZ

# Fixed, documented thresholds -- each independently motivated, not tuned as a set.
DC_DRIFT_MAX_UV = 100.0          # physiological EEG epoch mean should sit near 0uV; sustained DC/drift above this is atypical
EMG_HF_RATIO_MAX = 0.5           # >50% of 1-40Hz band power sitting above 20Hz indicates muscle contamination dominance
LINE_NOISE_RATIO_MAX = 3.0       # 60Hz-band power more than 3x its neighbors indicates uncancelled mains contamination
SNR_DB_MIN = 0.0                 # signal-band power should exceed the out-of-band residual (SNR >= 0dB)
KURTOSIS_Z_MAX = 5.0             # |z| > 5 is the standard FASTER/Delorme-Makeig outlier threshold
CLIP_FRAC_MAX = 0.0              # any rail-hit sample fails this check

EMG_SPLIT_HZ = 20.0
LINE_BAND_HALFWIDTH_HZ = 1.0
LINE_NEIGHBOR_HALFWIDTH_HZ = 4.0


def _bandpower(freqs: np.ndarray, psd: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs <= hi)
    if not mask.any():
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))


def score_channel_epoch(raw_uv: np.ndarray, filt_uv: np.ndarray, fs: float, kurtosis_ref: tuple[float, float] | None = None) -> dict:
    """One channel's one epoch. `raw_uv` = pre-filter uV samples (for clip
    detection only), `filt_uv` = post notch+bandpass uV samples (for
    everything else), both 1-D arrays over the same time span at `fs`.
    `kurtosis_ref` = (mean, std) of kurtosis across some reference
    population (e.g. this participant's other epochs) to z-score against;
    if None, kurtosis_z is left as NaN (not enough context to judge)."""
    metrics: dict = {}

    metrics["clip_frac"] = float(np.mean(np.abs(raw_uv) > CLIP_UV)) if raw_uv.size else float("nan")
    metrics["clip_frac_pass"] = bool(metrics["clip_frac"] <= CLIP_FRAC_MAX)

    metrics["dc_drift_uv"] = float(np.abs(np.mean(filt_uv))) if filt_uv.size else float("nan")
    metrics["dc_drift_pass"] = bool(metrics["dc_drift_uv"] <= DC_DRIFT_MAX_UV)

    if filt_uv.size >= 4:
        freqs, psd = sps.welch(filt_uv, fs=fs, nperseg=min(filt_uv.size, int(fs)))
        signal_power = _bandpower(freqs, psd, *BANDPASS_HZ)
        hf_power = _bandpower(freqs, psd, EMG_SPLIT_HZ, BANDPASS_HZ[1])
        total_power = float(np.trapz(psd, freqs)) if freqs.size else 0.0
        residual_power = max(total_power - signal_power, 0.0)

        metrics["emg_hf_ratio"] = (hf_power / signal_power) if signal_power > 0 else float("nan")
        metrics["emg_hf_ratio_pass"] = bool(not np.isnan(metrics["emg_hf_ratio"]) and metrics["emg_hf_ratio"] <= EMG_HF_RATIO_MAX)

        line_band = _bandpower(freqs, psd, LINE_FREQ_HZ - LINE_BAND_HALFWIDTH_HZ, LINE_FREQ_HZ + LINE_BAND_HALFWIDTH_HZ)
        lo_neighbor = _bandpower(freqs, psd, LINE_FREQ_HZ - LINE_NEIGHBOR_HALFWIDTH_HZ, LINE_FREQ_HZ - LINE_BAND_HALFWIDTH_HZ)
        hi_neighbor = _bandpower(freqs, psd, LINE_FREQ_HZ + LINE_BAND_HALFWIDTH_HZ, LINE_FREQ_HZ + LINE_NEIGHBOR_HALFWIDTH_HZ)
        neighbor_power = (lo_neighbor + hi_neighbor) / 2.0
        metrics["line_noise_ratio"] = (line_band / neighbor_power) if neighbor_power > 0 else float("nan")
        metrics["line_noise_pass"] = bool(not np.isnan(metrics["line_noise_ratio"]) and metrics["line_noise_ratio"] <= LINE_NOISE_RATIO_MAX)

        metrics["snr_db"] = (10 * np.log10(signal_power / residual_power)) if residual_power > 0 else float("inf")
        metrics["snr_pass"] = bool(metrics["snr_db"] >= SNR_DB_MIN)
    else:
        for k in ("emg_hf_ratio", "line_noise_ratio", "snr_db"):
            metrics[k] = float("nan")
        for k in ("emg_hf_ratio_pass", "line_noise_pass", "snr_pass"):
            metrics[k] = False

    if filt_uv.size >= 4:
        raw_kurt = float(kurtosis(filt_uv, fisher=True, bias=False))
    else:
        raw_kurt = float("nan")
    metrics["kurtosis"] = raw_kurt
    if kurtosis_ref is not None and kurtosis_ref[1] > 0 and not np.isnan(raw_kurt):
        metrics["kurtosis_z"] = float((raw_kurt - kurtosis_ref[0]) / kurtosis_ref[1])
        metrics["kurtosis_pass"] = bool(abs(metrics["kurtosis_z"]) <= KURTOSIS_Z_MAX)
    else:
        metrics["kurtosis_z"] = float("nan")
        metrics["kurtosis_pass"] = True  # no reference population to judge against -- don't fail by default

    pass_keys = ["clip_frac_pass", "dc_drift_pass", "emg_hf_ratio_pass", "line_noise_pass", "snr_pass", "kurtosis_pass"]
    metrics["n_checks_passed"] = int(sum(bool(metrics[k]) for k in pass_keys))
    metrics["n_checks_total"] = len(pass_keys)
    return metrics


def kurtosis_reference_from_epochs(filt_epochs: list[np.ndarray]) -> tuple[float, float] | None:
    """Mean/std of per-channel kurtosis across a population of epochs (e.g.
    one participant's whole set of epochs) -- feed the result into
    score_channel_epoch/score_epoch's `kurtosis_ref` so the kurtosis
    outlier check has something to compare against, instead of defaulting
    to "no reference, don't fail" for every epoch. Per-participant (rather
    than cohort-wide) since resting kurtosis varies with electrode contact
    quality, which is participant-specific. Returns None if there isn't
    enough data to estimate a meaningful std (min 2 finite values, std>0)."""
    values = [
        float(kurtosis(epoch[c], fisher=True, bias=False))
        for epoch in filt_epochs
        for c in range(epoch.shape[0])
        if epoch.shape[1] >= 4
    ]
    values = [v for v in values if not np.isnan(v)]
    if len(values) < 2:
        return None
    mean, std = float(np.mean(values)), float(np.std(values))
    return (mean, std) if std > 0 else None


def score_epoch(raw_uv: np.ndarray, filt_uv: np.ndarray, fs: float, kurtosis_ref: tuple[float, float] | None = None) -> dict:
    """Multi-channel epoch: `raw_uv`/`filt_uv` shape (n_channels, n_samples).
    Aggregates per-channel scores: worst-case (min) n_checks_passed across
    channels, plus per-channel breakdown under `channels`."""
    n_channels = raw_uv.shape[0]
    per_channel = [score_channel_epoch(raw_uv[c], filt_uv[c], fs, kurtosis_ref) for c in range(n_channels)]
    return {
        "n_checks_passed_min_over_channels": min(m["n_checks_passed"] for m in per_channel) if per_channel else 0,
        "n_checks_total": per_channel[0]["n_checks_total"] if per_channel else 0,
        "channels": per_channel,
    }
