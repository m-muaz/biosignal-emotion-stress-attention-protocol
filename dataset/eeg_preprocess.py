"""Shared ADC-counts -> uV conversion and standard EEG-band filtering for
the out-ear/in-ear ADS1299 channels.

Moved here (from analysis/eeg_continuous_preprocess.py, which now imports
these) so both the analysis/ scripts and the dataset/ export pipelines
(dataset/export_npy.py's emotion task, dataset/stress_labels/export_npy.py's
stress task) share one implementation instead of drifting apart.

Filtering an already-cut short epoch (e.g. with a 1Hz-highpass Butterworth)
is not standard practice -- the filter's settling time is comparable to the
epoch length, so edge transients dominate. The correct order is: convert ->
resample onto a uniform grid -> filter the *continuous* signal -> THEN cut
into epochs.

ADC->uV: confirmed from ADS1299_BLE_muaz/main/AD1299.c -- gain=24x
(AD1299_CHnSET_GAIN_24X, line 344), VREF=4.5V internal reference (line
336, "Set internal reference"). Differential FS = +-VREF/gain = +-187.5mV
mapped to signed 24-bit codes (+-2^23).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import signal as sps

CHANNELS = [f"ch{i}" for i in range(1, 9)]
NATIVE_FS = 250.0  # ADS1299 CONFIG1 SAMPLE_RATE_250SPS (AD1299.c); wall_utc_s spacing is jittered
                   # (~3.3-4.7ms, median ~248Hz) from clock sync, not evenly sampled -- must be
                   # resampled onto a uniform grid before filtfilt/iirnotch, which assume uniform sampling.

VREF, GAIN, FULL_SCALE_CODE = 4.5, 24, 2**23
UV_PER_CODE = (VREF / GAIN) / FULL_SCALE_CODE * 1e6  # ADC code -> microvolts
CLIP_MARGIN = 0.001  # within 0.1% of the ADC rail counts as clipped
CLIP_UV = FULL_SCALE_CODE * UV_PER_CODE * (1 - CLIP_MARGIN)

LINE_FREQ_HZ = 60.0  # North America mains
BANDPASS_HZ = (1.0, 40.0)


def counts_to_uv(raw_df: pd.DataFrame, channels: list[str] = CHANNELS) -> pd.DataFrame:
    """Raw ADC-code columns (int) -> microvolts (float64), same frame shape."""
    out = raw_df.copy()
    out[channels] = out[channels].astype(np.float64) * UV_PER_CODE
    return out


def resample_uniform(sec: pd.DataFrame, fs: float = NATIVE_FS, channels: list[str] = CHANNELS) -> pd.DataFrame:
    """Linearly interpolates onto an evenly-spaced wall_utc_s grid at `fs`
    -- required before any IIR filtering, since the synced timestamps are
    jittered (clock-sync correction), not natively evenly sampled."""
    wall = sec["wall_utc_s"].to_numpy()
    grid = np.arange(wall[0], wall[-1], 1.0 / fs)
    out = {"wall_utc_s": grid}
    for ch in channels:
        out[ch] = np.interp(grid, wall, sec[ch].to_numpy())
    return pd.DataFrame(out)


def filter_continuous_uv(sec_uniform: pd.DataFrame, fs: float = NATIVE_FS, channels: list[str] = CHANNELS) -> pd.DataFrame:
    """Standard EEG pipeline on the continuous, uniformly-resampled signal:
    60Hz notch (North America mains), then 1-40Hz zero-phase bandpass. Both
    applied once over the whole continuous section, not per-epoch."""
    notch_b, notch_a = sps.iirnotch(LINE_FREQ_HZ, Q=30.0, fs=fs)
    sos = sps.butter(4, list(BANDPASS_HZ), btype="bandpass", fs=fs, output="sos")
    out = sec_uniform.copy()
    for ch in channels:
        x = sps.filtfilt(notch_b, notch_a, sec_uniform[ch].to_numpy())
        out[ch] = sps.sosfiltfilt(sos, x)
    return out


def convert_and_filter(raw_df: pd.DataFrame, channels: list[str] = CHANNELS, fs: float = NATIVE_FS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Raw ADC-code frame -> (uv_raw_uniform, uv_filtered), both on the same
    uniform `wall_utc_s` grid at `fs`. `uv_raw_uniform` (pre-filter, post-
    resample) is what clip detection must run against -- a zero-phase
    filter smears a railed plateau's ringing across neighboring samples and
    erases its exact-rail signature. `uv_filtered` is what every spectral/
    amplitude QC metric and the model-input epochs should use."""
    uv = counts_to_uv(raw_df, channels)
    uv_uniform = resample_uniform(uv, fs, channels)
    uv_filtered = filter_continuous_uv(uv_uniform, fs, channels)
    return uv_uniform, uv_filtered
