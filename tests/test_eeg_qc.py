"""Unit tests for dataset/eeg_qc.py and dataset/eeg_preprocess.py, using
synthetic signals so each heuristic's pass/fail behavior is verified in
isolation rather than against real (noisy, ambiguous) cohort data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import json

from dataset.eeg_preprocess import CLIP_UV, UV_PER_CODE, FULL_SCALE_CODE, counts_to_uv
from dataset.eeg_qc import (
    DC_DRIFT_MAX_UV,
    EMG_HF_RATIO_MAX,
    LINE_NOISE_RATIO_MAX,
    SNR_DB_MIN,
    kurtosis_reference_from_epochs,
    score_channel_epoch,
    score_epoch,
)
from dataset.stress_labels.qc import QC_FLAG_COLUMNS, eeg_signal_qc_report

FS = 250.0
DURATION_S = 1.0
N = int(FS * DURATION_S)
T = np.arange(N) / FS


def _clean_alpha_uv(amplitude_uv: float = 20.0, freq_hz: float = 10.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    return amplitude_uv * np.sin(2 * np.pi * freq_hz * T) + rng.normal(0, 0.1, N)


def test_adc_to_uv_conversion_matches_datasheet_formula():
    raw = pd.DataFrame({"ch1": [0, FULL_SCALE_CODE // 2, -FULL_SCALE_CODE // 2]})
    out = counts_to_uv(raw, channels=["ch1"])
    assert out["ch1"].iloc[0] == pytest.approx(0.0)
    assert out["ch1"].iloc[1] == pytest.approx((FULL_SCALE_CODE // 2) * UV_PER_CODE)
    assert out["ch1"].iloc[2] == pytest.approx(-(FULL_SCALE_CODE // 2) * UV_PER_CODE)


def test_clean_signal_passes_all_checks():
    sig = _clean_alpha_uv()
    m = score_channel_epoch(sig, sig, FS)
    assert m["clip_frac_pass"]
    assert m["dc_drift_pass"]
    assert m["emg_hf_ratio_pass"]
    assert m["line_noise_pass"]
    assert m["snr_pass"]
    assert m["n_checks_passed"] == m["n_checks_total"]


def test_clipped_signal_fails_clip_check():
    sig = _clean_alpha_uv()
    railed = sig.copy()
    railed[:10] = CLIP_UV * 1.01
    m = score_channel_epoch(railed, sig, FS)
    assert not m["clip_frac_pass"]
    assert m["clip_frac"] > 0


def test_dc_offset_fails_drift_check():
    sig = _clean_alpha_uv() + 500.0  # large sustained offset, well above DC_DRIFT_MAX_UV
    m = score_channel_epoch(sig, sig, FS)
    assert not m["dc_drift_pass"]


def test_line_noise_fails_at_60hz():
    line = 200.0 * np.sin(2 * np.pi * 60.0 * T)
    m = score_channel_epoch(line, line, FS)
    assert not m["line_noise_pass"]


def test_high_frequency_dominant_signal_fails_emg_check():
    emg_like = 100.0 * np.sin(2 * np.pi * 35.0 * T)  # dominant power well above the 20Hz EMG split
    m = score_channel_epoch(emg_like, emg_like, FS)
    assert not m["emg_hf_ratio_pass"]


def test_low_snr_signal_fails_snr_check():
    rng = np.random.default_rng(1)
    pure_noise = rng.normal(0, 50.0, N)  # no coherent in-band signal, just broadband noise
    m = score_channel_epoch(pure_noise, pure_noise, FS)
    # broadband noise spreads power across the whole spectrum roughly evenly,
    # so signal-band (1-40Hz) power is not meaningfully above the out-of-band
    # residual -- this should not pass with a comfortable margin.
    assert m["snr_db"] < 10.0


def test_dc_drift_boundary_exactly_at_threshold_passes():
    sig = np.full(N, DC_DRIFT_MAX_UV)
    m = score_channel_epoch(sig, sig, FS)
    assert m["dc_drift_uv"] == pytest.approx(DC_DRIFT_MAX_UV)
    assert m["dc_drift_pass"]  # threshold is inclusive (<=)


def test_dc_drift_just_over_threshold_fails():
    sig = np.full(N, DC_DRIFT_MAX_UV + 1.0)
    m = score_channel_epoch(sig, sig, FS)
    assert not m["dc_drift_pass"]


def test_flatline_signal_is_not_silently_marked_clean():
    """A disconnected/flatlined channel has zero in-band power -- ratio
    metrics that divide by signal power become undefined (NaN). NaN must
    be treated as a fail, not a pass-by-default, since a total absence of
    detectable EEG signal is itself a quality problem worth surfacing."""
    flat = np.zeros(N)
    m = score_channel_epoch(flat, flat, FS)
    assert m["clip_frac_pass"]  # zero is not near the ADC rail
    assert m["dc_drift_pass"]  # zero mean
    assert not m["emg_hf_ratio_pass"]  # undefined ratio (0/0) -> fail
    assert not m["line_noise_pass"]  # undefined ratio -> fail


def test_short_epoch_forces_spectral_checks_to_fail_not_crash():
    short = _clean_alpha_uv()[:3]  # below the 4-sample minimum for welch()
    m = score_channel_epoch(short, short, FS)
    assert not m["emg_hf_ratio_pass"]
    assert not m["line_noise_pass"]
    assert not m["snr_pass"]
    assert np.isnan(m["emg_hf_ratio"])
    # clip/drift are simple time-domain stats and still well-defined on 3 samples
    assert isinstance(m["clip_frac_pass"], bool)
    assert isinstance(m["dc_drift_pass"], bool)


def test_multichannel_epoch_takes_worst_case_across_channels():
    clean = _clean_alpha_uv()
    railed = clean.copy()
    railed[:20] = CLIP_UV * 1.01
    raw = np.stack([clean, clean, railed])  # 3 channels, only the 3rd is bad
    result = score_epoch(raw, raw, fs=FS)
    assert len(result["channels"]) == 3
    assert result["channels"][0]["n_checks_passed"] == result["channels"][0]["n_checks_total"]
    assert result["channels"][2]["n_checks_passed"] < result["channels"][2]["n_checks_total"]
    # the epoch-level summary must reflect the WORST channel, not an average
    # or the best channel -- otherwise one bad electrode could be masked by
    # clean ones sharing the same epoch.
    assert result["n_checks_passed_min_over_channels"] == result["channels"][2]["n_checks_passed"]


def _fake_channels_json(n_checks_total: int = 6, all_pass: bool = True) -> str:
    ch = {f: all_pass for f in QC_FLAG_COLUMNS}
    ch["n_checks_passed"] = n_checks_total if all_pass else 0
    ch["n_checks_total"] = n_checks_total
    return json.dumps([ch])


def test_eeg_signal_qc_report_includes_baseline_nan_tier_epochs():
    """Regression test: baseline-block epochs have cond_task_pressure_tier
    == NaN (no tier applies), and pandas' groupby() drops NaN-key groups
    by default -- eeg_signal_qc_report must use dropna=False or every
    baseline epoch silently vanishes from the QC rollup."""
    meta = pd.DataFrame(
        {
            "participant_id": ["P001", "P001", "P001"],
            "cond_task_pressure_tier": [1.0, 2.0, float("nan")],  # last row = a baseline epoch
            "qc_n_checks_passed_min_over_channels": [6, 6, 6],
            "qc_n_checks_total": [6, 6, 6],
            "qc_channels_json": [_fake_channels_json(), _fake_channels_json(), _fake_channels_json()],
        }
    )
    report = eeg_signal_qc_report(meta)
    assert report["epoch_count"].sum() == 3  # all 3 rows accounted for, not just the 2 tiered ones
    assert report["cond_task_pressure_tier"].isna().any()  # the baseline (NaN-tier) row survived


def test_kurtosis_reference_from_epochs_flags_outlier_against_typical_population():
    rng = np.random.default_rng(2)
    typical_epochs = [
        (amp * np.sin(2 * np.pi * 10.0 * T) + rng.normal(0, 0.1, N)).reshape(1, N) for amp in rng.uniform(10, 30, 40)
    ]
    ref = kurtosis_reference_from_epochs(typical_epochs)
    assert ref is not None
    mean, std = ref
    assert std > 0

    # a single sharp spike in an otherwise-flat epoch has very different
    # (much higher) kurtosis than a population of clean sinusoids
    spiky = np.zeros((1, N))
    spiky[0, N // 2] = 500.0
    m_spiky = score_channel_epoch(spiky[0], spiky[0], FS, kurtosis_ref=ref)
    assert not m_spiky["kurtosis_pass"]

    # a fresh, similarly-shaped typical epoch should NOT be flagged against its own population's reference
    m_typical = score_channel_epoch(typical_epochs[0][0], typical_epochs[0][0], FS, kurtosis_ref=ref)
    assert m_typical["kurtosis_pass"]


def test_kurtosis_reference_returns_none_with_insufficient_data():
    assert kurtosis_reference_from_epochs([_clean_alpha_uv().reshape(1, N)]) is None
    assert kurtosis_reference_from_epochs([]) is None


def test_kurtosis_z_uses_reference_population_when_given():
    sig = _clean_alpha_uv()
    m_no_ref = score_channel_epoch(sig, sig, FS, kurtosis_ref=None)
    assert np.isnan(m_no_ref["kurtosis_z"])
    assert m_no_ref["kurtosis_pass"]  # no reference -> don't fail by default

    m_with_ref = score_channel_epoch(sig, sig, FS, kurtosis_ref=(m_no_ref["kurtosis"] + 50.0, 1.0))
    assert not np.isnan(m_with_ref["kurtosis_z"])
    assert not m_with_ref["kurtosis_pass"]  # far from the (deliberately shifted) reference mean
