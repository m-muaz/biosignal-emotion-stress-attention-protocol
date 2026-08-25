"""Regression tests for dataset/export_npy.py's EEG handling: X.npy for the
EEG streams (ear_eeg_out.ads1299, ear_eeg_in.ads1299) must be ADC->uV
converted and notch/bandpass filtered -- NOT raw ADC counts -- and every
EEG epoch's meta.csv row must carry per-channel QC columns from
dataset/eeg_qc.py. Non-EEG streams must NOT gain QC columns they have no
pipeline for.

Integration-only (needs the real cohort's processed parquet, which is
machine-local and never committed -- see .gitignore's `data/` entry).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dataset.eeg_preprocess import FULL_SCALE_CODE
from dataset.export_npy import QC_META_COLUMNS, export_all

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
requires_real_data = pytest.mark.skipif(not PROCESSED_DIR.exists(), reason="data/processed not built locally")


def _first_participant_with_events() -> str:
    for p in sorted(PROCESSED_DIR.iterdir()):
        if p.is_dir() and (p / "events.parquet").exists() and (p / "ear_eeg_out.ads1299.parquet").exists():
            return p.name
    pytest.skip("no participant with events.parquet + ear_eeg_out.ads1299.parquet found")


@pytest.fixture(scope="module")
def emotion_export(tmp_path_factory):
    """One export_all run (emotion_only) shared across every test below --
    each test only reads its own output file, so there's no need to pay for
    a fresh export per assertion."""
    if not PROCESSED_DIR.exists():
        pytest.skip("data/processed not built locally")
    pid = _first_participant_with_events()
    out_dir = tmp_path_factory.mktemp("emotion_export")
    export_all(PROCESSED_DIR, out_dir, participant_ids=[pid], emotion_only=True, pool=False)
    return out_dir, pid


@requires_real_data
def test_emotion_trial_eeg_is_uv_scale_not_raw_adc(emotion_export):
    out_dir, pid = emotion_export
    X = np.load(out_dir / pid / "emotion_trial_X.npy")
    assert X.shape[0] > 0, "expected at least one emotion_trial epoch for this participant"

    # Raw ADC codes are signed 24-bit, i.e. up to ~8.4e6 in magnitude. A
    # correctly uV-converted + filtered EEG epoch should sit within a few
    # hundred uV for the vast majority of samples -- nowhere close to the
    # ADC's own full-scale code range.
    assert np.abs(X).max() < FULL_SCALE_CODE / 100, "X values are still ADC-code-scale, not uV -- conversion did not run"
    assert np.percentile(np.abs(X), 99) < 1000.0, "99th percentile amplitude implausibly high for filtered EEG in uV"
    assert abs(float(np.mean(X))) < 50.0, "filtered EEG should be near-zero-mean (DC/drift removed by the bandpass)"


@requires_real_data
def test_emotion_trial_eeg_meta_has_qc_columns(emotion_export):
    out_dir, pid = emotion_export
    meta = pd.read_csv(out_dir / pid / "emotion_trial_meta.csv")
    assert not meta.empty
    for col in QC_META_COLUMNS:
        assert col in meta.columns, f"missing QC column {col!r} on EEG meta.csv"
    assert meta["qc_n_checks_total"].iloc[0] == 6
    assert (meta["qc_n_checks_passed_min_over_channels"] <= meta["qc_n_checks_total"]).all()


@requires_real_data
def test_non_eeg_stream_meta_has_no_qc_columns(emotion_export):
    out_dir, pid = emotion_export
    ppg_meta_path = out_dir / pid / "emotion_trial_ppg_meta.csv"
    if not ppg_meta_path.exists():
        pytest.skip("no wristband.ppg data for this participant")
    meta = pd.read_csv(ppg_meta_path)
    assert not any(c.startswith("qc_") for c in meta.columns)
