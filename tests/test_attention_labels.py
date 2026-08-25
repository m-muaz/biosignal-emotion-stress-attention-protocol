"""Unit tests for dataset/attention_labels/ (plans/attention_task.md §63-67).

Synthetic-fixture tests run everywhere; a few integration tests read the
real cohort's canonical Parquet output under data/processed/ if present and
are skipped otherwise (that data is machine-local raw-data-derived output,
never committed -- see .gitignore's `data/` entry).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dataset.attention_labels import canonical_tables as ct
from dataset.attention_labels.config import WindowSpec, load_all_configs
from dataset.attention_labels.export_npy import compute_window_bounds

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
requires_real_data = pytest.mark.skipif(not PROCESSED_DIR.exists(), reason="data/processed not built locally")


def _row(ts, task, event_type, payload, block_index=None, trial_index=None, condition_label=None):
    return {
        "timestamp_host_utc": ts,
        "session_id": "s1",
        "task": task,
        "block_index": block_index,
        "trial_index": trial_index,
        "condition_label": condition_label,
        "event_type": event_type,
        "event_payload_json": payload,
    }


# ── SART: trial-type / error reconstruction (plan §63) ───────────────────
def _make_sart_trial(t0, digit, is_omit, responded, rt, accurate, trial_index=0):
    return [
        _row(t0, "sart", "trial_start", {"number": digit, "is_omit": is_omit, "practice": False}, block_index=1, trial_index=trial_index),
        _row(t0 + 1.16, "sart", "response", {"responded": responded, "rt": rt, "accurate": accurate}, block_index=1, trial_index=trial_index),
    ]


def test_sart_go_with_response_is_correct():
    events = pd.DataFrame(_make_sart_trial(100.0, digit=3, is_omit=False, responded=True, rt=0.3, accurate=True))
    df = ct.build_sart_trials("P999", "s1", events)
    row = df.iloc[0]
    assert row["trial_type"] == "go"
    assert row["correct"] == True
    assert row["commission_error"] == False
    assert row["omission_error"] == False
    assert row["accuracy_mismatch"] == False


def test_sart_go_without_response_is_omission():
    events = pd.DataFrame(_make_sart_trial(100.0, digit=3, is_omit=False, responded=False, rt=None, accurate=False))
    df = ct.build_sart_trials("P999", "s1", events)
    row = df.iloc[0]
    assert row["correct"] == False
    assert row["omission_error"] == True
    assert row["commission_error"] == False


def test_sart_no_go_with_response_is_commission():
    events = pd.DataFrame(_make_sart_trial(100.0, digit=6, is_omit=True, responded=True, rt=0.4, accurate=False))
    df = ct.build_sart_trials("P999", "s1", events)
    row = df.iloc[0]
    assert row["trial_type"] == "no_go"
    assert row["correct"] == False
    assert row["commission_error"] == True
    assert row["omission_error"] == False


def test_sart_no_go_without_response_is_correct():
    events = pd.DataFrame(_make_sart_trial(100.0, digit=6, is_omit=True, responded=False, rt=None, accurate=True))
    df = ct.build_sart_trials("P999", "s1", events)
    row = df.iloc[0]
    assert row["correct"] == True
    assert row["commission_error"] == False
    assert row["omission_error"] == False


def test_sart_reconstructed_response_timestamp():
    events = pd.DataFrame(_make_sart_trial(100.0, digit=3, is_omit=False, responded=True, rt=0.481, accurate=True))
    df = ct.build_sart_trials("P999", "s1", events)
    row = df.iloc[0]
    assert row["reconstructed_response_timestamp"] == pytest.approx(100.0 + 0.481, abs=1e-9)


def test_sart_accuracy_mismatch_flagged_not_silently_resolved():
    # is_omit + responded => commission error => reconstructed correct=False,
    # but the logged `accurate` field (deliberately) disagrees.
    events = pd.DataFrame(_make_sart_trial(100.0, digit=6, is_omit=True, responded=True, rt=0.4, accurate=True))
    df = ct.build_sart_trials("P999", "s1", events)
    row = df.iloc[0]
    assert row["correct"] == False
    assert row["accurate"] == True
    assert row["accuracy_mismatch"] == True


# ── Stroop: congruency / correctness / RT parsing (plan §64) ─────────────
def _make_stroop_repetition(t0, trials):
    rows = [_row(t0, "attention_focus_stroop", "task_start", {})]
    ts = t0 + 1.0
    for i, (word, ink, congruent, clicked, correct, rt_ms) in enumerate(trials):
        rows.append(
            _row(ts, "attention_focus_stroop", "trial_start", {"word": word, "ink_color": ink, "congruent": congruent}, block_index=1, trial_index=i, condition_label="stroop_trial")
        )
        rows.append(
            _row(ts + rt_ms / 1000.0, "attention_focus_stroop", "response", {"clicked_color": clicked, "correct": correct, "rt_ms": rt_ms}, block_index=1, trial_index=i, condition_label="stroop_trial")
        )
        ts += rt_ms / 1000.0 + 0.5
    rows.append(_row(ts + 0.1, "attention_focus_stroop", "task_end", {}))
    return rows


def test_stroop_congruent_and_incongruent_parsing():
    trials = [("red", "red", True, "red", True, 800.0), ("blue", "green", False, "green", True, 900.0)]
    events = pd.DataFrame(_make_stroop_repetition(200.0, trials))
    df = ct.build_stroop_trials("P999", "s1", events)
    assert list(df["congruent"]) == [True, False]
    assert list(df["correct"]) == [True, True]


def test_stroop_rt_ms_to_sec_conversion():
    trials = [("red", "red", True, "red", True, 731.0)]
    events = pd.DataFrame(_make_stroop_repetition(200.0, trials))
    df = ct.build_stroop_trials("P999", "s1", events)
    row = df.iloc[0]
    assert row["rt_ms"] == 731.0
    assert row["rt_sec"] == pytest.approx(0.731, abs=1e-9)


def test_stroop_missing_rt_is_not_silently_zero():
    trials = [("red", "red", True, None, False, None)]
    rows = _make_stroop_repetition(200.0, [])
    # build manually: a response with rt_ms=None must not become rt_sec=0.
    rows = [
        _row(200.0, "attention_focus_stroop", "task_start", {}),
        _row(201.0, "attention_focus_stroop", "trial_start", {"word": "red", "ink_color": "red", "congruent": True}, block_index=1, trial_index=0, condition_label="stroop_trial"),
        _row(202.0, "attention_focus_stroop", "response", {"clicked_color": None, "correct": False, "rt_ms": None}, block_index=1, trial_index=0, condition_label="stroop_trial"),
        _row(202.1, "attention_focus_stroop", "task_end", {}),
    ]
    events = pd.DataFrame(rows)
    df = ct.build_stroop_trials("P999", "s1", events)
    row = df.iloc[0]
    assert row["rt_ms"] is None
    assert row["rt_sec"] is None


# ── Schulte: click-to-block assignment + inter-click interval (plan §65) ─
def test_schulte_click_assignment_and_inter_click_interval():
    blocks = pd.DataFrame(
        [
            {"participant_id": "P999", "task": "attention_focus_schulte", "block_id": 1, "repetition_index": 0, "block_start": 100.0, "block_end": 150.0},
            {"participant_id": "P999", "task": "attention_focus_schulte", "block_id": 1, "repetition_index": 1, "block_start": 200.0, "block_end": 250.0},
        ]
    )
    rows = [
        _row(101.0, "attention_focus_schulte", "cell_click", {"number_clicked": 1, "expected_number": 1, "correct": True, "rt_ms": 1000.0}),
        _row(103.0, "attention_focus_schulte", "cell_click", {"number_clicked": 2, "expected_number": 2, "correct": True, "rt_ms": 3000.0}),
        _row(105.0, "attention_focus_schulte", "cell_click", {"number_clicked": 3, "expected_number": 3, "correct": True, "rt_ms": 5000.0}),
        # outside every block span -- must not be assigned to a block.
        _row(999.0, "attention_focus_schulte", "cell_click", {"number_clicked": 99, "expected_number": 99, "correct": True, "rt_ms": None}),
    ]
    events = pd.DataFrame(rows)
    clicks = ct.build_schulte_clicks("P999", "s1", events, blocks)

    assert len(clicks) == 3
    assert clicks.attrs["unassigned_click_count"] == 1
    assert clicks.attrs["ambiguous_click_count"] == 0
    assert list(clicks["repetition_index"]) == [0, 0, 0]

    first, second, third = clicks.iloc[0], clicks.iloc[1], clicks.iloc[2]
    assert first["first_click_latency_sec"] == pytest.approx(1.0, abs=1e-9)
    assert second["inter_click_interval_sec"] == pytest.approx(2.0, abs=1e-9)
    assert third["inter_click_interval_sec"] == pytest.approx(2.0, abs=1e-9)


def test_schulte_click_never_assigned_to_multiple_blocks():
    # overlapping block spans -- ambiguous_block_match should be flagged, not
    # silently duplicated into two rows.
    blocks = pd.DataFrame(
        [
            {"participant_id": "P999", "task": "attention_focus_schulte", "block_id": 1, "repetition_index": 0, "block_start": 100.0, "block_end": 150.0},
            {"participant_id": "P999", "task": "attention_focus_schulte", "block_id": 1, "repetition_index": 1, "block_start": 120.0, "block_end": 160.0},
        ]
    )
    events = pd.DataFrame([_row(130.0, "attention_focus_schulte", "cell_click", {"number_clicked": 1, "expected_number": 1, "correct": True, "rt_ms": 1000.0})])
    clicks = ct.build_schulte_clicks("P999", "s1", events, blocks)
    assert len(clicks) == 1
    assert clicks.attrs["ambiguous_click_count"] == 1


# ── Phase 4: window computation / leakage guard (plan §67) ───────────────
def test_concurrent_window_bounds():
    spec = WindowSpec(mode="concurrent", anchor_col="stimulus_timestamp", pre_sec=0.0, post_sec=1.0)
    row = pd.Series({"stimulus_timestamp": 100.0})
    t_start, t_end = compute_window_bounds(row, spec)
    assert (t_start, t_end) == (100.0, 101.0)


def test_prospective_window_ends_at_or_before_anchor():
    spec = WindowSpec(mode="prospective_pre_stimulus", anchor_col="stimulus_timestamp", pre_sec=5.0, post_sec=0.0)
    row = pd.Series({"stimulus_timestamp": 100.0})
    t_start, t_end = compute_window_bounds(row, spec)
    assert t_start == pytest.approx(95.0)
    assert t_end <= 100.0


def test_prospective_window_raises_on_leakage():
    # post_sec > 0 would push the window past the event it's supposed to
    # predict -- this must be caught, not silently allowed through.
    spec = WindowSpec(mode="prospective_pre_stimulus", anchor_col="stimulus_timestamp", pre_sec=5.0, post_sec=0.5)
    row = pd.Series({"stimulus_timestamp": 100.0})
    with pytest.raises(ValueError):
        compute_window_bounds(row, spec)


def test_pre_click_window_ends_at_or_before_click():
    spec = WindowSpec(mode="pre_click", anchor_col="click_timestamp", pre_sec=2.0, post_sec=0.0)
    row = pd.Series({"click_timestamp": 50.0})
    t_start, t_end = compute_window_bounds(row, spec)
    assert t_start == pytest.approx(48.0)
    assert t_end <= 50.0


def test_block_window_uses_start_end_columns_directly():
    spec = WindowSpec(mode="block", start_col="block_start", end_col="block_end")
    row = pd.Series({"block_start": 10.0, "block_end": 40.0})
    assert compute_window_bounds(row, spec) == (10.0, 40.0)


def test_window_spec_requires_anchor_col_for_non_block_modes():
    with pytest.raises(ValueError):
        WindowSpec(mode="concurrent")


def test_window_spec_requires_start_end_cols_for_block_mode():
    with pytest.raises(ValueError):
        WindowSpec(mode="block")


def test_all_six_configs_load_and_parse():
    configs = load_all_configs()
    assert set(configs) == {
        "a1_sart_behavior",
        "a2_sart_prospective_lapse",
        "a3_stroop_behavior",
        "a4_stroop_rt_residual",
        "a5_schulte_block",
        "a6_schulte_click",
    }
    for name, cfg in configs.items():
        assert cfg.version_builder
        assert cfg.windowing.mode


# ── integration: real cohort canonical tables, if built locally ──────────
@requires_real_data
def test_real_cohort_sart_has_no_accuracy_mismatch():
    import glob

    total, mismatches = 0, 0
    for f in glob.glob(str(PROCESSED_DIR / "P*" / "sart_trials.parquet")):
        df = pd.read_parquet(f)
        if df.empty:
            continue
        total += len(df)
        mismatches += int(df["accuracy_mismatch"].sum())
    assert total > 0
    assert mismatches == 0
