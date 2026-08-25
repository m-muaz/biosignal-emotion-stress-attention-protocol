"""Unit tests for dataset/stress_labels/ (plan §56).

Synthetic-fixture tests run everywhere; a few integration tests read the
real cohort's canonical Parquet output under data/processed/ if present and
are skipped otherwise (that data is machine-local raw-data-derived output,
never committed -- see .gitignore's `data/` entry).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dataset.stress_labels import canonical_tables as ct
from dataset.stress_labels.config import load_all_configs
from dataset.stress_labels.labels import (
    assert_no_raindrop_subjective_labels,
    build_block_samples,
    build_highway_obstacle_samples,
    build_highway_subjective_samples,
    build_raindrop_trial_samples,
)
from dataset.stress_labels.protocol import highway_known_bug, highway_protocol_version
from dataset.stress_labels.splits import ParticipantSplit, split_by_participant
from dataset.stress_labels.versions import VERSION_BUILDERS

REPO_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = REPO_ROOT / "data" / "processed"
requires_real_data = pytest.mark.skipif(not PROCESSED_DIR.exists(), reason="data/processed not built locally")


# ── protocol / bug boundary ───────────────────────────────────────────────
def test_highway_pre_fix_participants_exact():
    assert highway_protocol_version("P001") == "highway_pre_fix"
    assert highway_protocol_version("P004") == "highway_pre_fix"
    assert highway_known_bug("P004") is True
    assert highway_protocol_version("P005") == "highway_post_fix"
    assert highway_known_bug("P005") is False


# ── raindrop: deadline_fraction ───────────────────────────────────────────
def _make_stress_events(rt_sec, fall_duration_sec, tier=1, correct=True, missed=False):
    rows = [
        {
            "timestamp_host_utc": 100.0,
            "session_id": "s1",
            "task": "stress",
            "block_index": 1,
            "trial_index": 0,
            "condition_label": f"tier_{tier}",
            "event_type": "trial_start",
            "event_payload_json": {},
        },
        {
            "timestamp_host_utc": 100.0 + (rt_sec or 0),
            "session_id": "s1",
            "task": "stress",
            "block_index": 1,
            "trial_index": 0,
            "condition_label": f"tier_{tier}",
            "event_type": "response",
            "event_payload_json": {
                "tier": tier,
                "expression": "1 + 1",
                "correct_answer": 2,
                "participant_answer": 2 if correct else 3,
                "rt": rt_sec,
                "correct": correct,
                "missed": missed,
            },
        },
    ]
    return pd.DataFrame(rows)


def _config_with_stress_tier(tier, fall_duration_sec, spawn_interval_sec=3.0):
    return {"stress_task": {"tiers": [{"id": tier, "spawn_interval_sec": spawn_interval_sec, "fall_duration_sec": fall_duration_sec}]}}


def test_deadline_fraction_matches_formula():
    events = _make_stress_events(rt_sec=1.5, fall_duration_sec=3.0, tier=1)
    config = _config_with_stress_tier(1, fall_duration_sec=3.0)
    trials = ct.build_raindrop_trials("P999", "s1", events, config)
    assert trials.loc[0, "deadline_fraction"] == pytest.approx(1.5 / 3.0)


def test_deadline_fraction_null_on_missed_trial():
    events = _make_stress_events(rt_sec=None, fall_duration_sec=3.0, tier=1, correct=False, missed=True)
    config = _config_with_stress_tier(1, fall_duration_sec=3.0)
    trials = ct.build_raindrop_trials("P999", "s1", events, config)
    assert pd.isna(trials.loc[0, "deadline_fraction"])
    assert bool(trials.loc[0, "missed"]) is True


# ── highway: normalized_reaction_margin ───────────────────────────────────
def _make_highway_obstacle_events(reaction_margin_sec, fall_duration_sec, outcome="avoided", tier=1, near_miss=False, no_response=False):
    rows = [
        {
            "timestamp_host_utc": 200.0,
            "session_id": "s1",
            "task": "attention_highway",
            "block_index": 1,
            "trial_index": 0,
            "condition_label": f"tier_{tier}",
            "event_type": "obstacle_spawn",
            "event_payload_json": {"lane": 1, "fall_duration_sec": fall_duration_sec, "spawn_wave_index": 0, "spawn_wave_size": 1, "camp_nudged": False},
        },
        {
            "timestamp_host_utc": 200.0 + fall_duration_sec,
            "session_id": "s1",
            "task": "attention_highway",
            "block_index": 1,
            "trial_index": 0,
            "condition_label": f"tier_{tier}",
            "event_type": "obstacle_resolved",
            "event_payload_json": {
                "outcome": outcome,
                "near_miss": near_miss,
                "no_response": no_response,
                "response_attempted": not no_response,
                "reaction_margin_sec": reaction_margin_sec,
            },
        },
    ]
    return pd.DataFrame(rows)


def test_normalized_reaction_margin_matches_formula():
    events = _make_highway_obstacle_events(reaction_margin_sec=0.6, fall_duration_sec=2.0)
    obstacles = ct.build_highway_obstacles("P999", "s1", events)
    assert obstacles.loc[0, "normalized_reaction_margin"] == pytest.approx(0.6 / 2.0)


def test_collision_and_avoided_derived_from_outcome():
    events = _make_highway_obstacle_events(reaction_margin_sec=0.1, fall_duration_sec=2.0, outcome="collision")
    obstacles = ct.build_highway_obstacles("P999", "s1", events)
    assert bool(obstacles.loc[0, "collision"]) is True
    assert bool(obstacles.loc[0, "avoided"]) is False


def test_no_response_case():
    events = _make_highway_obstacle_events(reaction_margin_sec=None, fall_duration_sec=2.0, outcome="collision", no_response=True)
    obstacles = ct.build_highway_obstacles("P999", "s1", events)
    assert bool(obstacles.loc[0, "no_response"]) is True
    assert pd.isna(obstacles.loc[0, "normalized_reaction_margin"])


# ── highway self-report only ever attaches to attention_highway ──────────
def test_self_report_never_from_raindrop_task():
    rows = [
        {
            "timestamp_host_utc": 300.0,
            "session_id": "s1",
            "task": "stress",
            "block_index": 1,
            "trial_index": None,
            "condition_label": "tier_1",
            "event_type": "self_report_prompt_onset",
            "event_payload_json": {},
        },
        {
            "timestamp_host_utc": 301.0,
            "session_id": "s1",
            "task": "stress",
            "block_index": 1,
            "trial_index": None,
            "condition_label": "tier_1",
            "event_type": "self_report_response",
            "event_payload_json": {"tier": 1, "stress": 5},
        },
    ]
    events = pd.DataFrame(rows)
    config = {"attention_highway_task": {"self_report": {"items": ["stress"]}}}
    sr = ct.build_highway_self_report("P999", "s1", events, config)
    assert sr.empty  # task="stress" rows are never picked up by the attention_highway filter


def test_self_report_tier_matches_its_own_block_only():
    rows = []
    for tier in (1, 2, 3):
        rows.append(
            {
                "timestamp_host_utc": 300.0 + tier,
                "session_id": "s1",
                "task": "attention_highway",
                "block_index": tier,
                "trial_index": None,
                "condition_label": f"tier_{tier}",
                "event_type": "self_report_prompt_onset",
                "event_payload_json": {},
            }
        )
        rows.append(
            {
                "timestamp_host_utc": 300.5 + tier,
                "session_id": "s1",
                "task": "attention_highway",
                "block_index": tier,
                "trial_index": None,
                "condition_label": f"tier_{tier}",
                "event_type": "self_report_response",
                "event_payload_json": {"tier": tier, "stress": tier * 2},
            }
        )
    events = pd.DataFrame(rows)
    config = {"attention_highway_task": {"self_report": {"items": ["stress"]}}}
    sr = ct.build_highway_self_report("P999", "s1", events, config)
    assert len(sr) == 3
    for _, row in sr.iterrows():
        assert row["stress"] == row["tier"] * 2


# ── Raindrop subjective-label safety test ─────────────────────────────────
def test_raindrop_samples_never_carry_subjective_labels():
    events = _make_stress_events(rt_sec=1.0, fall_duration_sec=3.0)
    config = _config_with_stress_tier(1, fall_duration_sec=3.0)
    trials = ct.build_raindrop_trials("P999", "s1", events, config)
    samples = build_raindrop_trial_samples(trials)
    assert_no_raindrop_subjective_labels(samples)  # must not raise


def test_raindrop_subjective_safety_catches_a_regression():
    df = pd.DataFrame({"subj_stress": [1, 2, 3]})
    with pytest.raises(AssertionError):
        assert_no_raindrop_subjective_labels(df)


# ── block samples: baseline vs active binary target ───────────────────────
def test_block_samples_baseline_vs_active_binary():
    blocks = pd.DataFrame(
        [
            {"task": "stress", "block_type": "baseline", "tier": None, "block_id": 0},
            {"task": "stress", "block_type": "active", "tier": 2, "block_id": 1},
        ]
    )
    samples = build_block_samples(blocks)
    assert list(samples["cond_stress_induction_binary"]) == [0, 1]
    assert samples.loc[1, "cond_task_pressure_tier"] == 2


# ── participant-level splits ───────────────────────────────────────────────
def test_split_by_participant_is_disjoint_and_covers_everyone():
    ids = [f"P{i:03d}" for i in range(1, 18)]
    split = split_by_participant(ids, seed=42)
    split.assert_disjoint()
    assert sorted(split.train + split.val + split.test) == sorted(ids)


def test_split_disjoint_assertion_catches_overlap():
    bad = ParticipantSplit(train=["P001", "P002"], val=["P002"], test=["P003"])
    with pytest.raises(AssertionError):
        bad.assert_disjoint()


def test_split_is_deterministic_given_seed():
    ids = [f"P{i:03d}" for i in range(1, 18)]
    a = split_by_participant(ids, seed=7)
    b = split_by_participant(ids, seed=7)
    assert a.to_dict() == b.to_dict()


# ── Phase 6: YAML config layer (mirrors dataset.attention_labels.config) ─
def test_all_stress_configs_load_and_reference_a_real_version_builder():
    configs = load_all_configs()
    assert set(configs) == set(VERSION_BUILDERS)
    for name, cfg in configs.items():
        assert cfg.version_builder in VERSION_BUILDERS
        assert cfg.name == name


# ── integration tests against the real cohort, if built locally ──────────
@requires_real_data
def test_real_cohort_no_highway_bug_flag_outside_first_four():
    for pid_dir in sorted(PROCESSED_DIR.iterdir()):
        blocks_path = pid_dir / "stress_blocks.parquet"
        if not blocks_path.exists():
            continue
        df = pd.read_parquet(blocks_path)
        hw = df[df["task"] == "attention_highway"]
        if hw.empty:
            continue
        expected = highway_known_bug(pid_dir.name)
        assert bool(hw["known_highway_bug"].iloc[0]) == expected
