"""Dataset Versions 1-5 (plan §53) -- the FIRST batch to implement, built
directly on top of `labels.py`'s canonical sample tables. Nothing past
Version 5 (stress deltas, quantile classes, high-confidence stress, etc.)
is implemented here -- plan §54 says STOP after these five for manual
review.

Each version function takes the concatenated (across participants)
canonical sample tables and returns one samples DataFrame ready to be
paired with a participant split + manifest.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dataset.stress_labels.labels import (
    TargetSelector,
    assert_no_raindrop_subjective_labels,
    build_block_samples,
    build_highway_obstacle_samples,
    build_highway_subjective_samples,
    build_raindrop_trial_samples,
)


def load_canonical_table(processed_dir: Path, participant_ids: list[str], table_name: str) -> pd.DataFrame:
    frames = []
    for pid in participant_ids:
        path = Path(processed_dir) / pid / f"{table_name}.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            if not df.empty:
                frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_all_canonical_tables(processed_dir: Path, participant_ids: list[str]) -> dict[str, pd.DataFrame]:
    names = ["stress_blocks", "raindrop_trials", "raindrop_unmatched_events", "highway_obstacles", "highway_state", "highway_self_report"]
    return {name: load_canonical_table(processed_dir, participant_ids, name) for name in names}


# ── Version 1: baseline vs active ────────────────────────────────────────
def version1a_raindrop_baseline_vs_active(tables: dict) -> pd.DataFrame:
    blocks = build_block_samples(tables["stress_blocks"])
    df = blocks[blocks["task"] == "stress"]
    return TargetSelector("cond_stress_induction_binary").apply(df)


def version1b_highway_baseline_vs_active(tables: dict, exclude_highway_pre_fix: bool = False) -> pd.DataFrame:
    blocks = build_block_samples(tables["stress_blocks"])
    df = blocks[blocks["task"] == "attention_highway"]
    if exclude_highway_pre_fix:
        df = df[df["meta_known_highway_bug"] != True]  # noqa: E712
    return TargetSelector("cond_stress_induction_binary").apply(df)


def version1c_combined_baseline_vs_active(tables: dict, exclude_highway_pre_fix: bool = False) -> pd.DataFrame:
    raindrop = version1a_raindrop_baseline_vs_active(tables)
    highway = version1b_highway_baseline_vs_active(tables, exclude_highway_pre_fix=exclude_highway_pre_fix)
    return pd.concat([raindrop, highway], ignore_index=True)


# ── Version 2: active pressure tier ──────────────────────────────────────
def version2a_raindrop_tier(tables: dict) -> pd.DataFrame:
    blocks = build_block_samples(tables["stress_blocks"])
    df = blocks[(blocks["task"] == "stress") & (blocks["block_type"] == "active")]
    return TargetSelector("cond_task_pressure_tier").apply(df)


def version2b_highway_tier(tables: dict, exclude_highway_pre_fix: bool = False) -> pd.DataFrame:
    blocks = build_block_samples(tables["stress_blocks"])
    df = blocks[(blocks["task"] == "attention_highway") & (blocks["block_type"] == "active")]
    if exclude_highway_pre_fix:
        df = df[df["meta_known_highway_bug"] != True]  # noqa: E712
    return TargetSelector("cond_task_pressure_tier").apply(df)


# ── Version 3: Raindrop behavioral dataset ───────────────────────────────
def version3_raindrop_behavior(tables: dict) -> pd.DataFrame:
    """Multiple valid targets (beh_correct / beh_missed / beh_rt_sec /
    derived_deadline_fraction) kept side by side, per plan §22/§44 -- no
    single `model_target` is forced at this layer."""
    samples = build_raindrop_trial_samples(tables["raindrop_trials"])
    assert_no_raindrop_subjective_labels(samples)
    return samples


# ── Version 4: Highway behavioral dataset ────────────────────────────────
def version4_highway_behavior(tables: dict, exclude_highway_pre_fix: bool = False) -> pd.DataFrame:
    samples = build_highway_obstacle_samples(tables["highway_obstacles"])
    if exclude_highway_pre_fix and not samples.empty:
        samples = samples[samples["meta_known_highway_bug"] != True]  # noqa: E712
    return samples


# ── Version 5: Highway subjective dataset ────────────────────────────────
def version5_highway_subjective(tables: dict) -> pd.DataFrame:
    """HIGHWAY ONLY -- built from highway_self_report, which by
    construction never contains Raindrop rows (self_report_prompt_onset/
    self_report_response only ever fire under task="attention_highway",
    see dataset/windows.py::extract_attention_highway_windows)."""
    samples = build_highway_subjective_samples(tables["highway_self_report"])
    return samples


VERSION_BUILDERS = {
    "s0a_raindrop_baseline_vs_active": version1a_raindrop_baseline_vs_active,
    "s0b_highway_baseline_vs_active": version1b_highway_baseline_vs_active,
    "s0c_combined_baseline_vs_active": version1c_combined_baseline_vs_active,
    "s1a_raindrop_tiers": version2a_raindrop_tier,
    "s1b_highway_tiers": version2b_highway_tier,
    "s4_raindrop_behavior": version3_raindrop_behavior,
    "s5_highway_behavior": version4_highway_behavior,
    "s6_highway_subjective": version5_highway_subjective,
}
