"""Label-strategy / target-selector layer over the canonical stress tables.

Per the plan (plans_for_data_labeling/stress_task.md §17, §45-48): the
canonical sample tables built here keep condition/behavior/subjective/
derived label groups as SEPARATE prefixed columns (`cond_*`, `beh_*`,
`subj_*`, `derived_*`) -- there is deliberately no single global `label`
column baked in at this layer. Each Dataset-Version builder in
`versions.py` then does the actual TARGET SELECTION for that specific
experiment on top of these tables. This module only builds the tables and
the small `TargetSelector` helper for picking one column out as
`model_target` once a version has actually chosen its target.

Physiological windowing (Phase 4 of the plan) is intentionally NOT done
here -- these are table/label-only samples, one row per block/trial/
obstacle/self-report, keyed by (participant_id, block_id, trial_or_obstacle_id)
so a later windowing pass can join biosignals in without recomputing any
label logic. This matches the plan's explicit "STOP FOR REVIEW" checkpoint
after Versions 1-5's label tables/distributions, before generating any
model-ready tensors.
"""

from __future__ import annotations

import pandas as pd


def _tier_to_int(x):
    return None if pd.isna(x) else int(x)


# ── block-level samples (raindrop + highway share this shape) ────────────
def build_block_samples(blocks_df: pd.DataFrame) -> pd.DataFrame:
    """One row per block (baseline or active/tier), for both `task in
    {"stress","attention_highway"}`. `cond_stress_induction_binary` is
    0=baseline/1=active -- named per plan §18 to avoid implying baseline
    pressure is a physiological-stress ground truth."""
    if blocks_df.empty:
        return blocks_df.copy()
    df = blocks_df.copy()
    df["cond_stress_induction_binary"] = (df["block_type"] == "active").astype(int)
    df["cond_task_pressure_tier"] = df["tier"].apply(_tier_to_int)
    df["cond_stress_induction_level"] = df.apply(
        lambda r: 0 if r["block_type"] == "baseline" else (int(r["tier"]) if pd.notna(r["tier"]) else None), axis=1
    )
    df = df.rename(
        columns={
            "spawn_interval_sec": "cond_spawn_interval_sec",
            "spawn_interval_min_sec": "cond_spawn_interval_min_sec",
            "spawn_interval_max_sec": "cond_spawn_interval_max_sec",
            "fall_duration_sec": "cond_fall_duration_sec",
            "concurrent_obstacles": "cond_concurrent_obstacles",
            "accuracy": "beh_accuracy",
            "avoided": "beh_avoided",
            "collisions": "beh_collisions",
            "protocol_version": "meta_protocol_version",
            "known_highway_bug": "meta_known_highway_bug",
            "focus_loss_count": "meta_focus_loss_count",
            "frame_drop_warning_count": "meta_frame_drop_warning_count",
        }
    )
    return df


# ── raindrop trial-level samples ─────────────────────────────────────────
def build_raindrop_trial_samples(trials_df: pd.DataFrame) -> pd.DataFrame:
    """Trial-level Raindrop samples (plan §46). `subj_*` columns are
    deliberately absent -- Raindrop has no questionnaire; see
    `assert_no_raindrop_subjective_labels` for the safety test this
    guarantees."""
    if trials_df.empty:
        return trials_df.copy()
    df = trials_df.copy()
    df["cond_task_pressure_tier"] = df["tier"].apply(_tier_to_int)
    df = df.rename(
        columns={
            "spawn_interval_sec": "cond_spawn_interval_sec",
            "fall_duration_sec": "cond_fall_duration_sec",
            "correct": "beh_correct",
            "missed": "beh_missed",
            "rt_sec": "beh_rt_sec",
            "deadline_fraction": "derived_deadline_fraction",
            "protocol_version": "meta_protocol_version",
        }
    )
    df["subjective_labels_available"] = False
    return df


def assert_no_raindrop_subjective_labels(raindrop_samples: pd.DataFrame) -> None:
    subj_cols = [c for c in raindrop_samples.columns if c.startswith("subj_")]
    assert not subj_cols, f"Raindrop sample table must never carry subjective columns, found: {subj_cols}"
    if "subjective_labels_available" in raindrop_samples.columns:
        assert (raindrop_samples["subjective_labels_available"] == False).all(), "Raindrop subjective_labels_available must always be False"  # noqa: E712


# ── highway obstacle-level samples ───────────────────────────────────────
def build_highway_obstacle_samples(obstacles_df: pd.DataFrame) -> pd.DataFrame:
    """Per-obstacle Highway behavior samples (plan §10-11, §24)."""
    if obstacles_df.empty:
        return obstacles_df.copy()
    df = obstacles_df.copy()
    df["cond_task_pressure_tier"] = df["tier"].apply(_tier_to_int)
    df = df.rename(
        columns={
            "fall_duration_sec": "cond_fall_duration_sec",
            "collision": "beh_collision",
            "near_miss": "beh_near_miss",
            "no_response": "beh_no_response",
            "response_attempted": "beh_response_attempted",
            "reaction_margin_sec": "beh_reaction_margin_sec",
            "normalized_reaction_margin": "derived_normalized_reaction_margin",
            "protocol_version": "meta_protocol_version",
            "known_highway_bug": "meta_known_highway_bug",
            "same_wave_duplicate_suspected": "meta_same_wave_duplicate_suspected",
        }
    )
    return df


# ── highway self-report (block-resolution) samples ───────────────────────
def build_highway_subjective_samples(self_report_df: pd.DataFrame) -> pd.DataFrame:
    """HIGHWAY ONLY, one row per questionnaire response (plan §13, §27).
    `label_resolution="block"` is set explicitly per plan §26/§36 -- many
    physiological windows within a tier will share this ONE questionnaire
    observation; that must not be double-counted as independent
    annotations downstream."""
    if self_report_df.empty:
        return self_report_df.copy()
    known_cols = {"participant_id", "session_id", "block_id", "tier", "prompt_timestamp", "response_timestamp", "skipped"}
    item_cols = [c for c in self_report_df.columns if c not in known_cols and not c.endswith("_rt")]

    df = self_report_df.copy()
    df["cond_task_pressure_tier"] = df["tier"].apply(_tier_to_int)
    for item in item_cols:
        df[f"subj_{item}"] = df[item]
    df["subjective_labels_available"] = True
    df["label_resolution"] = "block"
    return df


class TargetSelector:
    """Picks one column out of a sample table as `model_target`, without
    mutating or dropping the other label columns -- keeps the underlying
    biosignal window reusable across different target choices (plan §52)."""

    def __init__(self, column: str):
        self.column = column

    def apply(self, samples: pd.DataFrame) -> pd.DataFrame:
        if samples.empty:
            out = samples.copy()
            out["model_target"] = pd.Series(dtype="object")
            return out
        if self.column not in samples.columns:
            raise KeyError(f"target column {self.column!r} not found in sample table (columns: {list(samples.columns)})")
        out = samples.copy()
        out["model_target"] = out[self.column]
        return out
