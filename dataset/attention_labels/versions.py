"""Dataset Versions 1-6 (plans/attention_task.md §60) -- the FIRST batch to
implement. Nothing past Version 6 (cross-task attention score, low/medium/
high labels, etc.) is implemented here -- plan §61/Rule 25 says STOP after
these six for manual review.

Each version function takes the concatenated (across participants)
canonical sample tables and returns one samples DataFrame. Versions 1/3/5
deliberately do NOT force a single `model_target` (plan Rule 10: "do not
make rare Stroop errors the only primary target" generalizes to every
behavioral-dataset version here) -- multiple `beh_*`/`cond_*`/`derived_*`
columns are kept side by side; a caller picks one at train time via
TargetSelector.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dataset.attention_labels.labels import (
    build_sart_trial_samples,
    build_schulte_block_samples,
    build_schulte_click_samples,
    build_stroop_trial_samples,
)
from dataset.stress_labels.splits import ParticipantSplit

CANONICAL_TABLE_NAMES = ["attention_blocks", "sart_trials", "stroop_trials", "schulte_blocks", "schulte_clicks"]


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
    return {name: load_canonical_table(processed_dir, participant_ids, name) for name in CANONICAL_TABLE_NAMES}


# ── Version 1: SART canonical behavioral dataset (concurrent) ────────────
def version1_sart_behavior(tables: dict, history_n: int | None = None, minimum_history: int | None = None) -> pd.DataFrame:
    kwargs = {k: v for k, v in {"history_n": history_n, "minimum_history": minimum_history}.items() if v is not None}
    samples = build_sart_trial_samples(tables["sart_trials"], **kwargs)
    if not samples.empty:
        samples["meta_window_mode"] = "concurrent"
    return samples


# ── Version 2: SART prospective lapse-prediction dataset ─────────────────
def version2_sart_prospective_lapse(tables: dict, history_n: int | None = None, minimum_history: int | None = None) -> pd.DataFrame:
    """Pre-stimulus physiological windows must end before the stimulus
    they're predicting (plan §31 Rule 23): `window_end <= stimulus_
    timestamp`. That's a windowing-layer (Phase 4) constraint on the
    biosignal slice, not on this label table -- what this table does is
    drop trials with no next-trial outcome to predict (the last trial of
    each block) and tag `meta_window_mode` so Phase 4 knows to apply it."""
    kwargs = {k: v for k, v in {"history_n": history_n, "minimum_history": minimum_history}.items() if v is not None}
    samples = build_sart_trial_samples(tables["sart_trials"], **kwargs)
    if samples.empty:
        return samples
    samples = samples[samples["derived_next_trial_rt"].notna() | samples["derived_next_trial_error"].notna()].copy()
    samples["meta_window_mode"] = "prospective_pre_stimulus"
    return samples


# ── Version 3: Stroop behavioral dataset ──────────────────────────────────
def version3_stroop_behavior(tables: dict) -> pd.DataFrame:
    samples = build_stroop_trial_samples(tables["stroop_trials"])
    if not samples.empty:
        samples["meta_window_mode"] = "stimulus_locked"
    return samples


# ── Version 4: Stroop condition-adjusted RT residual dataset ─────────────
def version4_stroop_rt_residual(tables: dict, split: ParticipantSplit) -> pd.DataFrame:
    """expected_rt = mean RT per (congruency, repetition_index), FIT ON
    TRAIN PARTICIPANTS ONLY (plan §18-19 Rule 11/17), then applied frozen
    to every participant. Falls back to the train-wide mean RT for any
    (congruency, repetition_index) combination unseen in train."""
    samples = build_stroop_trial_samples(tables["stroop_trials"])
    if samples.empty:
        return samples
    samples["meta_window_mode"] = "stimulus_locked"

    train_df = samples[samples["participant_id"].isin(split.train) & samples["beh_rt_sec"].notna()]
    fit_table = train_df.groupby(["cond_congruency", "repetition_index"])["beh_rt_sec"].mean()
    fallback = train_df["beh_rt_sec"].mean()

    def _expected_rt(row):
        key = (row["cond_congruency"], row["repetition_index"])
        return fit_table.get(key, fallback)

    samples["derived_stroop_expected_rt"] = samples.apply(_expected_rt, axis=1)
    samples["derived_stroop_rt_residual"] = samples["beh_rt_sec"] - samples["derived_stroop_expected_rt"]
    samples["meta_rt_residual_fit_participant_ids"] = [sorted(split.train)] * len(samples)
    return samples


# ── Version 5: Schulte block behavioral dataset ───────────────────────────
def version5_schulte_block(tables: dict) -> pd.DataFrame:
    samples = build_schulte_block_samples(tables["schulte_blocks"])
    if not samples.empty:
        samples["meta_window_mode"] = "block"
    return samples


# ── Version 6: Schulte click-level dataset ────────────────────────────────
def version6_schulte_click(tables: dict, require_high_confidence: bool = True) -> pd.DataFrame:
    """Plan §36/Rule 14: do not manufacture click-level labels if
    reconstruction is unreliable. Rows flagged
    meta_reconstruction_confidence != "high" are excluded by default
    (require_high_confidence=True) rather than silently kept."""
    samples = build_schulte_click_samples(tables["schulte_clicks"])
    if samples.empty:
        return samples
    if require_high_confidence:
        samples = samples[samples["meta_reconstruction_confidence"] == "high"].copy()
    samples["meta_window_mode"] = "pre_click"
    return samples


VERSION_BUILDERS = {
    "a1_sart_behavior": version1_sart_behavior,
    "a2_sart_prospective_lapse": version2_sart_prospective_lapse,
    "a3_stroop_behavior": version3_stroop_behavior,
    "a4_stroop_rt_residual": version4_stroop_rt_residual,
    "a5_schulte_block": version5_schulte_block,
    "a6_schulte_click": version6_schulte_click,
}

# version builders that need the participant split passed in (train-safe fits)
VERSIONS_REQUIRING_SPLIT = {"a4_stroop_rt_residual"}
