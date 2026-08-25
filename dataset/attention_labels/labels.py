"""Label-strategy layer over the canonical SART/Stroop/Schulte tables
(plans/attention_task.md Phase 3, §27-41).

Mirrors dataset/stress_labels/labels.py: canonical columns get renamed/
derived into SEPARATE prefixed groups --

    cond_*      experimental condition (NOT an attention measure, Rule 2-4)
    beh_*       what the participant actually did
    derived_*   computed from behavioral history (rolling stats, residuals,
                prospective/next-trial labels)
    meta_*      provenance / resolution / window-mode bookkeeping

-- there is deliberately no single global `label` column (plan §50 Rule 16).
`versions.py`'s TargetSelector (reused from dataset.stress_labels.labels)
picks one column as `model_target` once a specific dataset version has
actually chosen its target; this module only builds the tables.

Physiological windowing (Phase 4) is intentionally NOT done here -- these
are table/label-only samples, one row per trial/block/click.
"""

from __future__ import annotations

import warnings

import pandas as pd

DEFAULT_HISTORY_N = 10
DEFAULT_MINIMUM_HISTORY = 3


# ── SART: rolling RT/error history + prospective (next-trial) labels ─────
def _sart_rolling_and_prospective(df: pd.DataFrame, history_n: int, minimum_history: int) -> pd.DataFrame:
    """Per participant, in trial_index order. Rolling stats use only valid
    GO-trial RTs from BEFORE the current trial (plan §11: `history = valid_
    go_trials_before_trial_i`, never including trial_i itself). Rolling
    error rate uses the previous `history_n` trials regardless of type.
    Prospective (`derived_next_trial_*`) labels are trial_i's own outcome
    shifted back one row, i.e. what trial_{i-1}'s pre-stimulus window would
    be predicting (plan §13, prediction_horizon_trials=1)."""
    out_frames = []
    for pid, g in df.groupby("participant_id", sort=False):
        g = g.sort_values("trial_index").reset_index(drop=True)
        is_error = (g["commission_error"] | g["omission_error"]).astype(float)

        rolling_mean, rolling_std, rolling_cv, rolling_valid_count = [], [], [], []
        rolling_commission_rate, rolling_omission_rate, rolling_total_error_rate = [], [], []

        valid_rt_history: list[float] = []
        commission_history: list[bool] = []
        omission_history: list[bool] = []

        for i in range(len(g)):
            recent_rt = pd.Series(valid_rt_history[-history_n:])
            if len(recent_rt) >= minimum_history:
                mean = recent_rt.mean()
                std = recent_rt.std(ddof=0)
                rolling_mean.append(mean)
                rolling_std.append(std)
                rolling_cv.append((std / mean) if mean else None)
            else:
                rolling_mean.append(None)
                rolling_std.append(None)
                rolling_cv.append(None)
            rolling_valid_count.append(len(recent_rt))

            recent_commission = commission_history[-history_n:]
            recent_omission = omission_history[-history_n:]
            n_hist = len(recent_commission)
            if n_hist >= minimum_history:
                rolling_commission_rate.append(sum(recent_commission) / n_hist)
                rolling_omission_rate.append(sum(recent_omission) / n_hist)
                rolling_total_error_rate.append((sum(recent_commission) + sum(recent_omission)) / n_hist)
            else:
                rolling_commission_rate.append(None)
                rolling_omission_rate.append(None)
                rolling_total_error_rate.append(None)

            row = g.iloc[i]
            if row["trial_type"] == "go" and bool(row["responded"]) and row["rt_sec"] is not None and not pd.isna(row["rt_sec"]):
                valid_rt_history.append(float(row["rt_sec"]))
            commission_history.append(bool(row["commission_error"]))
            omission_history.append(bool(row["omission_error"]))

        g["derived_rolling_rt_mean"] = rolling_mean
        g["derived_rolling_rt_std"] = rolling_std
        g["derived_rolling_rt_cv"] = rolling_cv
        g["derived_rolling_valid_rt_count"] = rolling_valid_count
        g["derived_rolling_commission_rate"] = rolling_commission_rate
        g["derived_rolling_omission_rate"] = rolling_omission_rate
        g["derived_rolling_total_error_rate"] = rolling_total_error_rate

        g["derived_next_trial_error"] = (g["commission_error"] | g["omission_error"]).shift(-1)
        g["derived_next_trial_commission_error"] = g["commission_error"].shift(-1)
        g["derived_next_trial_omission_error"] = g["omission_error"].shift(-1)
        g["derived_next_trial_rt"] = g["rt_sec"].shift(-1)
        g["derived_next_trial_rt_deviation"] = g["derived_next_trial_rt"] - g["derived_rolling_rt_mean"]
        g["meta_prediction_horizon_trials"] = 1

        out_frames.append(g)
    if not out_frames:
        return df
    # per-participant groups can have all-NA derived_* columns (e.g. a
    # participant whose last trial has no next-trial label) -- harmless
    # dtype-inference noise across groups, not a real data issue.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, message=".*empty or all-NA entries.*")
        return pd.concat(out_frames, ignore_index=True)


def build_sart_trial_samples(trials_df: pd.DataFrame, history_n: int = DEFAULT_HISTORY_N, minimum_history: int = DEFAULT_MINIMUM_HISTORY) -> pd.DataFrame:
    """One row per SART trial. Keeps trial_type/rt/correct/commission/
    omission as SEPARATE targets (plan §29 Version 1) rather than forcing
    one -- no TargetSelector applied here."""
    if trials_df.empty:
        return trials_df.copy()
    df = trials_df.copy()
    df = df.rename(
        columns={
            "trial_type": "cond_trial_type",
            "is_no_go": "cond_is_no_go",
            "digit": "cond_digit",
            "responded": "beh_responded",
            "rt_sec": "beh_rt_sec",
            "correct": "beh_correct",
            "commission_error": "beh_commission_error",
            "omission_error": "beh_omission_error",
            "accurate": "beh_accurate_logged",
            "accuracy_mismatch": "meta_accuracy_mismatch",
            "reconstructed_response_timestamp": "meta_reconstructed_response_timestamp",
            "protocol_version": "meta_protocol_version",
        }
    )
    # rolling/prospective helper reads plain "commission_error"/"omission_error"/"rt_sec"/"trial_type"/"responded"
    df = _sart_rolling_and_prospective(
        df.rename(
            columns={
                "cond_trial_type": "trial_type",
                "beh_responded": "responded",
                "beh_rt_sec": "rt_sec",
                "beh_commission_error": "commission_error",
                "beh_omission_error": "omission_error",
            }
        ),
        history_n,
        minimum_history,
    )
    df = df.rename(
        columns={
            "trial_type": "cond_trial_type",
            "responded": "beh_responded",
            "rt_sec": "beh_rt_sec",
            "commission_error": "beh_commission_error",
            "omission_error": "beh_omission_error",
        }
    )
    df["meta_label_resolution"] = "trial"
    df["meta_history_n"] = history_n
    df["meta_minimum_history"] = minimum_history
    return df


# ── Stroop ────────────────────────────────────────────────────────────
def build_stroop_trial_samples(trials_df: pd.DataFrame) -> pd.DataFrame:
    """One row per Stroop stimulus (plan §33 Version 3). `cond_congruency`
    is an experimental-condition label, NOT an attention measure (Rule 4).
    `derived_stroop_rt_residual` is intentionally NOT computed here -- it
    requires a train-participants-only fit (Rule 11/17), which needs the
    participant split; see versions.py's Version 4 builder."""
    if trials_df.empty:
        return trials_df.copy()
    df = trials_df.copy()
    df["cond_congruency"] = df["congruent"].map({True: "congruent", False: "incongruent"})
    df = df.rename(
        columns={
            "correct": "beh_correct",
            "clicked_color": "beh_clicked_color",
            "rt_ms": "beh_rt_ms",
            "rt_sec": "beh_rt_sec",
            "protocol_version": "meta_protocol_version",
        }
    )
    df["meta_label_resolution"] = "trial"
    return df


# ── Schulte ───────────────────────────────────────────────────────────
def build_schulte_block_samples(blocks_df: pd.DataFrame) -> pd.DataFrame:
    """One row per Schulte grid (plan §35 Version 5). Keeps
    completion_time_sec continuous -- NO cohort-median attention split
    (Rule 12)."""
    if blocks_df.empty:
        return blocks_df.copy()
    df = blocks_df.copy()
    df = df.rename(
        columns={
            "completion_time_sec": "beh_completion_time_sec",
            "misclicks": "beh_misclicks",
            "grid_size": "meta_grid_size",
            "expected_click_count": "meta_expected_click_count",
            "protocol_version": "meta_protocol_version",
        }
    )
    df["meta_label_resolution"] = "block"
    return df


def build_schulte_click_samples(clicks_df: pd.DataFrame) -> pd.DataFrame:
    """One row per reconstructed click (plan §36 Version 6). Only emitted
    if click reconstruction succeeded upstream (dataset.attention_labels.
    canonical_tables.build_schulte_clicks) -- rows with low timestamp/
    rt_ms cross-check agreement are kept but flagged via
    meta_reconstruction_confidence rather than silently discarded, so the
    version builder can decide whether to filter them (Rule 14)."""
    if clicks_df.empty:
        return clicks_df.copy()
    df = clicks_df.copy()
    df = df.rename(
        columns={
            "inter_click_interval_sec": "beh_inter_click_interval_sec",
            "first_click_latency_sec": "beh_first_click_latency_sec",
            "misclick": "beh_misclick",
            "reconstruction_confidence": "meta_reconstruction_confidence",
            "ambiguous_block_match": "meta_ambiguous_block_match",
        }
    )
    df["meta_label_resolution"] = "click"
    return df
