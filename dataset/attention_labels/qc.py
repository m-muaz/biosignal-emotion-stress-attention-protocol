"""Quality-control reports for the SART/Stroop/Schulte canonical tables
(plan §54-57), computed straight from the canonical tables before any
label strategy is applied. `distribution_report` (generic, per-dataset-
version) is reused as-is from dataset.stress_labels.qc -- it only looks at
`participant_id`/`block_id`/target-column shape, nothing stress-specific.
"""

from __future__ import annotations

import pandas as pd

from dataset.stress_labels.qc import distribution_report  # noqa: F401 (re-exported for callers)


def sart_qc_report(trials_df: pd.DataFrame) -> pd.DataFrame:
    if trials_df.empty:
        return pd.DataFrame()
    rows = []
    for pid, g in trials_df.groupby("participant_id"):
        n = len(g)
        go = g[g["trial_type"] == "go"]
        no_go = g[g["trial_type"] == "no_go"]
        responded = g[g["responded"] == True]  # noqa: E712
        correct = g[g["correct"] == True]  # noqa: E712
        commission = int(g["commission_error"].sum())
        omission = int(g["omission_error"].sum())
        valid_rt = g[g["rt_sec"].notna()]["rt_sec"]
        rows.append(
            {
                "participant_id": pid,
                "total_trials": n,
                "go_trials": len(go),
                "no_go_trials": len(no_go),
                "responded_trials": len(responded),
                "correct_trials": len(correct),
                "total_errors": commission + omission,
                "commission_errors": commission,
                "commission_error_rate": commission / len(no_go) if len(no_go) else None,
                "omission_errors": omission,
                "omission_error_rate": omission / len(go) if len(go) else None,
                "median_rt": valid_rt.median() if not valid_rt.empty else None,
                "mean_rt": valid_rt.mean() if not valid_rt.empty else None,
                "rt_std": valid_rt.std() if not valid_rt.empty else None,
                "rt_cv": (valid_rt.std() / valid_rt.mean()) if (not valid_rt.empty and valid_rt.mean()) else None,
                "invalid_rt_count": int((g["responded"] == True).sum() - len(valid_rt)),  # noqa: E712
                "accuracy_mismatch_count": int(g["accuracy_mismatch"].sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("participant_id").reset_index(drop=True)


def stroop_qc_report(trials_df: pd.DataFrame) -> pd.DataFrame:
    if trials_df.empty:
        return pd.DataFrame()
    rows = []
    for (pid, rep), g in trials_df.groupby(["participant_id", "repetition_index"]):
        n = len(g)
        congruent = g[g["congruent"] == True]  # noqa: E712
        incongruent = g[g["congruent"] == False]  # noqa: E712
        correct = g[g["correct"] == True]  # noqa: E712
        incorrect = g[g["correct"] == False]  # noqa: E712
        rows.append(
            {
                "participant_id": pid,
                "repetition_index": rep,
                "total_trials": n,
                "congruent_trials": len(congruent),
                "incongruent_trials": len(incongruent),
                "correct_trials": len(correct),
                "incorrect_trials": len(incorrect),
                "accuracy": len(correct) / n if n else None,
                "mean_rt": g["rt_sec"].mean(),
                "median_rt": g["rt_sec"].median(),
                "median_rt_congruent": congruent["rt_sec"].median() if not congruent.empty else None,
                "median_rt_incongruent": incongruent["rt_sec"].median() if not incongruent.empty else None,
                "congruency_rt_effect": (
                    (incongruent["rt_sec"].median() - congruent["rt_sec"].median())
                    if (not congruent.empty and not incongruent.empty)
                    else None
                ),
                "missing_rt_count": int(g["rt_sec"].isna().sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["participant_id", "repetition_index"]).reset_index(drop=True)


def schulte_qc_report(blocks_df: pd.DataFrame, clicks_df: pd.DataFrame) -> pd.DataFrame:
    if blocks_df.empty:
        return pd.DataFrame()
    rows = []
    for _, b in blocks_df.iterrows():
        pid, rep = b["participant_id"], b["repetition_index"]
        block_clicks = (
            clicks_df[(clicks_df["participant_id"] == pid) & (clicks_df["repetition_index"] == rep)]
            if not clicks_df.empty
            else pd.DataFrame()
        )
        reconstructed_count = len(block_clicks)
        expected_count = b.get("expected_click_count")
        intervals = block_clicks["inter_click_interval_sec"] if not block_clicks.empty else pd.Series(dtype=float)
        confidences = block_clicks["reconstruction_confidence"] if not block_clicks.empty else pd.Series(dtype=object)
        rows.append(
            {
                "participant_id": pid,
                "repetition_index": rep,
                "completion_time_sec": b.get("completion_time_sec"),
                "misclicks": b.get("misclicks"),
                "reconstructed_click_count": reconstructed_count,
                "expected_click_count": expected_count,
                "click_count_valid": (reconstructed_count == expected_count) if pd.notna(expected_count) else None,
                "first_click_latency": block_clicks[block_clicks["click_index"] == 0]["first_click_latency_sec"].iloc[0]
                if not block_clicks.empty and (block_clicks["click_index"] == 0).any()
                else None,
                "median_inter_click_interval": intervals.median() if not intervals.empty else None,
                "mean_inter_click_interval": intervals.mean() if not intervals.empty else None,
                "click_reconstruction_valid": bool((confidences == "high").all()) if not confidences.empty else None,
                "reconstruction_confidence_high_frac": (confidences == "high").mean() if not confidences.empty else None,
            }
        )
    return pd.DataFrame(rows).sort_values(["participant_id", "repetition_index"]).reset_index(drop=True)
