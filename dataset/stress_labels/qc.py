"""Quality-control and dataset-distribution reports (plan §49-51).

Two kinds of report:
  - per-task/tier QC tables (raindrop_qc_report / highway_qc_report),
    computed straight from the canonical tables, before any label strategy
    is applied.
  - generic per-dataset-version distribution report (distribution_report),
    computed on a Version's sample table after a target has been picked.
"""

from __future__ import annotations

import pandas as pd

QC_FLAG_COLUMNS = ["clip_frac_pass", "dc_drift_pass", "emg_hf_ratio_pass", "line_noise_pass", "snr_pass", "kurtosis_pass"]


def raindrop_qc_report(trials_df: pd.DataFrame) -> pd.DataFrame:
    if trials_df.empty:
        return pd.DataFrame()
    rows = []
    for (pid, tier), g in trials_df.groupby(["participant_id", "tier"]):
        correct = g["correct"].sum()
        missed = g["missed"].sum()
        n = len(g)
        answered = g[g["rt_sec"].notna()]
        rows.append(
            {
                "participant_id": pid,
                "tier": tier,
                "trial_count": n,
                "correct_count": int(correct),
                "accuracy": correct / n if n else None,
                "miss_count": int(missed),
                "miss_rate": missed / n if n else None,
                "mean_rt": answered["rt_sec"].mean() if not answered.empty else None,
                "median_rt": answered["rt_sec"].median() if not answered.empty else None,
                "mean_deadline_fraction": g["deadline_fraction"].mean(),
                "median_deadline_fraction": g["deadline_fraction"].median(),
            }
        )
    return pd.DataFrame(rows).sort_values(["participant_id", "tier"]).reset_index(drop=True)


def raindrop_wrong_submission_counts(unmatched_df: pd.DataFrame) -> pd.DataFrame:
    if unmatched_df.empty:
        return pd.DataFrame()
    return (
        unmatched_df.groupby(["participant_id", "tier"])
        .size()
        .reset_index(name="wrong_submission_count")
        .sort_values(["participant_id", "tier"])
        .reset_index(drop=True)
    )


def highway_qc_report(obstacles_df: pd.DataFrame, self_report_df: pd.DataFrame) -> pd.DataFrame:
    if obstacles_df.empty:
        return pd.DataFrame()
    rows = []
    for (pid, tier), g in obstacles_df.groupby(["participant_id", "tier"]):
        n = len(g)
        collisions = g["collision"].sum()
        near_misses = g["near_miss"].sum()
        avoided = g["avoided"].sum()
        row = {
            "participant_id": pid,
            "tier": tier,
            "obstacle_count": n,
            "collision_count": int(collisions),
            "collision_rate": collisions / n if n else None,
            "near_miss_count": int(near_misses),
            "near_miss_rate": near_misses / n if n else None,
            "avoidance_rate": avoided / n if n else None,
            "median_reaction_margin": g["reaction_margin_sec"].median(),
            "median_normalized_reaction_margin": g["normalized_reaction_margin"].median(),
            "protocol_version": g["protocol_version"].iloc[0],
            "known_highway_bug": bool(g["known_highway_bug"].iloc[0]),
            "same_wave_duplicate_suspected_count": int(g["same_wave_duplicate_suspected"].sum()),
        }
        if not self_report_df.empty:
            sr = self_report_df[(self_report_df["participant_id"] == pid) & (self_report_df["tier"] == tier)]
            for item in ("stress", "workload", "frustration", "arousal", "perceived_difficulty"):
                row[f"self_report_{item}"] = sr[item].iloc[0] if (not sr.empty and item in sr.columns) else None
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["participant_id", "tier"]).reset_index(drop=True)


def cross_tier_manipulation_check(qc_df: pd.DataFrame, participant_id: str, metric_cols: list[str]) -> pd.DataFrame:
    """Tier 1 -> 2 -> 3 for one participant, across the requested metric
    columns -- flags nothing automatically (plan §50: 'do not assume the
    effect must be monotonic'), just lays the numbers out for inspection."""
    sub = qc_df[qc_df["participant_id"] == participant_id].sort_values("tier")
    return sub[["tier"] + [c for c in metric_cols if c in sub.columns]].reset_index(drop=True)


def eeg_signal_qc_report(meta_df: pd.DataFrame) -> pd.DataFrame:
    """Rolls up dataset/stress_labels/export_npy.py's per-epoch EEG QC
    columns to participant (x tier, if present) level: mean fraction of
    checks passed, and % of epochs where ANY channel fails each individual
    check (the same worst-case-over-channels philosophy already used for
    `qc_n_checks_passed_min_over_channels`). Mirrors raindrop_qc_report/
    highway_qc_report's plain-DataFrame style (plan §49-50)."""
    if meta_df.empty or "qc_channels_json" not in meta_df.columns:
        return pd.DataFrame()

    import json

    group_cols = ["participant_id"] + (["cond_task_pressure_tier"] if "cond_task_pressure_tier" in meta_df.columns else [])
    rows = []
    # dropna=False: tier is NaN for baseline-block epochs -- pandas'
    # groupby default silently drops NaN-key groups, which would erase
    # every baseline epoch from this report.
    for key, g in meta_df.groupby(group_cols, dropna=False):
        key = (key,) if not isinstance(key, tuple) else key
        row = dict(zip(group_cols, key))
        row["epoch_count"] = len(g)
        row["mean_checks_passed_fraction"] = (g["qc_n_checks_passed_min_over_channels"] / g["qc_n_checks_total"]).mean()

        per_flag_fail_rate = {flag: 0 for flag in QC_FLAG_COLUMNS}
        for channels_json in g["qc_channels_json"]:
            channels = json.loads(channels_json)
            for flag in QC_FLAG_COLUMNS:
                if any(not ch.get(flag, True) for ch in channels):
                    per_flag_fail_rate[flag] += 1
        for flag, count in per_flag_fail_rate.items():
            row[f"pct_epochs_failing_{flag.removesuffix('_pass')}"] = count / len(g) if len(g) else None
        rows.append(row)

    return pd.DataFrame(rows).sort_values(group_cols).reset_index(drop=True)


def distribution_report(samples: pd.DataFrame, target_col: str = "model_target") -> dict:
    """Generic distribution report for one dataset-version sample table
    (plan §51). Handles both categorical and continuous targets."""
    if samples.empty:
        return {"participant_count": 0, "sample_count": 0}

    report = {
        "participant_count": samples["participant_id"].nunique(),
        "sample_count": len(samples),
        "samples_per_participant": samples.groupby("participant_id").size().to_dict(),
    }
    if "block_id" in samples.columns:
        report["block_count"] = samples[["participant_id", "block_id"]].drop_duplicates().shape[0]
    if "tier" in samples.columns:
        report["samples_per_tier"] = samples["tier"].value_counts(dropna=False).to_dict()

    if target_col in samples.columns:
        target = samples[target_col]
        report["missing_labels"] = int(target.isna().sum())
        non_null = target.dropna()
        if non_null.empty:
            pass
        elif pd.api.types.is_numeric_dtype(non_null) and non_null.nunique() > 10:
            report["target_kind"] = "continuous"
            report["target_stats"] = {
                "mean": float(non_null.mean()),
                "std": float(non_null.std()),
                "median": float(non_null.median()),
                "min": float(non_null.min()),
                "max": float(non_null.max()),
                "quantiles": {q: float(non_null.quantile(q)) for q in (0.25, 0.5, 0.75, 0.9)},
            }
        else:
            report["target_kind"] = "categorical"
            counts = non_null.value_counts()
            pct = (counts / counts.sum() * 100).round(2)
            report["class_counts"] = counts.to_dict()
            report["class_percentages"] = pct.to_dict()
            report["imbalance_ratio"] = float(counts.max() / counts.min()) if counts.min() > 0 else None

    if "subjective_labels_available" in samples.columns and bool(samples["subjective_labels_available"].any()):
        report["unique_questionnaire_observations"] = samples[["participant_id", "block_id"]].drop_duplicates().shape[0]
        report["windows_generated_from_those_observations"] = len(samples)

    return report
