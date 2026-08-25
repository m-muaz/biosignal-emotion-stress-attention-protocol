"""Canonical intermediate tables for the stress task (raindrop + attention_highway).

Reuses `dataset/windows.py`'s already-validated event-pairing helpers
(`_pair_events` et al.) rather than re-deriving trial/block boundary logic --
this module's job is to turn those pairings into the tidy, richly-columned
per-participant tables `plans_for_data_labeling/stress_task.md` §5-13 asks
for (raw payload fields windows.py's `Window` doesn't carry, task-pressure
config joined in from `session_manifest.json`, derived metrics, protocol/QC
flags), reusable across every later label-generation strategy. Nothing here
picks a `stress` label -- see `labels.py` for that.
"""

from __future__ import annotations

import pandas as pd

from dataset.stress_labels.protocol import (
    highway_known_bug,
    highway_protocol_version,
    highway_self_report_items,
    highway_tier_config,
    stress_tier_config,
)
from dataset.windows import _pair_events, _to_int_or_none

RAINDROP_PROTOCOL_VERSION = "raindrop_v1"  # no known scoring bug for raindrop; kept for provenance (plan §58)


def _payload(row) -> dict:
    return row["event_payload_json"]


def _count_events_in_span(events: pd.DataFrame, event_type: str, t_start: float, t_end: float) -> int:
    mask = (events["event_type"] == event_type) & (events["timestamp_host_utc"] >= t_start) & (events["timestamp_host_utc"] <= t_end)
    return int(mask.sum())


# ── stress_blocks (both raindrop and highway tier/baseline blocks) ───────
def build_stress_blocks(participant_id: str, session_id: str, events: pd.DataFrame, config_snapshot: dict) -> pd.DataFrame:
    """One row per block: raindrop baseline/tier blocks + highway
    baseline/tier blocks. `task` distinguishes the two. Task-pressure config
    (spawn_interval_sec/fall_duration_sec/concurrent_obstacles) is joined in
    from `config_snapshot` by tier id -- it is not present on the block
    events themselves."""
    rows = []

    # -- raindrop (task="stress") --
    st = events[events["task"] == "stress"]
    baseline_df = st[st["condition_label"] == "baseline"]
    for srow, erow in _pair_events(baseline_df, "block_start", "block_end", key_specs=["block_index"]):
        rows.append(
            _stress_block_row(
                participant_id, session_id, "stress", srow, erow, block_type="baseline", tier=None, tier_cfg=None
            )
        )
    tier_df = st[st["condition_label"].astype(str).str.startswith("tier_", na=False)]
    for srow, erow in _pair_events(tier_df, "block_start", "block_end", key_specs=["block_index"]):
        tier = _payload(srow).get("tier")
        tier_cfg = stress_tier_config(config_snapshot, tier) if tier is not None else None
        row = _stress_block_row(participant_id, session_id, "stress", srow, erow, block_type="active", tier=tier, tier_cfg=tier_cfg)
        row["accuracy"] = _payload(erow).get("accuracy")
        row["spawn_interval_sec"] = tier_cfg.get("spawn_interval_sec") if tier_cfg else None
        row["fall_duration_sec"] = tier_cfg.get("fall_duration_sec") if tier_cfg else None
        rows.append(row)

    # -- highway (task="attention_highway") --
    hw = events[events["task"] == "attention_highway"]
    protocol_version = highway_protocol_version(participant_id)
    known_bug = highway_known_bug(participant_id)

    baseline_df = hw[hw["condition_label"] == "baseline"]
    for srow, erow in _pair_events(baseline_df, "block_start", "block_end", key_specs=["block_index"]):
        row = _stress_block_row(
            participant_id, session_id, "attention_highway", srow, erow, block_type="baseline", tier=None, tier_cfg=None
        )
        row["protocol_version"] = protocol_version
        row["known_highway_bug"] = known_bug
        rows.append(row)

    tier_df = hw[hw["condition_label"].notna() & (hw["condition_label"] != "baseline")]
    for srow, erow in _pair_events(tier_df, "block_start", "block_end", key_specs=["block_index"]):
        tier = _payload(srow).get("tier")
        tier_cfg = highway_tier_config(config_snapshot, tier) if tier is not None else None
        row = _stress_block_row(participant_id, session_id, "attention_highway", srow, erow, block_type="active", tier=tier, tier_cfg=tier_cfg)
        row["nominal_difficulty"] = _payload(srow).get("nominal_difficulty")
        row["avoided"] = _payload(erow).get("avoided")
        row["collisions"] = _payload(erow).get("collisions")
        spawn_range = tier_cfg.get("spawn_interval_range_sec") if tier_cfg else None
        row["spawn_interval_min_sec"] = spawn_range[0] if spawn_range else None
        row["spawn_interval_max_sec"] = spawn_range[1] if spawn_range else None
        row["fall_duration_sec"] = tier_cfg.get("fall_duration_sec") if tier_cfg else None
        row["concurrent_obstacles"] = tier_cfg.get("concurrent_obstacles") if tier_cfg else None
        row["protocol_version"] = protocol_version
        row["known_highway_bug"] = known_bug
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ("focus_loss_count", "frame_drop_warning_count"):
        df[col] = df.apply(lambda r: _count_events_in_span(events, "focus_lost" if col == "focus_loss_count" else "frame_drop_warning", r["block_start"], r["block_end"]), axis=1)
    return df


def _stress_block_row(participant_id, session_id, task, srow, erow, block_type, tier, tier_cfg) -> dict:
    return {
        "participant_id": participant_id,
        "session_id": session_id,
        "task": task,
        "block_id": _to_int_or_none(srow.get("block_index")),
        "block_type": block_type,
        "condition_label": srow.get("condition_label"),
        "tier": tier,
        "nominal_difficulty": None,
        "block_start": float(srow["timestamp_host_utc"]),
        "block_end": float(erow["timestamp_host_utc"]),
        "duration_sec": float(erow["timestamp_host_utc"]) - float(srow["timestamp_host_utc"]),
        "spawn_interval_sec": None,
        "spawn_interval_min_sec": None,
        "spawn_interval_max_sec": None,
        "fall_duration_sec": None,
        "concurrent_obstacles": None,
        "protocol_version": RAINDROP_PROTOCOL_VERSION if task == "stress" else None,
        "known_highway_bug": None,
        "accuracy": None,
        "avoided": None,
        "collisions": None,
    }


# ── raindrop_trials ────────────────────────────────────────────────────
def build_raindrop_trials(participant_id: str, session_id: str, events: pd.DataFrame, config_snapshot: dict) -> pd.DataFrame:
    st = events[events["task"] == "stress"]
    rows = []
    for srow, erow in _pair_events(st, "trial_start", "response", key_specs=["block_index", "trial_index"]):
        payload = _payload(erow)
        tier = payload.get("tier")
        rt_sec = payload.get("rt")
        tier_cfg = stress_tier_config(config_snapshot, tier) if tier is not None else None
        fall_duration_sec = tier_cfg.get("fall_duration_sec") if tier_cfg else None
        deadline_fraction = (rt_sec / fall_duration_sec) if (rt_sec is not None and fall_duration_sec) else None
        rows.append(
            {
                "participant_id": participant_id,
                "session_id": session_id,
                "block_id": _to_int_or_none(srow.get("block_index")),
                "tier": tier,
                "trial_id": _to_int_or_none(srow.get("trial_index")),
                "trial_start": float(srow["timestamp_host_utc"]),
                "response_timestamp": float(erow["timestamp_host_utc"]),
                "expression": payload.get("expression"),
                "correct_answer": payload.get("correct_answer"),
                "participant_answer": payload.get("participant_answer"),
                "rt_sec": rt_sec,
                "correct": payload.get("correct"),
                "missed": payload.get("missed"),
                "spawn_interval_sec": tier_cfg.get("spawn_interval_sec") if tier_cfg else None,
                "fall_duration_sec": fall_duration_sec,
                "deadline_fraction": deadline_fraction,
                "protocol_version": RAINDROP_PROTOCOL_VERSION,
            }
        )
    return pd.DataFrame(rows)


def build_raindrop_unmatched_events(participant_id: str, session_id: str, events: pd.DataFrame) -> pd.DataFrame:
    """`wrong_submission` events carry no `trial_index` -- per the plan,
    these are NOT paired to a trial by proximity (that association hasn't
    been verified against the app code); preserved as their own table so
    later block/window-level aggregates (count, rate) can still use them."""
    st = events[events["task"] == "stress"]
    rows = []
    for _, row in st[st["event_type"] == "wrong_submission"].iterrows():
        payload = _payload(row)
        rows.append(
            {
                "participant_id": participant_id,
                "session_id": session_id,
                "block_id": _to_int_or_none(row.get("block_index")),
                "tier": payload.get("tier"),
                "timestamp": float(row["timestamp_host_utc"]),
                "participant_answer": payload.get("participant_answer"),
            }
        )
    return pd.DataFrame(rows)


# ── highway_obstacles ──────────────────────────────────────────────────
def build_highway_obstacles(participant_id: str, session_id: str, events: pd.DataFrame) -> pd.DataFrame:
    hw = events[events["task"] == "attention_highway"]
    protocol_version = highway_protocol_version(participant_id)
    known_bug = highway_known_bug(participant_id)

    rows = []
    for srow, erow in _pair_events(hw, "obstacle_spawn", "obstacle_resolved", key_specs=["block_index", "trial_index"]):
        sp, rp = _payload(srow), _payload(erow)
        outcome = rp.get("outcome")
        # obstacle_spawn/obstacle_resolved payloads carry no "tier" field
        # (only nominal_difficulty) -- the row's own condition_label
        # ("tier_1"/"tier_2"/"tier_3") is the source of truth here.
        condition_label = srow.get("condition_label")
        tier = int(condition_label.split("_")[1]) if isinstance(condition_label, str) and condition_label.startswith("tier_") else None
        rows.append(
            {
                "participant_id": participant_id,
                "session_id": session_id,
                "block_id": _to_int_or_none(srow.get("block_index")),
                "tier": tier,
                "obstacle_id": _to_int_or_none(srow.get("trial_index")),
                "spawn_timestamp": float(srow["timestamp_host_utc"]),
                "resolve_timestamp": float(erow["timestamp_host_utc"]),
                "lane": sp.get("lane"),
                "fall_duration_sec": sp.get("fall_duration_sec"),
                "spawn_wave_index": sp.get("spawn_wave_index"),
                "spawn_wave_size": sp.get("spawn_wave_size"),
                "camp_nudged": sp.get("camp_nudged"),
                "outcome": outcome,
                "collision": (outcome == "collision") if outcome is not None else None,
                "avoided": (outcome == "avoided") if outcome is not None else None,
                "near_miss": rp.get("near_miss"),
                "no_response": rp.get("no_response"),
                "response_attempted": rp.get("response_attempted"),
                "reaction_margin_sec": rp.get("reaction_margin_sec"),
                "protocol_version": protocol_version,
                "known_highway_bug": known_bug,
            }
        )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["normalized_reaction_margin"] = df.apply(
        lambda r: (r["reaction_margin_sec"] / r["fall_duration_sec"])
        if pd.notna(r["reaction_margin_sec"]) and pd.notna(r["fall_duration_sec"]) and r["fall_duration_sec"] not in (0, None)
        else None,
        axis=1,
    )
    # Issue 1 (notes.txt): same-wave duplicate scoring, pre-fix sessions only.
    # Flagged, not collapsed/dropped -- plan §15/§61 rule 13 forbid silently
    # altering pre-fix data.
    if known_bug and not df.empty:
        dup_mask = df.duplicated(subset=["block_id", "spawn_wave_index", "lane"], keep=False)
        df["same_wave_duplicate_suspected"] = dup_mask
    else:
        df["same_wave_duplicate_suspected"] = False
    return df


# ── highway_state (10Hz telemetry) ────────────────────────────────────
def build_highway_state(participant_id: str, session_id: str, events: pd.DataFrame, blocks: pd.DataFrame) -> pd.DataFrame:
    hw_blocks = blocks[(blocks["task"] == "attention_highway") & (blocks["block_type"] == "active")] if not blocks.empty else pd.DataFrame()

    def _block_id_for(ts: float):
        if hw_blocks.empty:
            return None
        match = hw_blocks[(hw_blocks["block_start"] <= ts) & (ts <= hw_blocks["block_end"])]
        return int(match.iloc[0]["block_id"]) if not match.empty and pd.notna(match.iloc[0]["block_id"]) else None

    rows = []
    for _, row in events[(events["task"] == "attention_highway") & (events["event_type"] == "state_tick")].iterrows():
        p = _payload(row)
        ts = float(row["timestamp_host_utc"])
        rows.append(
            {
                "participant_id": participant_id,
                "session_id": session_id,
                "block_id": _block_id_for(ts),
                "tier": p.get("tier"),
                "timestamp": ts,
                "player_lane": p.get("player_lane"),
                "active_obstacle_count": p.get("active_obstacle_count"),
                "blocked_lane_count": p.get("blocked_lane_count"),
                "nearest_time_to_collision_ms": p.get("nearest_time_to_collision_ms"),
                "own_lane_time_to_collision_ms": p.get("own_lane_time_to_collision_ms"),
                "rolling_collision_rate": p.get("rolling_collision_rate"),
                "rolling_avg_reaction_time_ms": p.get("rolling_avg_reaction_time_ms"),
                "rolling_window_sec": p.get("rolling_window_sec"),
                "time_since_last_collision_sec": p.get("time_since_last_collision_sec"),
                "time_since_last_near_miss_sec": p.get("time_since_last_near_miss_sec"),
            }
        )
    return pd.DataFrame(rows)


# ── highway_self_report ────────────────────────────────────────────────
def build_highway_self_report(participant_id: str, session_id: str, events: pd.DataFrame, config_snapshot: dict) -> pd.DataFrame:
    """Ratings columns are read dynamically from the session's own
    `config_snapshot` self-report item list -- the actual set collected
    varies by session (this cohort never configured frustration/arousal/
    valence, despite the plan listing 8 possible dimensions)."""
    items = highway_self_report_items(config_snapshot)
    hw = events[events["task"] == "attention_highway"]
    rows = []
    for srow, erow in _pair_events(hw, "self_report_prompt_onset", "self_report_response", key_specs=["block_index"]):
        payload = _payload(erow)
        row = {
            "participant_id": participant_id,
            "session_id": session_id,
            "block_id": _to_int_or_none(srow.get("block_index")),
            "tier": payload.get("tier"),
            "prompt_timestamp": float(srow["timestamp_host_utc"]),
            "response_timestamp": float(erow["timestamp_host_utc"]),
            "skipped": payload.get("skipped"),
        }
        for item in items:
            row[item] = payload.get(item)
            row[f"{item}_rt"] = payload.get(f"{item}_rt")
        rows.append(row)
    return pd.DataFrame(rows)
