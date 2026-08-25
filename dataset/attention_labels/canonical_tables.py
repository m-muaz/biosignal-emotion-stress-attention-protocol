"""Canonical intermediate tables for the SART / Stroop / Schulte attention
tasks (plans/attention_task.md Phase 2, §5-26).

Mirrors dataset/stress_labels/canonical_tables.py's approach: reuse
dataset/windows.py's already-validated event-pairing helper (`_pair_events`)
rather than re-deriving trial/block boundary logic, and turn those pairings
into tidy, richly-columned per-participant tables. Raw fields only -- no
target/label selection here (see labels.py, once written, for that).

SART runs as a single continuous block (one repetition); Stroop and Schulte
each run as 3 repetitions, one subprocess per repetition bracketed by
task_start/task_end, with no top-level repetition-index column of their own
-- repetition_index is derived here the same way dataset/windows.py derives
`session_trial_index`: by counting task_start occurrences.
"""

from __future__ import annotations

import pandas as pd

from dataset.windows import _pair_events, _to_int_or_none

ATTENTION_PROTOCOL_VERSION = "attention_v1"  # no known scoring bug for sart/stroop/schulte; kept for provenance


def _payload(row) -> dict:
    return row["event_payload_json"]


def _count_events_in_span(events: pd.DataFrame, event_type: str, t_start: float, t_end: float) -> int:
    mask = (events["event_type"] == event_type) & (events["timestamp_host_utc"] >= t_start) & (events["timestamp_host_utc"] <= t_end)
    return int(mask.sum())


def _repetition_slices(sub: pd.DataFrame) -> list[tuple[int, float, float, pd.DataFrame]]:
    """[(repetition_index, t_start, t_end, events_in_that_repetition), ...],
    one subprocess run per task_start/task_end pair, in chronological order."""
    out = []
    for rep_idx, (rep_start, rep_end) in enumerate(_pair_events(sub, "task_start", "task_end", key_specs=None)):
        t0, t1 = float(rep_start["timestamp_host_utc"]), float(rep_end["timestamp_host_utc"])
        out.append((rep_idx, t0, t1, sub[(sub["timestamp_host_utc"] >= t0) & (sub["timestamp_host_utc"] <= t1)]))
    return out


# ── attention_blocks (one row per block, all 3 tasks) ────────────────────
def build_attention_blocks(participant_id: str, session_id: str, events: pd.DataFrame) -> pd.DataFrame:
    """One row per task block: SART's single continuous run, plus one row
    per Stroop/Schulte repetition's game block. Each row also carries its
    paired baseline window's start/end (plan §6) rather than the baseline
    being a separate row, since every game block has exactly one baseline
    touchpoint attached to it 1:1."""
    rows = []

    sart = events[events["task"] == "sart"]
    baseline_pairs = _pair_events(sart, "baseline_start", "baseline_end", key_specs=["payload:position"])
    baseline = baseline_pairs[0] if baseline_pairs else None
    for srow, erow in _pair_events(sart, "block_start", "block_end", key_specs=["block_index"]):
        rows.append(
            _block_row(
                participant_id,
                session_id,
                "sart",
                block_id=_to_int_or_none(srow.get("block_index")),
                repetition_index=0,
                srow=srow,
                erow=erow,
                baseline=baseline,
                extra={"practice": _payload(srow).get("practice")},
            )
        )

    for task_label in ("attention_focus_stroop", "attention_focus_schulte"):
        sub = events[events["task"] == task_label]
        for rep_idx, t0, t1, rep_slice in _repetition_slices(sub):
            baseline_df = rep_slice[rep_slice["condition_label"] == "baseline"]
            baseline_pairs = _pair_events(baseline_df, "block_start", "block_end", key_specs=None)
            baseline = baseline_pairs[0] if baseline_pairs else None

            game_df = rep_slice[rep_slice["condition_label"] != "baseline"]
            for srow, erow in _pair_events(game_df, "block_start", "block_end", key_specs=None):
                epayload = _payload(erow)
                extra = {"skipped": epayload.get("skipped")}
                if task_label == "attention_focus_schulte":
                    grid_size = epayload.get("grid_size")
                    extra.update(
                        completion_time_sec=epayload.get("completion_time_sec"),
                        misclicks=epayload.get("misclicks"),
                        grid_size=grid_size,
                        expected_click_count=(grid_size * grid_size) if grid_size else None,
                    )
                rows.append(
                    _block_row(
                        participant_id,
                        session_id,
                        task_label,
                        block_id=_to_int_or_none(srow.get("block_index")),
                        repetition_index=rep_idx,
                        srow=srow,
                        erow=erow,
                        baseline=baseline,
                        extra=extra,
                    )
                )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ("focus_loss_count", "frame_drop_warning_count"):
        event_type = "focus_lost" if col == "focus_loss_count" else "frame_drop_warning"
        df[col] = df.apply(lambda r: _count_events_in_span(events, event_type, r["block_start"], r["block_end"]), axis=1)
    return df


def _block_row(participant_id, session_id, task, block_id, repetition_index, srow, erow, baseline, extra: dict) -> dict:
    row = {
        "participant_id": participant_id,
        "session_id": session_id,
        "task": task,
        "block_id": block_id,
        "repetition_index": repetition_index,
        "block_start": float(srow["timestamp_host_utc"]),
        "block_end": float(erow["timestamp_host_utc"]),
        "duration_sec": float(erow["timestamp_host_utc"]) - float(srow["timestamp_host_utc"]),
        "baseline_start": float(baseline[0]["timestamp_host_utc"]) if baseline else None,
        "baseline_end": float(baseline[1]["timestamp_host_utc"]) if baseline else None,
        "protocol_version": ATTENTION_PROTOCOL_VERSION,
    }
    row.update(extra)
    return row


# ── sart_trials ────────────────────────────────────────────────────────
def build_sart_trials(participant_id: str, session_id: str, events: pd.DataFrame) -> pd.DataFrame:
    """One row per SART trial. Uses the logged `rt` field as the
    authoritative RT (plan §10/Rule 5-6) -- NOT the fixed ~1.16s
    trial-slot response-event timestamp. Reconstructs `correct` from
    trial_type + responded and cross-checks it against the logged
    `accurate` field, flagging (not silently resolving) any disagreement
    (plan §9)."""
    sart = events[events["task"] == "sart"]
    rows = []
    for srow, erow in _pair_events(sart, "trial_start", "response", key_specs=["block_index", "trial_index"]):
        sp, rp = _payload(srow), _payload(erow)
        is_no_go = bool(sp.get("is_omit"))
        responded = bool(rp.get("responded"))
        rt_sec = rp.get("rt")
        stimulus_timestamp = float(srow["timestamp_host_utc"])
        reconstructed_response_timestamp = (stimulus_timestamp + rt_sec) if (responded and rt_sec is not None) else None

        commission_error = is_no_go and responded
        omission_error = (not is_no_go) and (not responded)
        reconstructed_correct = ((not is_no_go) and responded) or (is_no_go and not responded)
        logged_accurate = rp.get("accurate")
        accuracy_mismatch = (logged_accurate is not None) and (bool(logged_accurate) != reconstructed_correct)

        rows.append(
            {
                "participant_id": participant_id,
                "session_id": session_id,
                "block_id": _to_int_or_none(srow.get("block_index")),
                "trial_id": _to_int_or_none(srow.get("trial_index")),
                "trial_index": _to_int_or_none(srow.get("trial_index")),
                "stimulus_timestamp": stimulus_timestamp,
                "digit": sp.get("number"),
                "font_size": sp.get("font_size"),
                "is_no_go": is_no_go,
                "trial_type": "no_go" if is_no_go else "go",
                "responded": responded,
                "rt_sec": rt_sec,
                "reconstructed_response_timestamp": reconstructed_response_timestamp,
                "accurate": logged_accurate,
                "correct": reconstructed_correct,
                "accuracy_mismatch": accuracy_mismatch,
                "commission_error": commission_error,
                "omission_error": omission_error,
                "practice": sp.get("practice"),
                "protocol_version": ATTENTION_PROTOCOL_VERSION,
            }
        )
    return pd.DataFrame(rows)


# ── stroop_trials ─────────────────────────────────────────────────────
def build_stroop_trials(participant_id: str, session_id: str, events: pd.DataFrame) -> pd.DataFrame:
    """One row per Stroop stimulus. Unlike SART, `rt_ms` corresponds to
    the real participant response (plan §3.2) -- no truncation/offset
    correction needed."""
    sub = events[events["task"] == "attention_focus_stroop"]
    rows = []
    for rep_idx, t0, t1, rep_slice in _repetition_slices(sub):
        game_df = rep_slice[rep_slice["condition_label"] != "baseline"]
        for srow, erow in _pair_events(game_df, "trial_start", "response", key_specs=["block_index", "trial_index"]):
            sp, rp = _payload(srow), _payload(erow)
            rt_ms = rp.get("rt_ms")
            rt_sec = (rt_ms / 1000.0) if rt_ms is not None else None
            stimulus_timestamp = float(srow["timestamp_host_utc"])
            rows.append(
                {
                    "participant_id": participant_id,
                    "session_id": session_id,
                    "block_id": _to_int_or_none(srow.get("block_index")),
                    "repetition_index": rep_idx,
                    "trial_id": _to_int_or_none(srow.get("trial_index")),
                    "trial_index": _to_int_or_none(srow.get("trial_index")),
                    "stimulus_timestamp": stimulus_timestamp,
                    "word": sp.get("word"),
                    "ink_color": sp.get("ink_color"),
                    "congruent": sp.get("congruent"),
                    "clicked_color": rp.get("clicked_color"),
                    "correct": rp.get("correct"),
                    "rt_ms": rt_ms,
                    "rt_sec": rt_sec,
                    "response_timestamp": float(erow["timestamp_host_utc"]),
                    "protocol_version": ATTENTION_PROTOCOL_VERSION,
                }
            )
    return pd.DataFrame(rows)


# ── schulte_blocks ────────────────────────────────────────────────────
def build_schulte_blocks(participant_id: str, session_id: str, events: pd.DataFrame) -> pd.DataFrame:
    """One row per Schulte grid (one per repetition). Keeps completion
    time and misclicks continuous -- no cohort-median attention split
    here (plan §21/Rule 12)."""
    sub = events[events["task"] == "attention_focus_schulte"]
    rows = []
    for rep_idx, t0, t1, rep_slice in _repetition_slices(sub):
        baseline_df = rep_slice[rep_slice["condition_label"] == "baseline"]
        baseline_pairs = _pair_events(baseline_df, "block_start", "block_end", key_specs=None)
        baseline = baseline_pairs[0] if baseline_pairs else None

        game_df = rep_slice[rep_slice["condition_label"] != "baseline"]
        for srow, erow in _pair_events(game_df, "block_start", "block_end", key_specs=None):
            ep = _payload(erow)
            grid_size = ep.get("grid_size")
            rows.append(
                {
                    "participant_id": participant_id,
                    "session_id": session_id,
                    "block_id": _to_int_or_none(srow.get("block_index")),
                    "repetition_index": rep_idx,
                    "block_start": float(srow["timestamp_host_utc"]),
                    "block_end": float(erow["timestamp_host_utc"]),
                    "duration_sec": float(erow["timestamp_host_utc"]) - float(srow["timestamp_host_utc"]),
                    "baseline_start": float(baseline[0]["timestamp_host_utc"]) if baseline else None,
                    "baseline_end": float(baseline[1]["timestamp_host_utc"]) if baseline else None,
                    "completion_time_sec": ep.get("completion_time_sec"),
                    "misclicks": ep.get("misclicks"),
                    "grid_size": grid_size,
                    "expected_click_count": (grid_size * grid_size) if grid_size else None,
                    "skipped": ep.get("skipped"),
                    "protocol_version": ATTENTION_PROTOCOL_VERSION,
                }
            )
    return pd.DataFrame(rows)


# ── schulte_clicks ────────────────────────────────────────────────────
def build_schulte_clicks(participant_id: str, session_id: str, events: pd.DataFrame, blocks: pd.DataFrame) -> pd.DataFrame:
    """One row per click, reconstructed by assigning each `cell_click`
    event (which carries no block_index/trial_index of its own, plan §22)
    to the block whose [block_start, block_end] span contains it.

    `cell_click.rt_ms` is the game's own cumulative "time since block
    start" clock, verified empirically against wall-clock click
    timestamps for this cohort (see plans/attention_task.md discussion) --
    so inter_click_interval_sec / first_click_latency_sec can be derived
    two independent ways: from wall-clock timestamp deltas (primary, used
    for the output columns, since it doesn't depend on trusting the game
    client's internal clock) and from diff(rt_ms) (cross-check, feeds
    `reconstruction_confidence`). Clicks outside every block span, or
    matching more than one block, are flagged rather than silently
    dropped/duplicated (plan §23)."""
    sub = events[(events["task"] == "attention_focus_schulte") & (events["event_type"] == "cell_click")].sort_values("timestamp_host_utc")
    game_blocks = blocks[blocks["task"] == "attention_focus_schulte"] if not blocks.empty and "task" in blocks.columns else blocks

    rows = []
    unassigned = 0
    ambiguous = 0
    for _, row in sub.iterrows():
        ts = float(row["timestamp_host_utc"])
        match = game_blocks[(game_blocks["block_start"] <= ts) & (ts <= game_blocks["block_end"])]
        if match.empty:
            unassigned += 1
            continue
        if len(match) > 1:
            ambiguous += 1
        brow = match.iloc[0]
        p = _payload(row)
        rows.append(
            {
                "participant_id": participant_id,
                "session_id": session_id,
                "block_id": brow["block_id"],
                "repetition_index": brow["repetition_index"],
                "block_start": brow["block_start"],
                "click_timestamp": ts,
                "clicked_number": p.get("number_clicked"),
                "expected_number": p.get("expected_number"),
                "correct": p.get("correct"),
                "misclick": (p.get("correct") is False),
                "rt_ms_reported": p.get("rt_ms"),
                "ambiguous_block_match": len(match) > 1,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values(["participant_id", "repetition_index", "click_timestamp"]).reset_index(drop=True)
    out_rows = []
    for (pid, rep), g in df.groupby(["participant_id", "repetition_index"], sort=False):
        g = g.sort_values("click_timestamp").reset_index(drop=True)
        prev_ts = None
        prev_rt_ms = None
        for click_index, r in g.iterrows():
            ts = r["click_timestamp"]
            inter_click_interval_sec = (ts - prev_ts) if prev_ts is not None else None
            first_click_latency_sec = (ts - r["block_start"]) if click_index == 0 else None

            rt_ms = r["rt_ms_reported"]
            rt_ms_diff_sec = None
            if rt_ms is not None:
                rt_ms_diff_sec = (rt_ms / 1000.0) if prev_rt_ms is None else (rt_ms - prev_rt_ms) / 1000.0
            reconstruction_confidence = None
            if inter_click_interval_sec is not None and rt_ms_diff_sec is not None:
                reconstruction_confidence = "high" if abs(inter_click_interval_sec - rt_ms_diff_sec) <= 0.05 else "low"
            elif first_click_latency_sec is not None and rt_ms is not None:
                reconstruction_confidence = "high" if abs(first_click_latency_sec - rt_ms / 1000.0) <= 0.05 else "low"

            out_rows.append(
                {
                    "participant_id": pid,
                    "session_id": session_id,
                    "block_id": r["block_id"],
                    "repetition_index": rep,
                    "click_index": int(click_index),
                    "clicked_number": r["clicked_number"],
                    "expected_number": r["expected_number"],
                    "click_timestamp": ts,
                    "previous_click_timestamp": prev_ts,
                    "inter_click_interval_sec": inter_click_interval_sec,
                    "first_click_latency_sec": first_click_latency_sec,
                    "misclick": r["misclick"],
                    "rt_ms_reported": rt_ms,
                    "reconstruction_confidence": reconstruction_confidence,
                    "ambiguous_block_match": r["ambiguous_block_match"],
                }
            )
            prev_ts = ts
            prev_rt_ms = rt_ms if rt_ms is not None else prev_rt_ms

    out = pd.DataFrame(out_rows)
    out.attrs["unassigned_click_count"] = unassigned
    out.attrs["ambiguous_click_count"] = ambiguous
    return out
