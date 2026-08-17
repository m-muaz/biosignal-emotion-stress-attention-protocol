"""Event-defined window extraction: turns events.jsonl rows into labeled
time windows `(t_start, t_end, task, window_type, ...)` that
dataset/torch_dataset.py slices synced biosignal streams against.

Each task/sub-task gets its own extractor below, built by reading that
task's real events.jsonl schema first (event_type vocabulary, top-level
columns vs. event_payload_json fields, which event pairs bracket a
meaningful span). See docs/Dataset_Sync_Design.md §4 for the full per-task
writeup and the real-data numbers each extractor was verified against.

Window kinds (`Window.window_type`) -- baselines are their own window, not
folded into the following trial, and every gap not covered by any other
window (inter-task breaks, the pre-SART gap, etc.) is auto-detected too:
    "trial"         one scored/logged unit of task performance
    "block"         a whole tier/game-repetition span (may CONTAIN "trial" windows)
    "baseline"      a physiological-baseline touchpoint
    "self_report"   a post-block subjective rating
    "questionnaire" one preparation-phase questionnaire item response
    "transition"    an auto-detected gap not covered by any of the above
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


def load_events_parquet(path: Path) -> pd.DataFrame:
    """events.parquet stores event_payload_json as a JSON string (pyarrow
    can't infer one Arrow type for a dict column whose shape varies by
    event_type) -- this re-parses it back into a dict per row."""
    df = pd.read_parquet(path)
    df["event_payload_json"] = df["event_payload_json"].apply(json.loads)
    return df


@dataclass
class Window:
    task: str
    window_type: str
    t_start: float
    t_end: float
    block_index: int | None = None
    trial_index: int | None = None
    condition_label: str | None = None
    stimulus: dict = field(default_factory=dict)
    labels: dict = field(default_factory=dict)
    meta: dict = field(default_factory=dict)


# ── generic helpers ──────────────────────────────────────────────────────
def _key_value(row, spec: str):
    """`spec` is a column name, or "payload:<field>" for event_payload_json.
    NaN/None normalized to None so keys compare consistently either way."""
    if spec.startswith("payload:"):
        val = row["event_payload_json"].get(spec.split(":", 1)[1])
    else:
        val = row.get(spec)
    return None if (val is None or (isinstance(val, float) and pd.isna(val))) else val


def _to_int_or_none(x):
    return None if (x is None or (isinstance(x, float) and pd.isna(x))) else int(x)


def _pair_events(df: pd.DataFrame, start_type: str, end_type: str, key_specs: list[str] | None = None):
    """Pairs each `start_type` row with the nearest following `end_type` row.

    key_specs=None: sequential FIFO -- only safe once `df` is filtered so
    starts/ends can't overlap (e.g. one condition_label at a time).
    key_specs=[...]: pairs only rows with matching key tuples (e.g.
    ["block_index", "trial_index"]) -- needed when multiple instances can
    be in flight at once (e.g. a slow response overlapping the next trial).

    Returns list[(start_row, end_row)]; unmatched starts are dropped.
    """
    starts = df[df["event_type"] == start_type].sort_values("timestamp_host_utc")
    ends = df[df["event_type"] == end_type].sort_values("timestamp_host_utc")

    if key_specs is None:
        end_rows = list(ends.iterrows())
        used = [False] * len(end_rows)
        pairs = []
        for _, srow in starts.iterrows():
            t0 = srow["timestamp_host_utc"]
            match = None
            for i, (_, erow) in enumerate(end_rows):
                if not used[i] and erow["timestamp_host_utc"] >= t0:
                    match = i
                    break
            if match is not None:
                used[match] = True
                pairs.append((srow, end_rows[match][1]))
        return pairs

    end_by_key: dict = {}
    for _, erow in ends.iterrows():
        end_by_key.setdefault(tuple(_key_value(erow, s) for s in key_specs), []).append(erow)

    pairs = []
    for _, srow in starts.iterrows():
        key = tuple(_key_value(srow, s) for s in key_specs)
        candidates = [e for e in end_by_key.get(key, []) if e["timestamp_host_utc"] >= srow["timestamp_host_utc"]]
        if not candidates:
            continue
        pairs.append((srow, min(candidates, key=lambda e: e["timestamp_host_utc"])))
    return pairs


def _find_matching(df: pd.DataFrame, event_type: str, key_specs: list[str], key_row, after_ts=None, before_ts=None):
    """First row of `event_type` whose key tuple matches `key_row`'s,
    optionally time-constrained. None if no match."""
    cand = df[df["event_type"] == event_type]
    if after_ts is not None:
        cand = cand[cand["timestamp_host_utc"] >= after_ts]
    if before_ts is not None:
        cand = cand[cand["timestamp_host_utc"] <= before_ts]
    key = tuple(_key_value(key_row, s) for s in key_specs)
    for _, row in cand.iterrows():
        if tuple(_key_value(row, s) for s in key_specs) == key:
            return row
    return None


# ── emotion ──────────────────────────────────────────────────────────────
def extract_emotion_windows(events: pd.DataFrame) -> list[Window]:
    """"trial": [clip_playback_start, clip_playback_end], paired by payload
    clip_id (clip_playback_start/end themselves have NaN block/trial/
    condition_label). Metadata from two other same-clip_id rows:
        trial_start     -> block_index, trial_index, condition_label (valence_group), fine_grained_label
        rating_response -> labels (valence/arousal/liking/emotion_pick + RTs)

    "baseline": [baseline_start, baseline_end], one per (block_index,
    position) -- 3 baseline touchpoints/block (start/mid/end), so position
    disambiguates since block_index alone repeats 3x per block.
    """
    emo = events[events["task"] == "emotion"] if "task" in events.columns else events
    windows = []

    starts = emo[emo["event_type"] == "clip_playback_start"]
    ends = emo[emo["event_type"] == "clip_playback_end"]
    trial_starts = emo[emo["event_type"] == "trial_start"]
    ratings = emo[emo["event_type"] == "rating_response"]

    for _, srow in starts.iterrows():
        clip_id = srow["event_payload_json"].get("clip_id")
        t0 = float(srow["timestamp_host_utc"])

        end_candidates = ends[(ends["event_payload_json"].apply(lambda p: p.get("clip_id")) == clip_id) & (ends["timestamp_host_utc"] >= t0)]
        if end_candidates.empty:
            continue
        erow = end_candidates.iloc[0]

        trial_candidates = trial_starts[
            (trial_starts["event_payload_json"].apply(lambda p: p.get("clip_id")) == clip_id) & (trial_starts["timestamp_host_utc"] <= t0)
        ]
        trow = trial_candidates.iloc[-1] if not trial_candidates.empty else None

        rating_candidates = ratings[
            (ratings["event_payload_json"].apply(lambda p: p.get("clip_id")) == clip_id)
            & (ratings["timestamp_host_utc"] >= float(erow["timestamp_host_utc"]))
        ]
        rrow = rating_candidates.iloc[0] if not rating_candidates.empty else None

        stimulus = {
            "clip_id": clip_id,
            "file_path": srow["event_payload_json"].get("file_path"),
            "valence_group": (trow["condition_label"] if trow is not None else None),
            "fine_grained_label": (trow["event_payload_json"].get("fine_grained_label") if trow is not None else None),
        }
        windows.append(
            Window(
                task="emotion",
                window_type="trial",
                t_start=t0,
                t_end=float(erow["timestamp_host_utc"]),
                block_index=_to_int_or_none(trow["block_index"]) if trow is not None else None,
                trial_index=_to_int_or_none(trow["trial_index"]) if trow is not None else None,
                condition_label=stimulus["valence_group"],
                stimulus=stimulus,
                labels=dict(rrow["event_payload_json"]) if rrow is not None else {},
            )
        )

    for srow, erow in _pair_events(emo, "baseline_start", "baseline_end", key_specs=["block_index", "payload:position"]):
        windows.append(
            Window(
                task="emotion",
                window_type="baseline",
                t_start=float(srow["timestamp_host_utc"]),
                t_end=float(erow["timestamp_host_utc"]),
                block_index=_to_int_or_none(srow["block_index"]),
                meta={"position": srow["event_payload_json"].get("position")},
            )
        )

    return windows


# ── stress (raindrop) ────────────────────────────────────────────────────
def extract_stress_windows(events: pd.DataFrame) -> list[Window]:
    """task="stress" (raindrop only -- highway is a separate task label,
    see extract_attention_highway_windows).

    "baseline"/"block" (tier): block_start/block_end share ONE event_type
    across baseline/tier/inter_trial_pause condition_labels, so each
    condition is filtered first, then paired by block_index -- except
    inter_trial_pause, which carries no block_index, paired sequentially.

    "trial": [trial_start, response] (one arithmetic problem), keyed by
    (block_index, trial_index) since a slow response can overlap the next
    problem's trial_start.

    Known simplification: `wrong_submission` events (not tied to a
    trial_index) aren't turned into their own windows.
    """
    st = events[events["task"] == "stress"] if "task" in events.columns else events
    windows = []

    baseline_df = st[st["condition_label"] == "baseline"]
    for srow, erow in _pair_events(baseline_df, "block_start", "block_end", key_specs=["block_index"]):
        windows.append(
            Window(
                task="stress",
                window_type="baseline",
                t_start=float(srow["timestamp_host_utc"]),
                t_end=float(erow["timestamp_host_utc"]),
                block_index=_to_int_or_none(srow["block_index"]),
                condition_label="baseline",
            )
        )

    pause_df = st[st["condition_label"] == "inter_trial_pause"]
    for srow, erow in _pair_events(pause_df, "block_start", "block_end", key_specs=None):
        windows.append(
            Window(
                task="stress",
                window_type="transition",
                t_start=float(srow["timestamp_host_utc"]),
                t_end=float(erow["timestamp_host_utc"]),
                condition_label="inter_trial_pause",
            )
        )

    tier_df = st[st["condition_label"].astype(str).str.startswith("tier_", na=False)]
    for srow, erow in _pair_events(tier_df, "block_start", "block_end", key_specs=["block_index"]):
        windows.append(
            Window(
                task="stress",
                window_type="block",
                t_start=float(srow["timestamp_host_utc"]),
                t_end=float(erow["timestamp_host_utc"]),
                block_index=_to_int_or_none(srow["block_index"]),
                condition_label=srow["condition_label"],
                meta={"tier": srow["event_payload_json"].get("tier"), "accuracy": erow["event_payload_json"].get("accuracy")},
            )
        )

    for srow, erow in _pair_events(st, "trial_start", "response", key_specs=["block_index", "trial_index"]):
        payload = erow["event_payload_json"]
        windows.append(
            Window(
                task="stress",
                window_type="trial",
                t_start=float(srow["timestamp_host_utc"]),
                t_end=float(erow["timestamp_host_utc"]),
                block_index=_to_int_or_none(srow["block_index"]),
                trial_index=_to_int_or_none(srow["trial_index"]),
                condition_label=srow["condition_label"],
                stimulus={"expression": payload.get("expression"), "correct_answer": payload.get("correct_answer"), "tier": payload.get("tier")},
                labels={"participant_answer": payload.get("participant_answer"), "rt": payload.get("rt"), "correct": payload.get("correct"), "missed": payload.get("missed")},
            )
        )

    return windows


# ── attention_highway (2nd half of "stress", but its own task label) ────
def extract_attention_highway_windows(events: pd.DataFrame) -> list[Window]:
    """task="attention_highway" -- NOT logged under task="stress" despite
    being "the second half of the stress task"; filter on this exact label.

    "baseline": single occurrence (no per-tier baseline, unlike raindrop).
    "block" (tier): block_start/block_end per tier label.
    "trial": [obstacle_spawn, obstacle_resolved], keyed by (block_index, trial_index).
    "self_report": [self_report_prompt_onset, self_report_response], one per tier.

    Known simplification: continuous telemetry (`lane_change`, `state_tick`
    @10Hz, `frame_drop_warning`, `focus_lost`/`focus_regained`) has no
    natural discrete boundary and isn't turned into windows.
    """
    hw = events[events["task"] == "attention_highway"] if "task" in events.columns else events
    windows = []

    baseline_df = hw[hw["condition_label"] == "baseline"]
    for srow, erow in _pair_events(baseline_df, "block_start", "block_end", key_specs=["block_index"]):
        windows.append(
            Window(
                task="attention_highway",
                window_type="baseline",
                t_start=float(srow["timestamp_host_utc"]),
                t_end=float(erow["timestamp_host_utc"]),
                block_index=_to_int_or_none(srow["block_index"]),
                condition_label="baseline",
            )
        )

    tier_df = hw[hw["condition_label"].notna() & (hw["condition_label"] != "baseline")]
    for srow, erow in _pair_events(tier_df, "block_start", "block_end", key_specs=["block_index"]):
        windows.append(
            Window(
                task="attention_highway",
                window_type="block",
                t_start=float(srow["timestamp_host_utc"]),
                t_end=float(erow["timestamp_host_utc"]),
                block_index=_to_int_or_none(srow["block_index"]),
                condition_label=srow["condition_label"],
                meta={
                    "tier": srow["event_payload_json"].get("tier"),
                    "nominal_difficulty": srow["event_payload_json"].get("nominal_difficulty"),
                    "avoided": erow["event_payload_json"].get("avoided"),
                    "collisions": erow["event_payload_json"].get("collisions"),
                },
            )
        )

    for srow, erow in _pair_events(hw, "obstacle_spawn", "obstacle_resolved", key_specs=["block_index", "trial_index"]):
        spawn_payload, resolve_payload = srow["event_payload_json"], erow["event_payload_json"]
        windows.append(
            Window(
                task="attention_highway",
                window_type="trial",
                t_start=float(srow["timestamp_host_utc"]),
                t_end=float(erow["timestamp_host_utc"]),
                block_index=_to_int_or_none(srow["block_index"]),
                trial_index=_to_int_or_none(srow["trial_index"]),
                condition_label=srow["condition_label"],
                stimulus={"lane": spawn_payload.get("lane"), "fall_duration_sec": spawn_payload.get("fall_duration_sec")},
                labels={
                    "outcome": resolve_payload.get("outcome"),
                    "near_miss": resolve_payload.get("near_miss"),
                    "reaction_margin_sec": resolve_payload.get("reaction_margin_sec"),
                    "response_attempted": resolve_payload.get("response_attempted"),
                },
            )
        )

    for srow, erow in _pair_events(hw, "self_report_prompt_onset", "self_report_response", key_specs=["block_index"]):
        windows.append(
            Window(
                task="attention_highway",
                window_type="self_report",
                t_start=float(srow["timestamp_host_utc"]),
                t_end=float(erow["timestamp_host_utc"]),
                block_index=_to_int_or_none(srow["block_index"]),
                condition_label=srow["condition_label"],
                labels={k: v for k, v in erow["event_payload_json"].items() if k not in ("scale_min", "scale_max", "skipped")},
            )
        )

    return windows


# ── SART (own separate session) ─────────────────────────────────────────
def extract_sart_windows(events: pd.DataFrame) -> list[Window]:
    """task="sart". block_index/trial_index are populated natively here
    (Python-side event_logger.log(), not JS) -- 0=practice block, 1=real
    (both real sessions checked had practice disabled, so only block_index=1
    appears; the 0 path is handled identically).

    "baseline": single pre-task baseline (position="pre_task").
    "block": block_start/block_end, keyed by block_index.
    "trial": [trial_start, response], keyed by (block_index, trial_index).
    """
    sart = events[events["task"] == "sart"] if "task" in events.columns else events
    windows = []

    for srow, erow in _pair_events(sart, "baseline_start", "baseline_end", key_specs=["payload:position"]):
        windows.append(
            Window(
                task="sart",
                window_type="baseline",
                t_start=float(srow["timestamp_host_utc"]),
                t_end=float(erow["timestamp_host_utc"]),
                meta={"position": srow["event_payload_json"].get("position")},
            )
        )

    for srow, erow in _pair_events(sart, "block_start", "block_end", key_specs=["block_index"]):
        windows.append(
            Window(
                task="sart",
                window_type="block",
                t_start=float(srow["timestamp_host_utc"]),
                t_end=float(erow["timestamp_host_utc"]),
                block_index=_to_int_or_none(srow["block_index"]),
                meta={"practice": srow["event_payload_json"].get("practice")},
            )
        )

    for srow, erow in _pair_events(sart, "trial_start", "response", key_specs=["block_index", "trial_index"]):
        spayload, rpayload = srow["event_payload_json"], erow["event_payload_json"]
        windows.append(
            Window(
                task="sart",
                window_type="trial",
                t_start=float(srow["timestamp_host_utc"]),
                t_end=float(erow["timestamp_host_utc"]),
                block_index=_to_int_or_none(srow["block_index"]),
                trial_index=_to_int_or_none(srow["trial_index"]),
                stimulus={"number": spayload.get("number"), "font_size": spayload.get("font_size"), "is_omit": spayload.get("is_omit")},
                labels={"responded": rpayload.get("responded"), "rt": rpayload.get("rt"), "accurate": rpayload.get("accurate")},
                meta={"practice": spayload.get("practice")},
            )
        )

    return windows


# ── attention_focus (Schulte + Stroop, subprocess-per-trial) ────────────
def _extract_attention_focus_subgame(events: pd.DataFrame, task_label: str, has_stimulus_trials: bool) -> list[Window]:
    """Both sub-games run as one subprocess per repetition (task_start...
    task_end brackets one baseline + one game round), so there's no
    top-level "which repetition (0/1/2)" column -- session_trial_index is
    derived by counting task_start occurrences. Within one repetition,
    block_index (0=baseline, 1=game) is real but not unique ACROSS
    repetitions -- that's what session_trial_index is for.
    """
    sub = events[events["task"] == task_label] if "task" in events.columns else events
    windows = []

    repetitions = _pair_events(sub, "task_start", "task_end", key_specs=None)
    for session_trial_index, (rep_start, rep_end) in enumerate(repetitions):
        t0, t1 = float(rep_start["timestamp_host_utc"]), float(rep_end["timestamp_host_utc"])
        rep_slice = sub[(sub["timestamp_host_utc"] >= t0) & (sub["timestamp_host_utc"] <= t1)]

        baseline_df = rep_slice[rep_slice["condition_label"] == "baseline"]
        for srow, erow in _pair_events(baseline_df, "block_start", "block_end", key_specs=None):
            windows.append(
                Window(
                    task=task_label,
                    window_type="baseline",
                    t_start=float(srow["timestamp_host_utc"]),
                    t_end=float(erow["timestamp_host_utc"]),
                    block_index=_to_int_or_none(srow["block_index"]),
                    condition_label="baseline",
                    meta={"session_trial_index": session_trial_index},
                )
            )

        game_df = rep_slice[rep_slice["condition_label"] != "baseline"]
        game_pairs = _pair_events(game_df, "block_start", "block_end", key_specs=None)
        for srow, erow in game_pairs:
            windows.append(
                Window(
                    task=task_label,
                    window_type="block",
                    t_start=float(srow["timestamp_host_utc"]),
                    t_end=float(erow["timestamp_host_utc"]),
                    block_index=_to_int_or_none(srow["block_index"]),
                    condition_label=srow["condition_label"],
                    labels=dict(erow["event_payload_json"]),
                    meta={"session_trial_index": session_trial_index},
                )
            )

        if has_stimulus_trials:
            for srow, erow in _pair_events(rep_slice, "trial_start", "response", key_specs=["block_index", "trial_index"]):
                spayload, rpayload = srow["event_payload_json"], erow["event_payload_json"]
                windows.append(
                    Window(
                        task=task_label,
                        window_type="trial",
                        t_start=float(srow["timestamp_host_utc"]),
                        t_end=float(erow["timestamp_host_utc"]),
                        block_index=_to_int_or_none(srow["block_index"]),
                        trial_index=_to_int_or_none(srow["trial_index"]),
                        condition_label=srow["condition_label"],
                        stimulus={"word": spayload.get("word"), "ink_color": spayload.get("ink_color"), "congruent": spayload.get("congruent")},
                        labels={"clicked_color": rpayload.get("clicked_color"), "correct": rpayload.get("correct"), "rt_ms": rpayload.get("rt_ms")},
                        meta={"session_trial_index": session_trial_index},
                    )
                )

    return windows


def extract_attention_focus_windows(events: pd.DataFrame) -> list[Window]:
    """Schulte's `cell_click` carries no block/trial_index (25 clicks/trial,
    available from raw events if needed) so only baseline+block windows are
    extracted for it; Stroop additionally gets per-stimulus "trial" windows
    (20/block, real trial_index 0-19)."""
    return _extract_attention_focus_subgame(events, "attention_focus_schulte", has_stimulus_trials=False) + _extract_attention_focus_subgame(
        events, "attention_focus_stroop", has_stimulus_trials=True
    )


# ── preparation (questionnaire) ─────────────────────────────────────────
def extract_preparation_windows(events: pd.DataFrame) -> list[Window]:
    """No question-onset event exists -- only the response, logged with a
    self-timed `rt`. Window t_end=response timestamp, t_start=t_end-rt."""
    windows = []
    for _, row in events[events["event_type"] == "questionnaire_response"].iterrows():
        payload = row["event_payload_json"]
        rt = payload.get("rt")
        if rt is None or rt <= 0:
            continue
        t1 = float(row["timestamp_host_utc"])
        windows.append(
            Window(
                task="preparation",
                window_type="questionnaire",
                t_start=t1 - float(rt),
                t_end=t1,
                labels={"item_id": payload.get("item_id"), "value": payload.get("value")},
            )
        )
    return windows


# ── gaps / transitions ───────────────────────────────────────────────────
def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []
    intervals = sorted(intervals)
    merged = [list(intervals[0])]
    for s, e in intervals[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def extract_gap_windows(events: pd.DataFrame, windows: list[Window], span: tuple[float, float], min_gap_s: float = 1.0) -> list[Window]:
    """Complement of every window in `windows` within `span`, emitted as
    task="transition" windows -- covers inter-task breaks (which have no
    dedicated event_type; `runBreak()` is pure setTimeout, never calls
    logEvent) and any other unlabeled time. `min_gap_s` filters sub-second
    jitter. Each gap's `meta` records the nearest bracketing events."""
    covered = _merge_intervals([(w.t_start, w.t_end) for w in windows if w.t_start is not None and w.t_end is not None])
    span_start, span_end = span

    raw_gaps = []
    cursor = span_start
    for s, e in covered:
        if s - cursor >= min_gap_s:
            raw_gaps.append((cursor, s))
        cursor = max(cursor, e)
    if span_end - cursor >= min_gap_s:
        raw_gaps.append((cursor, span_end))

    sorted_events = events.sort_values("timestamp_host_utc")
    out = []
    for gs, ge in raw_gaps:
        before = sorted_events[sorted_events["timestamp_host_utc"] <= gs]
        after = sorted_events[sorted_events["timestamp_host_utc"] >= ge]
        prev_row = before.iloc[-1] if not before.empty else None
        next_row = after.iloc[0] if not after.empty else None
        out.append(
            Window(
                task="transition",
                window_type="transition",
                t_start=gs,
                t_end=ge,
                meta={
                    "preceding_task": prev_row["task"] if prev_row is not None else None,
                    "preceding_event_type": prev_row["event_type"] if prev_row is not None else None,
                    "following_task": next_row["task"] if next_row is not None else None,
                    "following_event_type": next_row["event_type"] if next_row is not None else None,
                },
            )
        )
    return out


# ── top-level orchestration ──────────────────────────────────────────────
WINDOW_EXTRACTORS = {
    "emotion": extract_emotion_windows,
    "stress": extract_stress_windows,
    "attention_highway": extract_attention_highway_windows,
    "attention_focus_schulte": lambda ev: _extract_attention_focus_subgame(ev, "attention_focus_schulte", has_stimulus_trials=False),
    "attention_focus_stroop": lambda ev: _extract_attention_focus_subgame(ev, "attention_focus_stroop", has_stimulus_trials=True),
    "sart": extract_sart_windows,
    "preparation": extract_preparation_windows,
}

# Each task has both a fine-grained "trial" window (one arithmetic problem/
# obstacle/stimulus/number) and a coarser "block" window (one uninterrupted
# run -- one tier, one Schulte/Stroop repetition, SART's single run). Since
# "trial" is often too fine-grained to be a useful atomic unit on its own,
# this is the single source of truth for "what counts as one trial" per
# task: "block" everywhere except emotion, where "trial" (one clip) already
# is the right unit -- there's no finer sub-event to fall back to.
RECOMMENDED_TRIAL_WINDOW_TYPE = {
    "emotion": "trial",
    "stress": "block",
    "attention_highway": "block",
    "attention_focus_schulte": "block",
    "attention_focus_stroop": "block",
    "sart": "block",
}


def extract_windows(events: pd.DataFrame, task: str) -> list[Window]:
    if task not in WINDOW_EXTRACTORS:
        raise ValueError(f"no window extractor registered for task {task!r}; available: {list(WINDOW_EXTRACTORS)}")
    return WINDOW_EXTRACTORS[task](events)


def extract_all_windows(events: pd.DataFrame, min_gap_s: float = 1.0) -> list[Window]:
    """Runs every registered task extractor, then auto-detects gap windows
    within each recording span present (`events`'s `session_part` column,
    "main"/"sart") plus the inter-process gap between the main session's
    last event and the SART sub-session's first, if both are present."""
    windows: list[Window] = []
    for extractor in WINDOW_EXTRACTORS.values():
        windows.extend(extractor(events))

    if "session_part" not in events.columns:
        span = (float(events["timestamp_host_utc"].min()), float(events["timestamp_host_utc"].max()))
        windows.extend(extract_gap_windows(events, windows, span, min_gap_s=min_gap_s))
        return windows

    parts = sorted(p for p in events["session_part"].dropna().unique())
    part_spans = {}
    for part in parts:
        part_df = events[events["session_part"] == part]
        span = (float(part_df["timestamp_host_utc"].min()), float(part_df["timestamp_host_utc"].max()))
        part_spans[part] = span
        part_windows = [w for w in windows if span[0] - 1e-6 <= w.t_start <= span[1] + 1e-6 and span[0] - 1e-6 <= w.t_end <= span[1] + 1e-6]
        windows.extend(extract_gap_windows(part_df, part_windows, span, min_gap_s=min_gap_s))

    if "main" in part_spans and "sart" in part_spans:
        main_end = part_spans["main"][1]
        sart_start = part_spans["sart"][0]
        if sart_start > main_end:
            windows.append(
                Window(
                    task="transition",
                    window_type="transition",
                    t_start=main_end,
                    t_end=sart_start,
                    meta={"reason": "gap between main session end and SART sub-session start (separate processes/recordings)"},
                )
            )

    return windows


def windows_to_dataframe(windows: list[Window]) -> pd.DataFrame:
    """Flattens a list[Window] into a tidy DataFrame for quick inspection --
    `stimulus`/`labels`/`meta` stay as dict-valued columns; use
    `pd.json_normalize`/`.apply(pd.Series)` if you need those flattened."""
    return pd.DataFrame(
        [
            {
                "task": w.task,
                "window_type": w.window_type,
                "t_start": w.t_start,
                "t_end": w.t_end,
                "duration_s": w.t_end - w.t_start,
                "block_index": w.block_index,
                "trial_index": w.trial_index,
                "condition_label": w.condition_label,
                "stimulus": w.stimulus,
                "labels": w.labels,
                "meta": w.meta,
            }
            for w in windows
        ]
    )
