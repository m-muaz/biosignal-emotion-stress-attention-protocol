"""Protocol-version metadata for the stress task label pipeline.

Two independent things live here:

1. The `attention_highway` obstacle-scoring bug (see
   `data-collection-sessions/notes/notes.txt`, logged 2026-08-13, fixed same
   day): the first 4 participants collected (P001-P004, 08-11 through
   08-13) ran under code where `concurrent_obstacles == lanes`, which by
   pigeonhole forced same-wave duplicate obstacle_spawn/obstacle_resolved
   pairs and inflated avoided/collision tallies ~30-100%. This is NOT fully
   correctable after the fact (see notes.txt Issue 2/4), so affected
   participants are tagged, never silently dropped or corrected.

2. Reading a participant's `session_manifest.json` `config_snapshot` --
   this is where tier-level task-pressure config (spawn_interval_sec,
   fall_duration_sec, concurrent_obstacles, self-report item list) actually
   lives; it is NOT present in individual event payloads. Reuses
   `session_resolver.resolve_participant`'s already-validated main-session-
   dir discovery (handles the 3 observed naming-convention variants) rather
   than re-deriving it.
"""

from __future__ import annotations

from pathlib import Path

from dataset.raw.events_jsonl import parse_session_manifest
from dataset.session_resolver import resolve_participant

# Confirmed against data-collection-sessions/notes/notes.txt: "PARTICIPANTS
# RUN UNDER THE BUGGY CODE (before this fix): 4" -- cross-checked against
# collection dates (P001 08-11, P002/P003 08-12, P004 08-13; fix landed
# 08-13) and docs/Dataset_Sync_Design.md / docs/label_design.mdx, which both
# independently describe "the first 4 participants."
HIGHWAY_PRE_FIX_PARTICIPANTS = frozenset({"P001", "P002", "P003", "P004"})


def highway_protocol_version(participant_id: str) -> str:
    return "highway_pre_fix" if participant_id in HIGHWAY_PRE_FIX_PARTICIPANTS else "highway_post_fix"


def highway_known_bug(participant_id: str) -> bool:
    return participant_id in HIGHWAY_PRE_FIX_PARTICIPANTS


def find_participant_raw_dir(raw_root: Path, participant_id: str) -> Path:
    """Raw participant folders are nested one level under a date-stamped
    directory (`<raw_root>/08-11-2026-Data/part-P001/`); walk to find it
    rather than assuming a fixed date-dir name."""
    raw_root = Path(raw_root)
    matches = sorted(raw_root.glob(f"*/part-{participant_id}"))
    if not matches:
        raise FileNotFoundError(f"no part-{participant_id} folder found under {raw_root}")
    if len(matches) > 1:
        raise ValueError(f"multiple part-{participant_id} folders found under {raw_root}: {matches}")
    return matches[0]


def load_config_snapshot(raw_root: Path, participant_id: str) -> dict:
    """The task-pressure config (tier definitions, self-report item list,
    etc.) a participant's session actually ran with -- read from their main
    session's `session_manifest.json`, not reconstructed from event
    payloads (which don't carry it)."""
    participant_dir = find_participant_raw_dir(raw_root, participant_id)
    resolved = resolve_participant(participant_dir)
    if resolved.main_session_dir is None:
        raise FileNotFoundError(f"no main session directory resolved for {participant_id} under {participant_dir}")
    manifest = parse_session_manifest(resolved.main_session_dir / "session_manifest.json")
    return manifest.config_snapshot


def stress_tier_config(config_snapshot: dict, tier: int) -> dict | None:
    tiers = config_snapshot.get("stress_task", {}).get("tiers", [])
    return next((t for t in tiers if t.get("id") == tier), None)


def highway_tier_config(config_snapshot: dict, tier: int) -> dict | None:
    tiers = config_snapshot.get("attention_highway_task", {}).get("tiers", [])
    return next((t for t in tiers if t.get("id") == tier), None)


def highway_self_report_items(config_snapshot: dict) -> list[str]:
    """The set of subjective dimensions actually collected varies by
    session (observed: some sessions configure only 5 of the 8 possible
    items, e.g. no frustration/arousal/valence) -- always read this from the
    session's own config rather than assuming a fixed 8-item list."""
    return list(config_snapshot.get("attention_highway_task", {}).get("self_report", {}).get("items", []))
