"""Parser for the session runner's events.jsonl + session_manifest.json.

events.jsonl: one JSON object per line, keys (see app/eventlog/event_logger.py):
    timestamp_host_utc   float, Unix epoch seconds (PC wall clock) -- this IS
                         the master timeline every device stream gets pulled
                         onto; no correction needed.
    timestamp_monotonic  float, PC monotonic/perf-counter seconds (session-
                         runner-process-uptime-relative, NOT epoch-anchored)
    session_id, participant_id, task, block_index, trial_index,
    condition_label, event_type, event_payload_json (dict, schema varies by
    event_type)

A participant can have more than one events.jsonl (e.g. the main session
plus a separate `_sart_<epoch>` sub-session) -- callers pass the specific
directory, this module doesn't guess which one(s) to load.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


def parse_events_jsonl(path: Path) -> pd.DataFrame:
    path = Path(path)
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{lineno}: malformed JSON line: {exc}") from exc
    df = pd.DataFrame(rows)
    if "timestamp_host_utc" not in df.columns:
        raise ValueError(f"{path}: no rows / missing timestamp_host_utc column")
    return df.sort_values("timestamp_host_utc", kind="stable").reset_index(drop=True)


@dataclass
class SessionManifest:
    path: Path
    session_id: str
    participant_id: str
    created_at_utc: float
    config_snapshot: dict
    device_sync_records: dict  # expected empty {} in practice -- see overrides.yaml


def parse_session_manifest(path: Path) -> SessionManifest:
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return SessionManifest(
        path=path,
        session_id=d.get("session_id", ""),
        participant_id=d.get("participant_id", ""),
        created_at_utc=float(d["created_at_utc"]),
        config_snapshot=d.get("config_snapshot", {}),
        device_sync_records=d.get("device_sync_records", {}),
    )
