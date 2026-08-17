"""Loads dataset/overrides.yaml and exposes it as typed lookups.

Kept as one small module (rather than inlining `yaml.safe_load` calls in the
resolver) so there's a single place that validates the schema and a single
place other code imports from.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

OVERRIDES_PATH = Path(__file__).resolve().parent / "overrides.yaml"


@dataclass
class EndAnchorReference:
    session_dir_glob: str  # e.g. "*_sart_*" -- matched against the participant's data_collection_logs subdirs
    event_type: str  # e.g. "session_end"
    reason: str


@dataclass
class WristbandOverride:
    use_session: int | None = None
    end_anchor: EndAnchorReference | None = None
    reason: str = ""


@dataclass
class ParticipantOverrides:
    participant_id: str
    wristband: WristbandOverride | None = None


def _load_raw(path: Path = OVERRIDES_PATH) -> dict:
    if not path.exists():
        return {"participants": {}}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {"participants": {}}


def load_overrides(path: Path = OVERRIDES_PATH) -> dict[str, ParticipantOverrides]:
    raw = _load_raw(path)
    out: dict[str, ParticipantOverrides] = {}
    for pid, entry in (raw.get("participants") or {}).items():
        wb_entry = entry.get("wristband")
        wristband = None
        if wb_entry:
            sync_override = wb_entry.get("sync_override")
            end_anchor = None
            if sync_override and sync_override.get("mode") == "end_anchor":
                ref = sync_override["reference"]
                end_anchor = EndAnchorReference(
                    session_dir_glob=ref["session_dir_glob"],
                    event_type=ref["event_type"],
                    reason=sync_override.get("reason", ""),
                )
            wristband = WristbandOverride(
                use_session=wb_entry.get("use_session"),
                end_anchor=end_anchor,
                reason=wb_entry.get("reason", ""),
            )
        out[pid] = ParticipantOverrides(participant_id=pid, wristband=wristband)
    return out


def get_participant_overrides(participant_id: str, path: Path = OVERRIDES_PATH) -> ParticipantOverrides:
    return load_overrides(path).get(participant_id, ParticipantOverrides(participant_id=participant_id))
