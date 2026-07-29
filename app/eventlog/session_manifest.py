import json
import time
from pathlib import Path


def write_session_manifest(
    session_dir: Path,
    session_id: str,
    participant_id: str,
    config_snapshot: dict,
    sync_records: dict,
    demo_scale: float,
) -> Path:
    """Session-level metadata: participant, device sync records, config snapshot.

    sync_records is device_id -> list[SyncRecord-as-dict] (list-per-device even
    in v1's single-sync design, per §5.3, so a future dual-sync upgrade doesn't
    need a schema migration here).
    """
    manifest = {
        "session_id": session_id,
        "participant_id": participant_id,
        "created_at_utc": time.time(),
        "demo_scale": demo_scale,
        "device_sync_records": sync_records,
        "config_snapshot": config_snapshot,
    }
    path = Path(session_dir) / "session_manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    return path
