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
    rng_seed: float,
) -> Path:
    """Session-level metadata: participant, device sync records, config snapshot.

    sync_records is device_id -> list[SyncRecord-as-dict] (list-per-device even
    in v1's single-sync design, per §5.3, so a future dual-sync upgrade doesn't
    need a schema migration here).

    rng_seed is recorded so a run can be replayed exactly if needed -- it's the
    session's start time, not derived from participant_id, so re-running the
    same participant_id never reproduces the same trial-question sequence.
    """
    manifest = {
        "session_id": session_id,
        "participant_id": participant_id,
        "created_at_utc": time.time(),
        "demo_scale": demo_scale,
        "rng_seed": rng_seed,
        "device_sync_records": sync_records,
        "config_snapshot": config_snapshot,
    }
    path = Path(session_dir) / "session_manifest.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    return path
