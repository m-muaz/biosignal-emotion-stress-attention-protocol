import time

from app.sync.device_sync import SyncRecord


async def sync_mock(role: str, cfg: dict) -> SyncRecord:
    """Simulated sync used for demo runs with no hardware present.

    Mirrors the shape of a real SyncRecord (see ear_eeg_sync.py /
    wristband_sync.py) so downstream code (session manifest, offline
    reconstruction) doesn't need to special-case mock mode.
    """
    return SyncRecord(
        device_role=role,
        device_id=f"mock-{role}",
        protocol=cfg.get("protocol", "mock"),
        sync_time_host_utc=time.time(),
        result="ok",
        detail="mock sync -- no hardware present",
    )
