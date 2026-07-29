import asyncio
from dataclasses import asdict, dataclass, field


@dataclass
class SyncRecord:
    """One BLE sync attempt against one device.

    Kept as a list-per-device in session_manifest.json (see §5.3 of
    Engineering_Document.md) even though v1 only ever produces one record per
    device, so a future dual-sync/drift-correction upgrade doesn't require a
    schema migration -- only DeviceSyncManager and the offline offset
    reconstruction step should need to change.
    """

    device_role: str
    device_id: str
    protocol: str
    sync_time_host_utc: float
    result: str  # "ok" | "error"
    detail: str = ""


class DeviceSyncManager:
    """Single entry point for syncing every enabled device at session start.

    Dispatches to a protocol-specific adapter (real hardware) or the mock
    adapter (devices.mode == "mock" in session_config.yaml), so task/session
    code never needs to know which protocol a given device speaks.
    """

    def __init__(self, devices_config: dict, mode: str):
        self.devices_config = devices_config
        self.mode = mode  # "mock" | "real"

    async def sync_all(self) -> dict:
        """Returns device_role -> list[SyncRecord-as-dict]."""
        records: dict = {}
        for role, cfg in self.devices_config.items():
            if not isinstance(cfg, dict) or not cfg.get("enabled"):
                continue
            record = await self._sync_one(role, cfg)
            records[role] = [asdict(record)]
        return records

    async def _sync_one(self, role: str, cfg: dict) -> SyncRecord:
        if self.mode == "mock":
            from app.sync.mock_sync import sync_mock

            return await sync_mock(role, cfg)

        protocol = cfg["protocol"]
        if protocol == "gatt_struct":
            from app.sync.ear_eeg_sync import sync_real as sync_fn
        elif protocol == "nus_text":
            from app.sync.wristband_sync import sync_real as sync_fn
        else:
            raise ValueError(
                f"No real-hardware sync adapter for protocol '{protocol}' "
                f"(device role '{role}'). Add one in app/sync/ or run in mock mode."
            )
        return await sync_fn(role, cfg)


def sync_all_blocking(devices_config: dict, mode: str) -> dict:
    """Convenience wrapper for callers not already inside an event loop."""
    manager = DeviceSyncManager(devices_config, mode)
    return asyncio.run(manager.sync_all())
