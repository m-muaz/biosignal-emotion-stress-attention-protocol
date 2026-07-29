"""Real-hardware BLE sync adapter for the ear-EEG devices.

Protocol reverse-engineered from ble_data_provider.py in the out-ear firmware
repo (ADS1299_BLE_muaz, branch feat/ads-ppg-timesync-ble) -- see
Engineering_Document.md §5.2.1 for the full write-up. The in-ear firmware
(ADS1299_BLE_gaoteng) does not have this capability yet (open item #2); once
it's ported, it should speak the same protocol and this adapter should work
unchanged.

Deliberate deviation from the reference script: ble_data_provider.py treats a
sync ACK timeout as non-fatal and silently lets the device fall back to
boot-relative SD timestamps. We treat it as fatal instead -- a silent
fallback here means that device's entire session has no usable wall-clock
mapping, discovered only during offline analysis.
"""

import asyncio
import struct
import time

from bleak import BleakClient, BleakScanner

from app.sync.device_sync import SyncRecord

COMMAND_CHAR_UUID = "0000abf3-0000-1000-8000-00805f9b34fb"
STATUS_CHAR_UUID = "0000abf4-0000-1000-8000-00805f9b34fb"
CMD_SET_TIME = 0x04

SCAN_TIMEOUT_SEC = 5.0
ACK_TIMEOUT_SEC = 3.0


async def _find_device(name_prefix: str, device_mac: str | None):
    devices = await BleakScanner.discover(timeout=SCAN_TIMEOUT_SEC)
    if device_mac:
        return next((d for d in devices if d.address == device_mac), None)

    matches = [d for d in devices if (d.name or "").startswith(name_prefix)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple devices match prefix '{name_prefix}' "
            f"({[d.name for d in matches]}) -- set devices.<role>.device_mac "
            f"in session_config.yaml to disambiguate."
        )
    return None


async def sync_real(role: str, cfg: dict) -> SyncRecord:
    name_prefix = cfg["name_prefix"]
    device_mac = cfg.get("device_mac")

    device = await _find_device(name_prefix, device_mac)
    if device is None:
        return SyncRecord(
            device_role=role,
            device_id="",
            protocol="gatt_struct",
            sync_time_host_utc=time.time(),
            result="error",
            detail=f"no device found matching prefix '{name_prefix}'",
        )

    ack_event = asyncio.Event()
    ack_payload: list[bytes] = []

    def _handle_status(_, data: bytearray):
        if len(data) >= 9 and data[0] == CMD_SET_TIME:
            ack_payload.append(bytes(data))
            ack_event.set()

    sync_time = time.time()
    try:
        async with BleakClient(device, timeout=10.0) as client:
            await client.start_notify(STATUS_CHAR_UUID, _handle_status)
            payload = struct.pack("<Bd", CMD_SET_TIME, sync_time)
            await client.write_gatt_char(COMMAND_CHAR_UUID, payload)
            await asyncio.wait_for(ack_event.wait(), timeout=ACK_TIMEOUT_SEC)
            await client.stop_notify(STATUS_CHAR_UUID)

        echoed = struct.unpack("<d", ack_payload[0][1:9])[0]
        return SyncRecord(
            device_role=role,
            device_id=device.address,
            protocol="gatt_struct",
            sync_time_host_utc=sync_time,
            result="ok",
            detail=f"device echoed {echoed:.3f}",
        )
    except Exception as exc:
        return SyncRecord(
            device_role=role,
            device_id=device.address,
            protocol="gatt_struct",
            sync_time_host_utc=sync_time,
            result="error",
            detail=repr(exc),
        )
