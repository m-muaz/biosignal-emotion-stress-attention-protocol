"""Real-hardware BLE sync adapter for the wristband.

Protocol matches ble_sync_wristbands.py (provided script): Nordic UART
Service, plain-text "SYNC_MS <unix_ms>" command, "SYNC_OK"/"SYNC_ERR" reply.
See Engineering_Document.md §5.2.1.
"""

import asyncio
import time

from bleak import BleakClient, BleakScanner

from app.sync.device_sync import SyncRecord

UART_RX_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
UART_TX_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"

SCAN_TIMEOUT_SEC = 8.0
ACK_TIMEOUT_SEC = 10.0


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
            f"({[d.name for d in matches]}) -- set devices.wristband.device_mac "
            f"in session_config.yaml to disambiguate which one belongs to "
            f"this participant."
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
            protocol="nus_text",
            sync_time_host_utc=time.time(),
            result="error",
            detail=f"no device found matching prefix '{name_prefix}'",
        )

    reply_event = asyncio.Event()
    replies: list[str] = []
    buffer = bytearray()

    def _handle_notify(_, data: bytearray):
        buffer.extend(data)
        while True:
            try:
                idx = buffer.index(ord("\n"))
            except ValueError:
                break
            line = bytes(buffer[:idx]).decode(errors="ignore").strip()
            del buffer[: idx + 1]
            if line:
                replies.append(line)
                if "SYNC_OK" in line or "SYNC_ERR" in line:
                    reply_event.set()

    sync_time_ms = int(time.time() * 1000)
    command = f"SYNC_MS {sync_time_ms}\n".encode()

    try:
        async with BleakClient(device, timeout=ACK_TIMEOUT_SEC) as client:
            await client.start_notify(UART_TX_UUID, _handle_notify)
            await client.write_gatt_char(UART_RX_UUID, command, response=True)
            await asyncio.wait_for(reply_event.wait(), timeout=ACK_TIMEOUT_SEC)
            await client.stop_notify(UART_TX_UUID)

        ok = any("SYNC_OK" in r for r in replies)
        return SyncRecord(
            device_role=role,
            device_id=device.address,
            protocol="nus_text",
            sync_time_host_utc=sync_time_ms / 1000.0,
            result="ok" if ok else "error",
            detail=" | ".join(replies),
        )
    except Exception as exc:
        return SyncRecord(
            device_role=role,
            device_id=device.address,
            protocol="nus_text",
            sync_time_host_utc=sync_time_ms / 1000.0,
            result="error",
            detail=repr(exc),
        )
