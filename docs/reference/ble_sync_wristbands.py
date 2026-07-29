import argparse
import asyncio
import csv
import time
from datetime import datetime
from pathlib import Path

from bleak import BleakClient, BleakScanner


UART_SERVICE_UUID = "6E400001-B5A3-F393-E0A9-E50E24DCCA9E"
UART_RX_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E"
UART_TX_UUID = "6E400003-B5A3-F393-E0A9-E50E24DCCA9E"


class LineBuffer:
    def __init__(self):
        self._buffer = bytearray()

    def feed(self, data):
        self._buffer.extend(data)
        lines = []

        while True:
            try:
                newline_index = self._buffer.index(ord("\n"))
            except ValueError:
                break

            raw_line = self._buffer[:newline_index]
            del self._buffer[:newline_index + 1]

            text = raw_line.decode(errors="ignore").strip()
            if text:
                lines.append(text)

        return lines

    def partial_text(self):
        return self._buffer.decode(errors="ignore").strip()


async def scan_wristbands(prefix, scan_seconds):
    devices = await BleakScanner.discover(timeout=scan_seconds)
    matches = []
    seen = set()

    for device in devices:
        name = device.name or ""
        if name.startswith(prefix) and device.address not in seen:
            matches.append(device)
            seen.add(device.address)

    return sorted(matches, key=lambda d: d.name or d.address)


async def sync_one_device(device, timeout_seconds):
    reply_event = asyncio.Event()
    replies = []
    reply_buffer = LineBuffer()

    def handle_notify(_, data):
        for text in reply_buffer.feed(data):
            replies.append(text)
            if "SYNC_OK" in text or "SYNC_ERR" in text:
                reply_event.set()

    started_ms = int(time.time() * 1000)
    command = f"SYNC_MS {started_ms}\n".encode()

    try:
        async with BleakClient(device, timeout=timeout_seconds) as client:
            await client.start_notify(UART_TX_UUID, handle_notify)
            await client.write_gatt_char(UART_RX_UUID, command, response=True)
            await asyncio.wait_for(reply_event.wait(), timeout=timeout_seconds)
            await client.stop_notify(UART_TX_UUID)

        reply = " | ".join(replies)
        ok = any("SYNC_OK" in item for item in replies)
        return {
            "device_name": device.name or "",
            "address": device.address,
            "sync_ms": started_ms,
            "result": "ok" if ok else "error",
            "reply": reply,
        }
    except Exception as exc:
        reply = repr(exc)
        partial = reply_buffer.partial_text()
        if partial:
            reply = f"{reply}; partial_reply={partial!r}"

        return {
            "device_name": device.name or "",
            "address": device.address,
            "sync_ms": started_ms,
            "result": "error",
            "reply": reply,
        }


def append_log(log_path, rows):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.exists()

    with log_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["local_time", "device_name", "address", "sync_ms", "result", "reply"],
        )
        if not file_exists:
            writer.writeheader()

        for row in rows:
            writer.writerow({"local_time": datetime.now().isoformat(timespec="seconds"), **row})


async def run_sync_cycle(args):
    print(f"Scanning for {args.prefix}* wristbands for {args.scan_seconds} s...")
    devices = await scan_wristbands(args.prefix, args.scan_seconds)

    if args.expected_count and len(devices) < args.expected_count:
        print(f"Warning: found {len(devices)} devices, expected {args.expected_count}.")

    if not devices:
        rows = [{
            "device_name": "",
            "address": "",
            "sync_ms": int(time.time() * 1000),
            "result": "error",
            "reply": "no devices found",
        }]
        append_log(args.log, rows)
        print("No wristbands found.")
        return rows

    rows = []
    for device in devices:
        print(f"Syncing {device.name} [{device.address}]...")
        row = await sync_one_device(device, args.timeout_seconds)
        rows.append(row)
        print(f"  {row['result']}: {row['reply']}")

    append_log(args.log, rows)
    return rows


async def main_async(args):
    cycle = 0
    while args.cycles == 0 or cycle < args.cycles:
        cycle += 1
        print(f"\n=== Sync cycle {cycle} ===")
        await run_sync_cycle(args)

        if args.cycles != 0 and cycle >= args.cycles:
            break

        sleep_seconds = args.interval_minutes * 60
        print(f"Sleeping {sleep_seconds} s before next sync cycle...")
        await asyncio.sleep(sleep_seconds)


def parse_args():
    parser = argparse.ArgumentParser(description="Batch BLE time sync for WristBand devices.")
    parser.add_argument("--prefix", default="WristBand_", help="BLE name prefix to scan for.")
    parser.add_argument("--expected-count", type=int, default=7, help="Expected wristband count.")
    parser.add_argument("--scan-seconds", type=float, default=8.0, help="BLE scan duration.")
    parser.add_argument("--timeout-seconds", type=float, default=10.0, help="Connect/reply timeout.")
    parser.add_argument("--interval-minutes", type=float, default=30.0, help="Repeat interval.")
    parser.add_argument("--cycles", type=int, default=1, help="Number of cycles; 0 means forever.")
    parser.add_argument("--log", type=Path, default=Path("sync_log.csv"), help="CSV sync log path.")
    return parser.parse_args()


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
