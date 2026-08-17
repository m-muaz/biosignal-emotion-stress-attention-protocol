"""CLI: turn raw participant folder(s) into canonical per-stream Parquet +
a sync_report.json (full resolution/sync audit trail) per participant.

Usage:
    python -m dataset.build_dataset --participant-dir "...\\part-P007" --out-dir "...\\processed"
    python -m dataset.build_dataset --root "C:\\emotion-stress-attention-task-ear-wristband-device-data\\data-collection-sessions" --out-dir "...\\processed"
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from dataset.session_resolver import DeviceResolution, resolve_participant
from dataset.sync import sync_all


def _device_resolution_to_dict(d: DeviceResolution) -> dict:
    out = asdict(d)
    out["files"] = {k: str(v) for k, v in d.files.items()}
    out["excluded"] = [{"path": str(e.path), "reason": e.reason} for e in d.excluded]
    return out


def build_participant(participant_dir: Path, out_dir: Path, validate_crc: bool = False) -> dict:
    participant_dir = Path(participant_dir)
    out_dir = Path(out_dir)
    resolved = resolve_participant(participant_dir)
    participant_out = out_dir / resolved.participant_id
    participant_out.mkdir(parents=True, exist_ok=True)

    synced = sync_all(resolved, validate_crc=validate_crc)

    written = []
    stream_qc = {device: {} for device in synced}
    for device, streams in synced.items():
        if device == "events":
            if len(streams):
                path = participant_out / "events.parquet"
                # event_payload_json is a dict per row with a schema that varies
                # by event_type -- pyarrow can't infer one Arrow type for that
                # column, so store it as a JSON string; windows.py re-parses it.
                out_df = streams.copy()
                out_df["event_payload_json"] = out_df["event_payload_json"].apply(json.dumps)
                out_df.to_parquet(path, index=False)
                written.append(str(path))
            continue
        for role, s in streams.items():
            df = s.data.copy()
            df.insert(0, "wall_utc_s", s.wall_utc_s)
            path = participant_out / f"{device}.{role}.parquet"
            df.to_parquet(path, index=False)
            written.append(str(path))
            stream_qc[device][role] = {"n_samples": len(s.wall_utc_s), "sample_rate_hz": s.sample_rate_hz, **s.qc}

    devices_report = {name: _device_resolution_to_dict(d) for name, d in resolved.devices.items()}
    for device, qc in stream_qc.items():
        if device in devices_report:
            devices_report[device]["stream_qc"] = qc

    report = {
        "participant_id": resolved.participant_id,
        "participant_dir": str(resolved.participant_dir),
        "log_start_reference_s": resolved.log_start_reference_s,
        "main_session_dir": str(resolved.main_session_dir) if resolved.main_session_dir else None,
        "sart_session_dir": str(resolved.sart_session_dir) if resolved.sart_session_dir else None,
        "global_warnings": resolved.global_warnings,
        "devices": devices_report,
        "streams_written": written,
    }

    report_path = participant_out / "sync_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    return report


def build_all(root: Path, out_dir: Path, validate_crc: bool = False) -> list[dict]:
    root = Path(root)
    reports = []
    for date_dir in sorted(root.iterdir()):
        if not date_dir.is_dir():
            continue
        for participant_dir in sorted(date_dir.iterdir()):
            if not (participant_dir.is_dir() and participant_dir.name.lower().startswith("part-")):
                continue
            print(f"Building {participant_dir} ...")
            try:
                report = build_participant(participant_dir, out_dir, validate_crc=validate_crc)
                n_warnings = sum(len(d["warnings"]) for d in report["devices"].values()) + len(report["global_warnings"])
                print(f"  OK -- {len(report['streams_written'])} streams written, {n_warnings} warning(s) (see sync_report.json)")
                reports.append(report)
            except Exception as exc:
                print(f"  FAILED: {exc}")
                reports.append({"participant_dir": str(participant_dir), "error": str(exc)})
    return reports


def main():
    parser = argparse.ArgumentParser(description="Build the canonical synced dataset from raw collection-session data.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--participant-dir", type=str, help="Build a single participant folder (e.g. .../part-P007)")
    group.add_argument("--root", type=str, help="Build every participant under a data-collection-sessions root")
    parser.add_argument("--out-dir", type=str, required=True, help="Output directory for canonical Parquet + reports")
    parser.add_argument("--validate-crc", action="store_true", help="Validate ADS1299 CRC-8 on every record (slower)")
    args = parser.parse_args()

    if args.participant_dir:
        report = build_participant(Path(args.participant_dir), Path(args.out_dir), validate_crc=args.validate_crc)
        print(json.dumps({k: v for k, v in report.items() if k != "devices"}, indent=2, default=str))
    else:
        build_all(Path(args.root), Path(args.out_dir), validate_crc=args.validate_crc)


if __name__ == "__main__":
    main()
