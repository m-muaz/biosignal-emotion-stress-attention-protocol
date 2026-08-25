"""CLI: build the canonical stress-task tables (dataset/stress_labels/canonical_tables.py)
for one or every participant and write them to `<processed_dir>/<pid>/stress_*.parquet`,
alongside dataset/build_dataset.py's existing per-stream Parquet output.

Usage:
    python -m dataset.stress_labels.build_canonical \\
        --processed-dir data/processed --raw-root /path/to/data-collection-sessions
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset.stress_labels import canonical_tables as ct
from dataset.stress_labels.protocol import load_config_snapshot
from dataset.windows import load_events_parquet

TABLE_BUILDERS = [
    "stress_blocks",
    "raindrop_trials",
    "raindrop_unmatched_events",
    "highway_obstacles",
    "highway_state",
    "highway_self_report",
]


def build_participant_canonical_tables(processed_dir: Path, raw_root: Path, participant_id: str) -> dict:
    processed_dir = Path(processed_dir)
    participant_dir = processed_dir / participant_id
    events_path = participant_dir / "events.parquet"
    if not events_path.exists():
        raise FileNotFoundError(f"{events_path} not found -- run dataset.build_dataset for {participant_id} first")

    events = load_events_parquet(events_path)
    config_snapshot = load_config_snapshot(raw_root, participant_id)
    session_id = events["session_id"].dropna().iloc[0] if events["session_id"].notna().any() else None

    blocks = ct.build_stress_blocks(participant_id, session_id, events, config_snapshot)
    trials = ct.build_raindrop_trials(participant_id, session_id, events, config_snapshot)
    unmatched = ct.build_raindrop_unmatched_events(participant_id, session_id, events)
    obstacles = ct.build_highway_obstacles(participant_id, session_id, events)
    state = ct.build_highway_state(participant_id, session_id, events, blocks)
    self_report = ct.build_highway_self_report(participant_id, session_id, events, config_snapshot)

    tables = {
        "stress_blocks": blocks,
        "raindrop_trials": trials,
        "raindrop_unmatched_events": unmatched,
        "highway_obstacles": obstacles,
        "highway_state": state,
        "highway_self_report": self_report,
    }

    row_counts = {}
    for name, df in tables.items():
        path = participant_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        row_counts[name] = len(df)

    return row_counts


def build_all(processed_dir: Path, raw_root: Path) -> dict:
    processed_dir = Path(processed_dir)
    report = {}
    for participant_dir in sorted(processed_dir.iterdir()):
        if not participant_dir.is_dir() or not (participant_dir / "events.parquet").exists():
            continue
        pid = participant_dir.name
        try:
            report[pid] = build_participant_canonical_tables(processed_dir, raw_root, pid)
            print(f"{pid}: OK -- {report[pid]}")
        except Exception as exc:
            report[pid] = {"error": str(exc)}
            print(f"{pid}: FAILED -- {exc}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Build canonical stress-task tables from processed events.parquet + raw session_manifest.json.")
    parser.add_argument("--processed-dir", type=str, required=True, help="dataset.build_dataset output dir")
    parser.add_argument("--raw-root", type=str, required=True, help="data-collection-sessions root (for session_manifest.json config_snapshot)")
    parser.add_argument("--participant-id", type=str, default=None, help="build one participant only; default builds every participant found")
    args = parser.parse_args()

    if args.participant_id:
        report = {args.participant_id: build_participant_canonical_tables(Path(args.processed_dir), Path(args.raw_root), args.participant_id)}
    else:
        report = build_all(Path(args.processed_dir), Path(args.raw_root))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
