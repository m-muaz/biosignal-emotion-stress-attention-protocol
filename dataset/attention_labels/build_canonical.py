"""CLI: build the canonical SART/Stroop/Schulte tables
(dataset/attention_labels/canonical_tables.py) for one or every participant
and write them to `<processed_dir>/<pid>/{attention_blocks,sart_trials,
stroop_trials,schulte_blocks,schulte_clicks}.parquet`, alongside
dataset/build_dataset.py's existing per-stream Parquet output.

Usage:
    python -m dataset.attention_labels.build_canonical \\
        --processed-dir data/processed
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset.attention_labels import canonical_tables as ct
from dataset.windows import load_events_parquet

TABLE_BUILDERS = [
    "attention_blocks",
    "sart_trials",
    "stroop_trials",
    "schulte_blocks",
    "schulte_clicks",
]


def build_participant_canonical_tables(processed_dir: Path, participant_id: str) -> dict:
    processed_dir = Path(processed_dir)
    participant_dir = processed_dir / participant_id
    events_path = participant_dir / "events.parquet"
    if not events_path.exists():
        raise FileNotFoundError(f"{events_path} not found -- run dataset.build_dataset for {participant_id} first")

    events = load_events_parquet(events_path)
    session_id = events["session_id"].dropna().iloc[0] if events["session_id"].notna().any() else None

    blocks = ct.build_attention_blocks(participant_id, session_id, events)
    sart_trials = ct.build_sart_trials(participant_id, session_id, events)
    stroop_trials = ct.build_stroop_trials(participant_id, session_id, events)
    schulte_blocks = ct.build_schulte_blocks(participant_id, session_id, events)
    schulte_clicks = ct.build_schulte_clicks(participant_id, session_id, events, schulte_blocks)

    tables = {
        "attention_blocks": blocks,
        "sart_trials": sart_trials,
        "stroop_trials": stroop_trials,
        "schulte_blocks": schulte_blocks,
        "schulte_clicks": schulte_clicks,
    }

    row_counts = {}
    for name, df in tables.items():
        path = participant_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        row_counts[name] = len(df)
    if not schulte_clicks.empty:
        row_counts["schulte_clicks_unassigned"] = schulte_clicks.attrs.get("unassigned_click_count", 0)
        row_counts["schulte_clicks_ambiguous"] = schulte_clicks.attrs.get("ambiguous_click_count", 0)

    return row_counts


def build_all(processed_dir: Path) -> dict:
    processed_dir = Path(processed_dir)
    report = {}
    for participant_dir in sorted(processed_dir.iterdir()):
        if not participant_dir.is_dir() or not (participant_dir / "events.parquet").exists():
            continue
        pid = participant_dir.name
        try:
            report[pid] = build_participant_canonical_tables(processed_dir, pid)
            print(f"{pid}: OK -- {report[pid]}")
        except Exception as exc:
            report[pid] = {"error": str(exc)}
            print(f"{pid}: FAILED -- {exc}")
    return report


def main():
    parser = argparse.ArgumentParser(description="Build canonical SART/Stroop/Schulte tables from processed events.parquet.")
    parser.add_argument("--processed-dir", type=str, required=True, help="dataset.build_dataset output dir")
    parser.add_argument("--participant-id", type=str, default=None, help="build one participant only; default builds every participant found")
    args = parser.parse_args()

    if args.participant_id:
        report = {args.participant_id: build_participant_canonical_tables(Path(args.processed_dir), args.participant_id)}
    else:
        report = build_all(Path(args.processed_dir))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
