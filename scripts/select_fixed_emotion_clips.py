#!/usr/bin/env python3
"""
Build a fixed, predefined emotion-clip selection from OpenLAV's video_data.csv
metadata, per PI request (2026-08-02): the same clip set should be shown to
every participant (only presentation order randomized at runtime), rather
than each participant getting a fresh random sample from the full manifest
pool.

Selection method: for the positive/negative valence groups, rank each
group's clips (drawn from emotion_manifest.json, so only already-curated,
downloaded, language-free clips are considered) by video_data.csv's
`arousal_wsd` score, descending -- arousal is used as a proxy for "induces
a strong emotional reaction" per the PI's suggestion; valence's sign
already determines which group a clip belongs to. For the neutral group,
rank by arousal_wsd ASCENDING (calmest) instead, since a neutral control
condition should minimize induced arousal rather than maximize it.

Uses `arousal_wsd`/`valence_wsd`, NOT the raw `valence`/`arousal` columns --
the raw columns mix units within the same column (some rows are tiny
decimals like 0.46, others are large sums like 1244, likely a scraping/
aggregation artifact), which produces a nonsensical ranking (e.g. several
"top arousal" positive clips would land at near-zero or negative raw
arousal). `_wsd` is consistently scaled across all clips and produces a
sane ranking (negative-group clips cluster meaningfully higher than
positive/neutral, as expected for genuinely intense stimuli).

Output feeds app/config/session_config.yaml's emotion_task.fixed_clips_path,
read by app/config/loader.py:load_fixed_clips() when
emotion_task.clip_selection_mode == "fixed".

Usage:
    python scripts/select_fixed_emotion_clips.py
    python scripts/select_fixed_emotion_clips.py --n-per-group 4
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "app" / "config" / "emotion_manifest.json"
DEFAULT_METADATA = REPO_ROOT / "assets" / "clips" / "video_data.csv"
DEFAULT_OUTPUT = REPO_ROOT / "app" / "config" / "emotion_fixed_clips.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--n-per-group", type=int, default=3,
        help="Clips to select per valence group -- should match emotion_task.clips_per_group in session_config.yaml.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    with args.manifest.open("r", encoding="utf-8") as f:
        clips = json.load(f)["clips"]

    metadata_by_title: dict[str, dict] = {}
    with args.metadata.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            metadata_by_title[row["video_code"]] = row

    enriched = []
    missing = []
    for clip in clips:
        row = metadata_by_title.get(clip["title"])
        if row is None:
            missing.append(clip["title"])
            continue
        enriched.append({**clip, "valence_score": float(row["valence_wsd"]), "arousal_score": float(row["arousal_wsd"])})
    if missing:
        print(f"WARNING: {len(missing)} manifest clips have no video_data.csv match, skipped: {missing}")

    groups: dict[str, list[dict]] = {}
    for clip in enriched:
        groups.setdefault(clip["valence_group"], []).append(clip)

    selection: dict = {
        "_comment": (
            "Predefined fixed clip set (PI request, 2026-08-02): the same clips are "
            "shown to every participant -- only presentation order is randomized "
            "per participant (see emotion_task.clip_selection_mode: fixed in "
            "session_config.yaml). Positive/negative groups ranked by HIGHEST "
            "arousal_score (video_data.csv 'arousal_wsd' column, used as a proxy "
            "for strong emotion induction -- NOT the raw 'arousal' column, which "
            "mixes inconsistent units across rows); neutral group ranked by "
            "LOWEST arousal_score (calmest). Regenerate with "
            "scripts/select_fixed_emotion_clips.py [--n-per-group N]."
        ),
    }
    for valence_group, group_clips in groups.items():
        rank_ascending = valence_group == "neutral"
        ranked = sorted(group_clips, key=lambda c: c["arousal_score"], reverse=not rank_ascending)
        top = ranked[: args.n_per_group]
        selection[valence_group] = [
            {
                "clip_id": c["clip_id"], "title": c["title"],
                "valence_score": c["valence_score"], "arousal_score": c["arousal_score"],
            }
            for c in top
        ]
        print(f"{valence_group}: selected {[c['clip_id'] for c in top]} (arousal {[c['arousal_score'] for c in top]})")

    args.output.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
