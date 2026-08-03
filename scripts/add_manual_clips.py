#!/usr/bin/env python3
"""
Add participant-picked OpenLAV clips to the manifest, including ones with
spoken dialogue (video_data.csv language_free=FALSE) that the original
curated manifest excluded to avoid a language bias -- if you're okay with
that trade-off for specific clips, this adds them explicitly.

Input: a plain text file, one clip per line: `VID_code,valence_group`
(valence_group is one of positive/negative/neutral, e.g. `VID_906,positive`).
Lines starting with # and blank lines are ignored -- see
scripts/manual_clips.txt for a template.

For each valence_group that appears in the input file, this REPLACES that
group's array in emotion_fixed_clips.json with exactly the clips listed (in
listed order) -- groups not mentioned in the input file are left untouched.
Clips already present in emotion_manifest.json are reused as-is (not
duplicated); new ones are appended with a fresh clip_id and metadata pulled
from video_data.csv (title, fine_grained_label=modal_emotion,
duration_sec=length_s, valence_score/arousal_score=valence_wsd/arousal_wsd,
matching scripts/select_fixed_emotion_clips.py's convention).

Usage:
    python scripts/add_manual_clips.py scripts/manual_clips.txt
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "app" / "config" / "emotion_manifest.json"
DEFAULT_METADATA = REPO_ROOT / "assets" / "clips" / "video_data.csv"
DEFAULT_FIXED_CLIPS = REPO_ROOT / "app" / "config" / "emotion_fixed_clips.json"
VALID_VALENCE_GROUPS = {"positive", "negative", "neutral"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("clip_list", type=Path, help="Text file: one `VID_code,valence_group` per line.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--fixed-clips", type=Path, default=DEFAULT_FIXED_CLIPS)
    return parser.parse_args()


def parse_clip_list(path: Path) -> list[tuple[str, str]]:
    entries = []
    for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            sys.exit(f"{path}:{lineno}: expected `VID_code,valence_group`, got: {raw_line!r}")
        video_code, valence_group = parts
        if valence_group not in VALID_VALENCE_GROUPS:
            sys.exit(f"{path}:{lineno}: valence_group must be one of {sorted(VALID_VALENCE_GROUPS)}, got {valence_group!r}")
        entries.append((video_code, valence_group))
    return entries


def main() -> int:
    args = parse_args()
    entries = parse_clip_list(args.clip_list)
    if not entries:
        sys.exit(f"No clip entries found in {args.clip_list}")

    with args.manifest.open("r", encoding="utf-8") as f:
        manifest = json.load(f)
    clips = manifest["clips"]
    clips_by_title = {c["title"]: c for c in clips}

    metadata_by_title = {}
    with args.metadata.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            metadata_by_title[row["video_code"]] = row

    existing_nums = [int(re.search(r"\d+", c["clip_id"]).group()) for c in clips]
    next_num = max(existing_nums, default=0) + 1

    with args.fixed_clips.open("r", encoding="utf-8") as f:
        fixed_selection = json.load(f)

    groups_touched: dict[str, list[dict]] = {}
    for video_code, requested_group in entries:
        clip = clips_by_title.get(video_code)
        if clip is None:
            row = metadata_by_title.get(video_code)
            if row is None:
                sys.exit(f"{video_code} not found in {args.metadata.name} -- check the code is correct.")
            modal_emotion = row["modal_emotion"]
            clip = {
                "clip_id": f"clip_{next_num:02d}",
                "valence_group": requested_group,
                "fine_grained_label": modal_emotion,
                "title": video_code,
                "language": (
                    "none (language_free source clip)" if row["language_free"] == "TRUE"
                    else "contains spoken dialogue/narration (language_free=FALSE) -- included by explicit request"
                ),
                "file_path": f"assets/clips/{modal_emotion}/{video_code}.webm",
                "duration_sec": float(row["length_s"]),
            }
            next_num += 1
            clips.append(clip)
            clips_by_title[video_code] = clip
            print(f"Added new manifest entry {clip['clip_id']} ({video_code}, {modal_emotion} -> {requested_group})")
        else:
            if clip["valence_group"] != requested_group:
                print(
                    f"NOTE: {video_code} is already in the manifest as valence_group="
                    f"{clip['valence_group']!r}; keeping that (not overriding to {requested_group!r})."
                )
            print(f"Reusing existing manifest entry {clip['clip_id']} ({video_code})")

        row = metadata_by_title.get(video_code)
        groups_touched.setdefault(clip["valence_group"], []).append({
            "clip_id": clip["clip_id"],
            "title": clip["title"],
            "valence_score": float(row["valence_wsd"]) if row else None,
            "arousal_score": float(row["arousal_wsd"]) if row else None,
        })

    for valence_group, entries_for_group in groups_touched.items():
        fixed_selection[valence_group] = entries_for_group
        print(f"Set fixed_clips[{valence_group!r}] = {[e['clip_id'] for e in entries_for_group]}")

    with args.manifest.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        f.write("\n")
    with args.fixed_clips.open("w", encoding="utf-8") as f:
        json.dump(fixed_selection, f, indent=2)
        f.write("\n")

    print(f"\nUpdated {args.manifest}")
    print(f"Updated {args.fixed_clips}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
