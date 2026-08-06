#!/usr/bin/env python3
"""Re-download any joy/sadness/indifference OpenLAV clip below 720p at a
higher resolution, from video_data.csv's `source_URL` (the clip's original
YouTube upload -- provenance metadata that ships with the OpenLAV dataset
itself, not guessed).

IMPORTANT CAVEAT (verify before trusting a downloaded file as a drop-in
replacement): video_data.csv has no start/end trim-offset columns, so this
script cannot tell whether an OpenLAV clip *is* the full linked source video
(safe to treat the download as equivalent) or a trimmed excerpt of a longer
video (the download would need the correct trim points, which aren't in this
metadata). To stay safe, this script:
  1. Never overwrites the original file -- downloads go to a staging
     directory (assets/clips/_upgraded/<category>/) for manual review.
  2. Reports each clip's downloaded duration next to video_data.csv's
     `length_s` for that video_code, flagging anything that doesn't match
     within a few seconds as NEEDS_TRIM_CHECK rather than silently treating
     it as a safe replacement.

Requirements: pip install yt-dlp imageio-ffmpeg opencv-python
(imageio-ffmpeg bundles a static ffmpeg binary -- no separate system install
needed; this script locates it automatically.)

Usage:
    python scripts/upgrade_low_res_emotion_clips.py
    python scripts/upgrade_low_res_emotion_clips.py --dry-run
    python scripts/upgrade_low_res_emotion_clips.py --categories joy sadness
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CLIPS_DIR = REPO_ROOT / "assets" / "clips"
STAGING_DIR = CLIPS_DIR / "_upgraded"
METADATA_PATH = CLIPS_DIR / "video_data.csv"
DEFAULT_CATEGORIES = ["joy", "sadness", "indifference"]
MIN_HEIGHT = 720
DURATION_TOLERANCE_SEC = 3.0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--categories", nargs="+", default=DEFAULT_CATEGORIES)
    parser.add_argument("--dry-run", action="store_true", help="List what would be downloaded without downloading.")
    return parser.parse_args()


def load_metadata() -> dict:
    metadata = {}
    with METADATA_PATH.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            metadata[row["video_code"]] = row
    return metadata


def probe_resolution_and_duration(path: Path) -> tuple[int, int, float]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    duration = frame_count / fps if fps else 0.0
    cap.release()
    return width, height, duration


def find_low_res_clips(categories: list) -> list:
    low_res = []
    for category in categories:
        folder = CLIPS_DIR / category
        if not folder.exists():
            print(f"WARNING: no clip folder for category {category!r} at {folder}", file=sys.stderr)
            continue
        for path in sorted(folder.glob("*.webm")):
            video_code = path.stem
            _, height, _ = probe_resolution_and_duration(path)
            if height < MIN_HEIGHT:
                low_res.append({"category": category, "video_code": video_code, "path": path, "height": height})
    return low_res


def find_ffmpeg() -> str:
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


def download_one(video_code: str, source_url: str, dest_dir: Path, ffmpeg_path: str, dry_run: bool) -> Path | None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(dest_dir / f"{video_code}.%(ext)s")
    # Prefer <=1080p, fall back to <=720p, then whatever's best available --
    # a single yt-dlp invocation with a fallback chain, so a video that only
    # has e.g. 480p upstream still downloads instead of erroring out.
    format_selector = (
        "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/"
        "bestvideo[height<=720][ext=mp4]+bestaudio[ext=m4a]/best[height<=720][ext=mp4]/"
        "best"
    )
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--ffmpeg-location", ffmpeg_path,
        "-f", format_selector,
        "--merge-output-format", "mp4",
        "--no-playlist",
        "-o", output_template,
        source_url,
    ]
    if dry_run:
        print("DRY RUN:", " ".join(cmd))
        return None
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR downloading {video_code}: {result.stderr.strip()[-500:]}", file=sys.stderr)
        return None
    matches = list(dest_dir.glob(f"{video_code}.*"))
    return matches[0] if matches else None


def main() -> int:
    args = parse_args()
    metadata = load_metadata()
    ffmpeg_path = find_ffmpeg()
    print(f"Using ffmpeg: {ffmpeg_path}")

    low_res = find_low_res_clips(args.categories)
    print(f"Found {len(low_res)} clip(s) below {MIN_HEIGHT}p across {args.categories}:")
    for item in low_res:
        print(f"  {item['category']}/{item['video_code']} -- currently {item['height']}p")

    report = []
    for item in low_res:
        video_code = item["video_code"]
        row = metadata.get(video_code)
        if row is None:
            print(f"  SKIP {video_code}: no video_data.csv row (can't find source_URL)", file=sys.stderr)
            continue
        source_url = row["source_URL"]
        expected_length_s = float(row["length_s"]) if row.get("length_s") else None

        dest_dir = STAGING_DIR / item["category"]
        print(f"\n{video_code}: downloading from {source_url}")
        downloaded_path = download_one(video_code, source_url, dest_dir, ffmpeg_path, args.dry_run)
        if args.dry_run or downloaded_path is None:
            continue

        width, height, duration = probe_resolution_and_duration(downloaded_path)
        needs_trim_check = (
            expected_length_s is not None and abs(duration - expected_length_s) > DURATION_TOLERANCE_SEC
        )
        entry = {
            "video_code": video_code,
            "category": item["category"],
            "original_height": item["height"],
            "downloaded_path": str(downloaded_path.relative_to(REPO_ROOT)),
            "downloaded_resolution": f"{width}x{height}",
            "downloaded_duration_sec": round(duration, 1),
            "expected_duration_sec": expected_length_s,
            "needs_trim_check": needs_trim_check,
        }
        report.append(entry)
        flag = " -- NEEDS_TRIM_CHECK (duration mismatch)" if needs_trim_check else " -- duration matches, likely safe"
        print(f"  -> {width}x{height}, {duration:.1f}s (expected {expected_length_s}s){flag}")

    if not args.dry_run:
        report_path = STAGING_DIR / "upgrade_report.json"
        STAGING_DIR.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWrote report: {report_path}")
        print(
            f"\n{sum(1 for r in report if not r['needs_trim_check'])} clip(s) look like safe drop-in replacements; "
            f"{sum(1 for r in report if r['needs_trim_check'])} need manual trim-point review before replacing "
            "the original file -- nothing under assets/clips/<category>/ was touched."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
