#!/usr/bin/env python3
"""
Post-hoc check: was the emotion task's video actually played at its native
resolution, undistorted, for a given session?

Reads events.jsonl's `clip_video_metadata` rows (logged by
app/webui/video_player/player.js's onLoadedMetadata handler, once per
real -- non-placeholder -- clip played) and prints, per clip:
  - native resolution (video.videoWidth/Height -- read straight from the
    file by the browser's decoder, so this is ground truth)
  - object_fit: the actual computed CSS object-fit value applied to
    #clip-video at playback time. Should always be "contain" (uniform
    scale + letterbox, no distortion) -- anything else (e.g. "fill", which
    WOULD stretch the video to fill its box regardless of aspect ratio) is
    a real bug. NOTE: this deliberately does NOT compare the <video>
    element's own bounding-box aspect ratio against the native aspect
    ratio -- with object-fit: contain, the element's box is always
    ~full-screen and is *meant* to have a different aspect ratio than its
    letterboxed content, so that comparison would flag perfectly normal
    non-16:9 clips (e.g. VID_603, 1280x676) as "stretched" when they
    aren't -- an early version of this script/check had exactly that bug.
  - whether the clip is below the 720p floor (see
    scripts/emotion_clip_picks_2026-08-04.txt's "<-- below 720p" notes --
    this is a property of the source file, not a playback bug, but worth
    surfacing per-session rather than only in the picks-file comments)

Usage:
    python scripts/check_video_playback_stats.py sessions/P001_1234567890
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session_dir", type=Path, help="A session folder containing events.jsonl.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    events_path = args.session_dir / "events.jsonl"
    if not events_path.exists():
        sys.exit(f"No events.jsonl found at {events_path}")

    rows = []
    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if row["event_type"] == "clip_video_metadata":
                rows.append(row["event_payload_json"])

    if not rows:
        print("No clip_video_metadata events found -- either no clips played, all were "
              "placeholders (file missing on disk), or this session predates the fix that "
              "added this logging.")
        return 1

    print(f"{'clip_id':<12} {'native':<12} {'object-fit':<12} {'below 720p'}")
    bad_fit = []
    below_720p = []
    for r in rows:
        native = f"{r['native_width']}x{r['native_height']}"
        object_fit = r["object_fit"]
        flag = "yes" if r["below_720p"] else ""
        marker = "" if object_fit == "contain" else "  <-- NOT contain!"
        print(f"{r['clip_id']:<12} {native:<12} {object_fit:<12} {flag}{marker}")
        if object_fit != "contain":
            bad_fit.append(r["clip_id"])
        if r["below_720p"]:
            below_720p.append(r["clip_id"])

    print(f"\n{len(rows)} clips checked.")
    if bad_fit:
        print(f"WARNING: {len(bad_fit)} clip(s) were NOT rendered with object-fit: contain "
              f"(stretching/distortion is possible): {bad_fit}")
    else:
        print("All clips rendered with object-fit: contain -- no stretching/distortion.")
    if below_720p:
        print(f"NOTE: {len(below_720p)} clip(s) were natively below 720p (source file limitation, "
              f"not a playback bug): {below_720p}")

    return 1 if bad_fit else 0


if __name__ == "__main__":
    raise SystemExit(main())
