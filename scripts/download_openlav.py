#!/usr/bin/env python3
"""
Download the OpenLAV videos from PsychArchives and organize them by the
dataset's modal_emotion label.

Output example:
OpenLAV/
├── anxiety/
│   └── VID_104.webm
├── fear/
│   └── VID_103.webm
├── joy/
├── surprise/
└── video_data.csv

Requirements:
    python -m pip install requests

Usage:
    python download_openlav.py
    python download_openlav.py --output D:\Datasets\OpenLAV
    python download_openlav.py --dry-run
    python download_openlav.py --emotions joy fear surprise
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from pathlib import Path

import requests


BASE_URL = "https://psycharchives.org"

# PsychArchives item containing the actual OpenLAV moving-image files.
VIDEO_ITEM_UUID = "18779e98-c04b-4299-8311-dc442dc89bcd"
ITEM_PAGE_URL = f"{BASE_URL}/en/item/{VIDEO_ITEM_UUID}"

# Official OpenLAV video-level metadata file.
METADATA_URL = (
    "https://pada.psycharchives.org/bitstream/"
    "dc95bddf-2b5c-48ff-916f-89de10694355"
)

VID_PATTERN = re.compile(r"(VID[_-]?\d+)", re.IGNORECASE)

# The item page is server-rendered HTML (no working REST API is exposed for
# this item), so each video's filename and download link are scraped from
# the "<strong>VID_102.webm</strong> ... bitstream-download href=..." markup.
BITSTREAM_ENTRY_PATTERN = re.compile(
    r'<strong>\s*(?P<name>VID[_-]?\d+\.\w+)\s*</strong>.*?'
    r'bitstream-download"\s+href="(?P<url>https://pada\.psycharchives\.org/'
    r'bitstream/[0-9a-fA-F-]{36})"',
    re.DOTALL,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download OpenLAV videos into modal-emotion directories."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("OpenLAV"),
        help="Output directory. Default: ./OpenLAV",
    )
    parser.add_argument(
        "--emotions",
        nargs="+",
        default=None,
        help=(
            "Only download selected modal emotions, for example: "
            "--emotions joy fear surprise"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading video files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload files that already exist.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=90,
        help="HTTP timeout in seconds. Default: 90",
    )
    return parser.parse_args()


def safe_folder_name(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^\w.-]+", "_", value)
    return value.strip("._") or "unlabelled"


def normalize_video_code(value: str) -> str:
    match = VID_PATTERN.search(value or "")
    if not match:
        return ""

    digits = re.search(r"\d+", match.group(1))
    if not digits:
        return ""

    return f"VID_{int(digits.group(0))}"


def download_metadata(
    session: requests.Session,
    output_path: Path,
    timeout: int,
) -> None:
    if output_path.exists() and output_path.stat().st_size > 0:
        return

    print(f"Downloading metadata -> {output_path}")
    response = session.get(METADATA_URL, timeout=timeout)
    response.raise_for_status()
    output_path.write_bytes(response.content)


def load_emotion_labels(metadata_path: Path) -> dict[str, str]:
    labels: dict[str, str] = {}

    with metadata_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)

        required = {"video_code", "modal_emotion"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(
                "Metadata is missing required column(s): "
                + ", ".join(sorted(missing))
            )

        for row in reader:
            code = normalize_video_code(row.get("video_code", ""))
            emotion = safe_folder_name(row.get("modal_emotion", ""))

            if code:
                labels[code] = emotion

    if not labels:
        raise RuntimeError("No OpenLAV video labels were found in the metadata.")

    return labels


def fetch_video_bitstreams(
    session: requests.Session,
    timeout: int,
) -> dict[str, tuple[str, str]]:
    """Scrape the OpenLAV item page for {video_code: (filename, url)}."""
    response = session.get(ITEM_PAGE_URL, timeout=timeout)
    response.raise_for_status()

    entries: dict[str, tuple[str, str]] = {}
    for match in BITSTREAM_ENTRY_PATTERN.finditer(response.text):
        filename = match.group("name")
        code = normalize_video_code(filename)
        if code:
            entries[code] = (filename, match.group("url"))

    if not entries:
        raise RuntimeError(
            "Could not find any video download links on the OpenLAV item "
            "page. PsychArchives may have changed its page layout."
        )

    return entries


def stream_download(
    session: requests.Session,
    url: str,
    destination: Path,
    timeout: int,
) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")

    with session.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()

        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)

    partial.replace(destination)


def main() -> int:
    args = parse_args()
    output_dir: Path = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_emotions = (
        {safe_folder_name(x) for x in args.emotions}
        if args.emotions
        else None
    )

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "OpenLAV-research-downloader/1.0 "
                "(academic dataset download)"
            ),
            "Accept": "application/hal+json, application/json;q=0.9, */*;q=0.8",
        }
    )

    metadata_path = output_dir / "video_data.csv"

    try:
        download_metadata(session, metadata_path, args.timeout)
        emotion_by_code = load_emotion_labels(metadata_path)

        print("Reading the official OpenLAV video repository...")
        bitstreams_by_code = fetch_video_bitstreams(session, args.timeout)

        video_entries: list[tuple[str, str, str, str]] = []
        ignored = 0

        for code, (filename, url) in bitstreams_by_code.items():
            if code not in emotion_by_code:
                ignored += 1
                continue

            emotion = emotion_by_code[code]
            if selected_emotions and emotion not in selected_emotions:
                continue

            video_entries.append((code, emotion, filename, url))

        video_entries.sort(key=lambda item: int(item[0].split("_")[1]))

        if not video_entries:
            raise RuntimeError(
                "No repository video filenames could be matched to the "
                "video_code values in video_data.csv. PsychArchives may have "
                "changed its page layout or filenames."
            )

        print(
            f"Matched {len(video_entries)} video files. "
            f"Ignored {ignored} video files not present in video_data.csv."
        )

        downloaded = 0
        skipped = 0
        failed: list[tuple[str, str]] = []

        for index, (code, emotion, filename, url) in enumerate(
            video_entries, start=1
        ):
            emotion_dir = output_dir / emotion
            emotion_dir.mkdir(parents=True, exist_ok=True)

            destination = emotion_dir / filename

            prefix = f"[{index:03d}/{len(video_entries):03d}]"

            if destination.exists() and not args.overwrite:
                print(f"{prefix} Skip existing: {destination}")
                skipped += 1
                continue

            print(f"{prefix} {code} -> {emotion}/{filename}")

            if args.dry_run:
                continue

            try:
                stream_download(session, url, destination, args.timeout)
                downloaded += 1
                time.sleep(0.15)
            except Exception as exc:
                failed.append((code, str(exc)))
                print(f"    ERROR: {exc}", file=sys.stderr)

        print("\nFinished.")
        print(f"Downloaded: {downloaded}")
        print(f"Skipped:    {skipped}")
        print(f"Failed:     {len(failed)}")
        print(f"Location:   {output_dir}")

        if failed:
            failure_log = output_dir / "failed_downloads.csv"
            with failure_log.open(
                "w", encoding="utf-8", newline=""
            ) as handle:
                writer = csv.writer(handle)
                writer.writerow(["video_code", "error"])
                writer.writerows(failed)

            print(f"Failure log: {failure_log}")
            return 2

        return 0

    except requests.HTTPError as exc:
        print(f"HTTP error: {exc}", file=sys.stderr)
        return 1
    except requests.RequestException as exc:
        print(f"Network error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
