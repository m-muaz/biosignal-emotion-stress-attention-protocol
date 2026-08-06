"""Subprocess entry point for the web-based emotion video player (Task 2
replacement -- see Engineering_Document.md SS4.2, and app/webui/bridge.py for
why this runs out-of-process from psychopy). Replaces VLC + vlc_rc_player.py.

Launched by app/tasks/emotion_web_player.py, which resolves each clip's
file_path to an absolute file:// URI (and flags any missing file as a
placeholder, same fallback emotion_faced.py had) and writes the whole block
list plus resolved (demo-scaled) durations into a bootstrap JSON file passed
via --bootstrap. Also used, with --practice and a single-clip block list, for
the session shell's (app/webui/shell/) familiarization preview clip.

Usage (normally only invoked by emotion_web_player.py, not run directly):
    python -m app.webui.video_player_app --session-dir sessions/P001_123 \\
        --session-id P001_123 --participant-id P001 --bootstrap C:\\temp\\x.json
"""

import argparse
import sys
from pathlib import Path

import webview

from app.eventlog.event_logger import EventLogger
from app.webui.bridge import QUIT_EXIT_CODE, WebTaskApi, load_json, set_windows_dpi_awareness

FRONTEND_DIR = Path(__file__).resolve().parent / "video_player"


class VideoPlayerApi(WebTaskApi):
    def __init__(self, event_logger: EventLogger, task_label: str, bootstrap: dict):
        super().__init__(event_logger, task_label)
        self._bootstrap = bootstrap

    def get_bootstrap(self) -> dict:
        return self._bootstrap


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--participant-id", required=True)
    parser.add_argument("--bootstrap", required=True, help="Path to the JSON file written by emotion_web_player.py.")
    parser.add_argument("--practice", action="store_true", help="Single practice clip -- logs under task='familiarization'.")
    return parser.parse_args()


def main() -> int:
    set_windows_dpi_awareness()
    args = parse_args()
    bootstrap = load_json(Path(args.bootstrap), None)
    if bootstrap is None:
        print(f"Bootstrap file not found: {args.bootstrap}", file=sys.stderr)
        return 1

    task_label = "familiarization" if args.practice else "emotion"
    event_logger = EventLogger(Path(args.session_dir), args.session_id, args.participant_id)
    api = VideoPlayerApi(event_logger, task_label, bootstrap)

    window = webview.create_window(
        "Emotion Task",
        url=(FRONTEND_DIR / "index.html").resolve().as_uri(),
        js_api=api,
        fullscreen=True,
        confirm_close=False,
        background_color="#000000",
    )
    api.bind_window(window)
    window.events.closing += api.on_os_close

    webview.start()
    event_logger.close()

    return QUIT_EXIT_CODE if api.quit_requested else 0


if __name__ == "__main__":
    raise SystemExit(main())
