"""Subprocess entry point for the highway dodge attention game (added per PI
discussion 2026-08-04 -- see app/tasks/attention_highway.py for why this
task exists alongside SART, and app/webui/bridge.py for why it runs
out-of-process from psychopy).

Launched by app/tasks/attention_highway.py, which resolves the (demo-scaled)
block durations and hands them to this process as a bootstrap JSON file
passed via --bootstrap. This process only renders the game, drives its own
event loop, and logs -- unlike stress_raindrop.py it doesn't need to
pre-generate anything from the session rng, since hazard lane/timing here is
never analyzed as a controlled stimulus set the way raindrop's questions are.

Usage (normally only invoked by attention_highway.py, not run directly):
    python -m app.webui.highway_app --session-dir sessions/P001_123 \\
        --session-id P001_123 --participant-id P001 --bootstrap C:\\temp\\x.json
"""

import argparse
import sys
from pathlib import Path

import webview

from app.eventlog.event_logger import EventLogger
from app.webui.bridge import QUIT_EXIT_CODE, WebTaskApi, load_json, set_windows_dpi_awareness

FRONTEND_DIR = Path(__file__).resolve().parent / "highway"


class HighwayApi(WebTaskApi):
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
    parser.add_argument("--bootstrap", required=True, help="Path to the JSON file written by attention_highway.py.")
    parser.add_argument("--practice", action="store_true", help="Short familiarization round -- not part of the scored dataset.")
    return parser.parse_args()


def main() -> int:
    set_windows_dpi_awareness()
    args = parse_args()
    bootstrap = load_json(Path(args.bootstrap), None)
    if bootstrap is None:
        print(f"Bootstrap file not found: {args.bootstrap}", file=sys.stderr)
        return 1

    task_label = "familiarization" if args.practice else "attention_highway"
    event_logger = EventLogger(Path(args.session_dir), args.session_id, args.participant_id)
    api = HighwayApi(event_logger, task_label, bootstrap)

    window = webview.create_window(
        "Highway Dodge",
        url=(FRONTEND_DIR / "index.html").resolve().as_uri(),
        js_api=api,
        fullscreen=True,
        confirm_close=False,
        background_color="#163247",
    )
    api.bind_window(window)
    # OS close-button (titlebar X) counts as a quit, same as Escape --
    # mirrors app/main.py's _install_close_handler for the psychopy window.
    window.events.closing += api.on_os_close

    webview.start()
    event_logger.close()

    return QUIT_EXIT_CODE if api.quit_requested else 0


if __name__ == "__main__":
    raise SystemExit(main())
