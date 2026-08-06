"""Subprocess entry point for the raindrop stress game (Task 3 replacement --
see Engineering_Document.md SS4.3, and app/webui/bridge.py for why this runs
out-of-process from psychopy).

Launched by app/tasks/stress_raindrop.py, which pre-generates every tier's
question queue (via app.tasks.stress_mat.generate_question, using the
session's own rng -- so provenance stays with the one rng_seed recorded in
session_manifest.json, same as everything else in the session) and writes it
plus the resolved (demo-scaled) durations into a bootstrap JSON file passed
via --bootstrap. This process only renders the game, drives its own event
loop, and logs -- it never touches randomness itself.

Usage (normally only invoked by stress_raindrop.py, not run directly):
    python -m app.webui.raindrop_app --session-dir sessions/P001_123 \\
        --session-id P001_123 --participant-id P001 --bootstrap C:\\temp\\x.json
"""

import argparse
import sys
from pathlib import Path

import webview

from app.eventlog.event_logger import EventLogger
from app.webui.bridge import QUIT_EXIT_CODE, WebTaskApi, load_json, save_json_atomic, set_windows_dpi_awareness

FRONTEND_DIR = Path(__file__).resolve().parent / "raindrop"


class RaindropApi(WebTaskApi):
    def __init__(self, event_logger: EventLogger, task_label: str, bootstrap: dict, participant_id: str, practice: bool):
        super().__init__(event_logger, task_label)
        self._bootstrap = bootstrap
        self._participant_id = participant_id
        self._practice = practice
        self._high_score_path = bootstrap.get("high_score_path")

    def get_bootstrap(self) -> dict:
        return self._bootstrap

    def get_high_score(self):
        """Current best score across every participant, or None if this is a
        practice run (familiarization) or no high score file yet."""
        if self._practice or not self._high_score_path:
            return None
        return load_json(Path(self._high_score_path), None)

    def submit_score(self, score: int):
        """Called once, when the game ends (all tiers complete or quit).
        Practice runs never touch the persistent high-score file -- a
        familiarization round shouldn't let a lucky guess become the
        study's all-time high score."""
        if self._practice or not self._high_score_path:
            return {"high_score": None}
        path = Path(self._high_score_path)
        current = load_json(path, None)
        if current is None or score > current.get("score", -1):
            current = {"participant_id": self._participant_id, "score": score}
            save_json_atomic(path, current)
        return {"high_score": current}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-dir", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--participant-id", required=True)
    parser.add_argument("--bootstrap", required=True, help="Path to the JSON file written by stress_raindrop.py.")
    parser.add_argument("--practice", action="store_true", help="Short familiarization round -- not written to the high-score file.")
    return parser.parse_args()


def main() -> int:
    set_windows_dpi_awareness()
    args = parse_args()
    bootstrap = load_json(Path(args.bootstrap), None)
    if bootstrap is None:
        print(f"Bootstrap file not found: {args.bootstrap}", file=sys.stderr)
        return 1

    task_label = "familiarization" if args.practice else "stress"
    event_logger = EventLogger(Path(args.session_dir), args.session_id, args.participant_id)
    api = RaindropApi(event_logger, task_label, bootstrap, args.participant_id, args.practice)

    window = webview.create_window(
        "Raindrop Math",
        url=(FRONTEND_DIR / "index.html").resolve().as_uri(),
        js_api=api,
        fullscreen=True,
        confirm_close=False,
        background_color="#1b3a4b",
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
