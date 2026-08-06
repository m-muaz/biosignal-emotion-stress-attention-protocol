"""Shared JS<->Python bridge for the web-based task subprocesses.

app/webui/raindrop_app.py and app/webui/video_player_app.py each run as their
OWN subprocess (launched by app/tasks/stress_raindrop.py /
app/tasks/emotion_web_player.py) -- never imported into the same process as
psychopy, so pywebview's blocking webview.start() event loop never has to
coexist with pyglet's window/event loop. Each subprocess constructs its own
EventLogger pointed at the SAME session_dir/events.jsonl the parent session
already has open; appending from a second process while the parent's handle
sits open (not writing, since it's blocked on subprocess.run()) is safe on
Windows -- verified empirically before building this.

QUIT_EXIT_CODE is how a subprocess tells its parent "the participant pressed
Escape" -- the parent (app/tasks/stress_raindrop.py /
app/tasks/emotion_web_player.py) turns that back into UserQuit, the same
control flow app/ui/common_widgets.py's request_quit()/check_quit() already
gives every PsychoPy-rendered task.
"""

import json
import time
from pathlib import Path

from app.eventlog.event_logger import EventLogger

QUIT_EXIT_CODE = 3


def set_windows_dpi_awareness() -> None:
    """Declare this process per-monitor-DPI-aware, before any window exists.

    Without this, Windows treats the process as DPI-unaware and silently
    bitmap-stretches the whole window through the DWM compositor on any
    display running above 100% scaling -- for a video/game window this shows
    up as playback looking soft/blurry ("wrong resolution"), even though the
    source file and CSS are both correct. Must run before the first window is
    created; no-ops on non-Windows platforms or if the API isn't available.

    app/main.py calls this once for the session-shell process, but DPI
    awareness does NOT inherit across subprocess.run() -- every *_app.py
    entry point below (video_player_app.py, raindrop_app.py, highway_app.py,
    schulte_app.py, stroop_app.py) launches as its own fresh process (see
    this module's docstring), so each must call this again itself before its
    own webview.create_window().
    """
    import platform

    if platform.system() != "Windows":
        return
    import ctypes

    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)  # PROCESS_PER_MONITOR_DPI_AWARE
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass

# Event payload keys that map onto EventLogger.log's own named parameters
# rather than being nested inside event_payload_json -- keeps the JS side's
# event rows shaped the same as every other task's rows in events.jsonl.
_LOGGER_FIELD_NAMES = ("block_index", "trial_index", "condition_label")


class WebTaskApi:
    """Base class for window.pywebview.api. Subclasses (RaindropApi,
    VideoPlayerApi) add task-specific methods; both share quit/log handling.
    """

    def __init__(self, event_logger: EventLogger, task_label: str | None):
        self._event_logger = event_logger
        self._task_label = task_label
        self.quit_requested = False
        self._window = None
        # On Windows, window.destroy() (winforms backend) calls Form.Close(),
        # which fires the SAME FormClosing -> events.closing hook a real
        # titlebar-X click does (verified against webview/platforms/
        # winforms.py: destroy_window() literally calls i.Close()) -- so
        # on_os_close() can't just unconditionally mean "the participant
        # quit". This flag records that WE initiated the close (via
        # request_quit()/close_window()) so on_os_close() can tell a real,
        # unsolicited OS close from our own normal-completion teardown.
        self._closing_intentionally = False

    def bind_window(self, window) -> None:
        """Entry points call this right after webview.create_window() so
        request_quit()/close_window() can close the window from a JS call."""
        self._window = window

    def log_event(self, event_type: str, fields: dict | None = None, task: str | None = None) -> dict:
        """`task` overrides self._task_label for callers that span several
        task labels in one process (app/webui/shell/ -- preparation ->
        familiarization -> emotion -> stress -> debrief); raindrop_app.py/
        video_player_app.py just omit it and use their one fixed label."""
        fields = dict(fields or {})
        logger_kwargs = {name: fields.pop(name, None) for name in _LOGGER_FIELD_NAMES}
        self._event_logger.log(event_type, task=task or self._task_label, **logger_kwargs, **fields)
        return {"ok": True}

    def sync_checkpoint(self, client_perf_now_ms: float) -> dict:
        """Clock-correspondence probe -- called from JS with its own
        performance.now() reading; returns the SAME two Python-side clocks
        EventLogger.log() stamps every row with, captured at the moment this
        call is received. JS wraps this in a round trip (its own
        performance.now() again after the await resolves) and logs the
        whole set -- see highway/game.js's runSyncCheckpoints() -- so
        offline processing gets: the browser's clock reading, the host's
        wall clock and monotonic clock, and a round-trip bound on the IPC
        latency between them, all from one probe. Doesn't log anything
        itself; the caller logs a sync_checkpoint event with this result
        plus its own before/after readings, exactly like every other event.
        Call this several times at session start AND end (not just once) so
        offline analysis can estimate drift, not just a single offset.
        """
        return {
            "host_utc": time.time(),
            "host_perf_counter": time.perf_counter(),
            "client_perf_now_ms": client_perf_now_ms,
        }

    def request_quit(self) -> dict:
        """Called from JS (Escape key) to close the window ourselves."""
        self.quit_requested = True
        self._closing_intentionally = True
        if self._window is not None:
            self._window.destroy()
        return {"ok": True}

    def on_os_close(self) -> None:
        """Bind to window.events.closing -- fires both for a real titlebar-X
        click AND for our own destroy() calls (see _closing_intentionally's
        comment). Only counts as a participant-initiated quit if neither
        request_quit() nor close_window() got there first. Must return
        None/False either way, since events.closing cancels the close if any
        handler returns a truthy value."""
        if not self._closing_intentionally:
            self.quit_requested = True

    def close_window(self) -> dict:
        """Called from JS when the task finishes normally (not a quit)."""
        self._closing_intentionally = True
        if self._window is not None:
            self._window.destroy()
        return {"ok": True}


def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json_atomic(path: Path, data) -> None:
    """Write-to-temp-then-replace so a crash mid-write never leaves the
    cross-participant high-score file truncated/corrupt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)
