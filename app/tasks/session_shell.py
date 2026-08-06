"""Session shell: the single persistent pywebview window that drives the
whole session (consent -> questionnaire -> familiarization -> emotion task ->
break -> stress task [raindrop -> highway] -> break -> attention/focus task ->
conclusion), replacing psychopy's visual.Window and app/main.py's Python-side
screen functions entirely -- see app/webui/shell/shell.js for the actual flow.

Task/game experiences (app/webui/raindrop_app.py, app/webui/highway_app.py,
app/webui/video_player_app.py, app/webui/schulte_app.py, app/webui/stroop_app.py)
still run as separate subprocesses (see app/webui/bridge.py for why); this
window just hides itself around each one via ShellApi's run_* bridge
methods, then shows again -- same pattern app/tasks/stress_raindrop.py and
app/tasks/emotion_web_player.py already used for the psychopy window.

The attention/focus task (app/tasks/attention_focus.py: Schulte table +
Stroop test) replaces attention_highway_task's OLD slot here (per PI
discussion 2026-08-04) -- highway was then repurposed and wired in
2026-08-05 as the SECOND HALF of the stress task instead (raindrop 6 min +
highway 4 min, see session_config.yaml's stress_task/attention_highway_task
comments and docs/session_overview_2026-08-05.md), run back-to-back with
raindrop, no separate break in between (the break after stress covers both).

SART (app/tasks/attention_sart.py) is deliberately NOT wired in here at
all -- per PI request 2026-08-05 it stays a fully separate task, run only
via `python -m app.run_task --task sart` (see README's "Attention task
(SART)" section and [[project_attention_task_sart]]).

UserQuit/crashes inside a launched subprocess are caught here (not left to
propagate across the JS bridge call boundary) and turned into a status dict
({"status": "quit"|"crashed"|"ok"}) so the JS shell can decide whether to
keep going, and log accordingly -- app/main.py's old top-level try/except
UserQuit/Exception is now one bridge call's job at a time.
"""

import traceback
from pathlib import Path

import webview

import app.tasks.attention_focus as attention_focus
import app.tasks.attention_highway as attention_highway
import app.tasks.emotion_web_player as emotion_web_player
import app.tasks.stress_raindrop as stress_raindrop
from app.eventlog.event_logger import EventLogger
from app.ui.common_widgets import UserQuit
from app.webui.bridge import WebTaskApi

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "webui" / "shell"


class ShellApi(WebTaskApi):
    def __init__(
        self, event_logger: EventLogger, ctx, config: dict, blocks: list, familiarization_clip: dict,
        attention_focus_trials: list, skip_questionnaire: bool, skip_familiarization: bool,
        breaks_after_task1_sec: float, breaks_after_task2_sec: float,
    ):
        super().__init__(event_logger, task_label=None)
        self._ctx = ctx
        self._blocks = blocks
        self._familiarization_clip = familiarization_clip
        self._attention_focus_trials = attention_focus_trials
        self._skip_questionnaire = skip_questionnaire
        self._skip_familiarization = skip_familiarization
        self._breaks_after_task1_sec = breaks_after_task1_sec
        self._breaks_after_task2_sec = breaks_after_task2_sec
        self._questionnaire_items = config.get("questionnaire", {}).get("items", [])
        self._rating_scale_min = config["emotion_task"]["rating_scale_min"]
        self._rating_scale_max = config["emotion_task"]["rating_scale_max"]

    def get_bootstrap(self) -> dict:
        return {
            "skip_questionnaire": self._skip_questionnaire,
            "skip_familiarization": self._skip_familiarization,
            "questionnaire_items": self._questionnaire_items,
            "rating_scale_min": self._rating_scale_min,
            "rating_scale_max": self._rating_scale_max,
            "break_duration_sec": self._breaks_after_task1_sec,
            "break_duration_after_task2_sec": self._breaks_after_task2_sec,
            # Which game (schulte/stroop) each attention_focus trial slot
            # is -- the actual per-trial content lives server-side in
            # self._attention_focus_trials; the shell only needs the game
            # name up front so it can show the right per-trial instructions
            # before calling run_attention_focus_trial(i) (see shell.js).
            "attention_focus_sequence": [t["game"] for t in self._attention_focus_trials],
        }

    def _run_guarded(self, fn) -> dict:
        """Runs a subprocess-launching call, translating UserQuit/crashes
        into a status the JS shell can act on instead of letting them
        propagate across the JS<->Python bridge boundary."""
        try:
            fn()
            return {"status": "ok"}
        except UserQuit:
            return {"status": "quit"}
        except Exception:
            self._event_logger.log("session_crashed", task=None, traceback=traceback.format_exc())
            return {"status": "crashed"}

    def run_practice_video(self) -> dict:
        return self._run_guarded(
            lambda: emotion_web_player.play_practice_clip(self._window, self._ctx, self._familiarization_clip)
        )

    def run_practice_raindrop_demo(self) -> dict:
        return self._run_guarded(lambda: stress_raindrop.run_demo(self._window, self._ctx))

    def run_practice_highway_demo(self) -> dict:
        return self._run_guarded(lambda: attention_highway.run_demo(self._window, self._ctx))

    def run_emotion_task(self) -> dict:
        return self._run_guarded(
            lambda: emotion_web_player.run_emotion_task(self._window, self._ctx, self._blocks)
        )

    def run_stress_task(self) -> dict:
        return self._run_guarded(lambda: stress_raindrop.run_stress_task(self._window, self._ctx))

    def run_highway_task(self) -> dict:
        # 2nd half of "stress" (raindrop 6 min + highway 4 min) per PI
        # request 2026-08-05 -- see session_config.yaml's stress_task/
        # attention_highway_task comments and docs/session_overview_2026-08-05.md.
        return self._run_guarded(lambda: attention_highway.run_attention_highway_task(self._window, self._ctx))

    def run_practice_schulte_demo(self) -> dict:
        return self._run_guarded(lambda: attention_focus.run_demo_schulte(self._window, self._ctx))

    def run_practice_stroop_demo(self) -> dict:
        return self._run_guarded(lambda: attention_focus.run_demo_stroop(self._window, self._ctx))

    def run_attention_focus_trial(self, trial_index: int) -> dict:
        trial_spec = self._attention_focus_trials[trial_index]
        return self._run_guarded(lambda: attention_focus.run_trial(self._window, self._ctx, trial_spec))


def run_session(
    ctx, config: dict, clips: list, blocks: list, skip_questionnaire: bool, skip_familiarization: bool,
) -> None:
    familiarization_clip = emotion_web_player.get_familiarization_clip(clips, config["emotion_task"])
    breaks_after_task1_sec = ctx.scaled(config["breaks"]["after_task1_min"] * 60)
    breaks_after_task2_sec = ctx.scaled(config["breaks"]["after_task2_min"] * 60)
    # Built once, up front -- see app/tasks/attention_focus.py's
    # build_sequence() docstring for why (same provenance reasoning as
    # blocks/familiarization_clip above).
    attention_focus_trials = attention_focus.build_sequence(ctx)

    api = ShellApi(
        ctx.event_logger, ctx, config, blocks, familiarization_clip, attention_focus_trials,
        skip_questionnaire, skip_familiarization, breaks_after_task1_sec, breaks_after_task2_sec,
    )
    window = webview.create_window(
        "Data Collection Session",
        url=(FRONTEND_DIR / "index.html").resolve().as_uri(),
        js_api=api,
        fullscreen=True,
        confirm_close=False,
        background_color="#163247",
    )
    api.bind_window(window)
    window.events.closing += api.on_os_close
    webview.start()
