"""Session runner. Orchestrates the emotion+stress portion of the protocol:
consent -> questionnaire -> device sync -> familiarization -> emotion task ->
break -> stress task -> conclusion.

The attention/focus task (OpenMATB) is deliberately NOT orchestrated here --
it's a separate pyglet application run as its own standalone process (its own
window/event loop, its own isolated venv -- see README.md "Attention task
(OpenMATB)"), launched after this script's session ends rather than embedded
in this flow. `app/tasks/attention_openmatb.py` and `app/run_task.py --task
attention` remain available for testing that integration in isolation.

Usage:
    python -m app.main --participant-id P001
    python -m app.main --participant-id DEMO001 --demo-scale 0.2 --skip-questionnaire

--demo-scale shortens passive/macro block lengths (baseline duration, rest,
cue, fixation, video length, number of trials per block) so the whole
protocol can be walked through quickly. It deliberately does NOT shorten
interactive per-event timing (stress per-question time limit) -- scaling
that would make the task impossible for a human to actually try.
"""

import argparse
import random
import time
from pathlib import Path

from psychopy import visual

from app.config.loader import load_config, load_emotion_manifest
from app.context import SessionContext
from app.eventlog.event_logger import EventLogger
from app.eventlog.session_manifest import write_session_manifest
from app.sync.device_sync import sync_all_blocking
from app.tasks.emotion_faced import (
    EMOTION_OPTIONS,
    EMOTION_PROMPT,
    ExternalPlayerNotFound,
    pick_practice_clip,
    play_clip,
    resolve_external_player,
    run_emotion_task,
    select_task_clips,
)
from app.tasks.stress_mat import generate_question, get_numeric_answer, run_stress_task
from app.ui.common_widgets import (
    BG_COLOR,
    UserQuit,
    discrete_choice,
    numeric_entry,
    rating_scale_0_7,
    show_message,
    show_title_screen,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Minimal intro splash -- replaces a full informed-consent paragraph that
# overflowed the window at a readable font size. The full consent language
# (IRB/PI review pending, see Engineering_Document.md §3.3) belongs in a
# proper printed/e-signed consent form before real data collection, not
# packed into a single fixed-size on-screen TextStim.
INTRO_HEADING = "DATA COLLECTION SESSION"
INTRO_SUBTEXT = "Emotion task, then a stress task. Press SPACE to begin."


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Emotion/Stress/Attention collection session.")
    parser.add_argument("--participant-id", required=True)
    parser.add_argument("--config", default=str(REPO_ROOT / "app" / "config" / "session_config.yaml"))
    parser.add_argument("--demo-scale", type=float, default=1.0, help="Multiplier for passive/macro durations; 1.0 = real timing.")
    parser.add_argument("--devices-mode", choices=["mock", "real"], default=None, help="Override devices.mode from config.")
    parser.add_argument("--fullscreen", action="store_true")
    parser.add_argument("--skip-questionnaire", action="store_true", help="Skip consent/questionnaire screens (quick demo runs).")
    parser.add_argument("--skip-familiarization", action="store_true")
    return parser.parse_args()


def run_consent_and_questionnaire(win, ctx, cfg):
    ctx.event_logger.log("phase_start", task="preparation")
    show_title_screen(win, INTRO_HEADING, INTRO_SUBTEXT)
    ctx.event_logger.log("consent_given", task="preparation")

    responses = {}
    for item in cfg.get("questionnaire", {}).get("items", []):
        item_id, item_type, prompt = item["id"], item["type"], item["prompt"]
        if item_type == "choice":
            value, rt = discrete_choice(win, prompt, item["options"])
        elif item_type == "numeric":
            value, rt = numeric_entry(win, prompt, allow_decimal=True)
        elif item_type == "rating":
            value, rt = rating_scale_0_7(win, prompt, cfg["emotion_task"]["rating_scale_min"], cfg["emotion_task"]["rating_scale_max"])
        else:
            continue
        responses[item_id] = value
        ctx.event_logger.log("questionnaire_response", task="preparation", item_id=item_id, value=value, rt=rt)

    return responses


def run_familiarization(win, ctx, cfg, clips, selected_by_group, player_path):
    """Returns the (possibly new) Window -- play_clip() closes/reopens our
    window around the practice clip's playback, so the caller must reassign
    its `win` reference to the return value."""
    ctx.event_logger.log("phase_start", task="familiarization")
    show_message(
        win,
        "FAMILIARIZATION\n\n"
        "Let's quickly walk through what each part of today's session looks like.\n"
        "Nothing in this section is scored.\n\n"
        "Press SPACE to continue.",
        wait_key=["space", "return"],
    )

    show_message(
        win,
        "Comfort check: make sure your headphones and sensors feel secure, and you can see the screen clearly.\n\n"
        "Press SPACE when ready.",
        wait_key=["space", "return"],
    )

    # Practice clip is drawn from clips NOT selected for the real task (see
    # pick_practice_clip) -- a genuinely new clip, not a preview of one the
    # participant will rate for real a few minutes from now.
    practice_clip = pick_practice_clip(clips, selected_by_group, ctx.rng)
    show_message(
        win,
        "Preview: EMOTION task. You'll watch a short clip in its own player window, then answer "
        "a couple of quick questions about how it made you feel, like this.\n\n"
        "Press SPACE to play the practice clip.",
        wait_key=["space", "return"],
    )
    win = play_clip(win, practice_clip, ctx.scaled(practice_clip.get("duration_sec", 90)), ctx=ctx, player_path=player_path, task_label="familiarization")
    discrete_choice(win, EMOTION_PROMPT, EMOTION_OPTIONS)
    rating_scale_0_7(
        win, "Rate your VALENCE (example only -- not a real clip).",
        cfg["emotion_task"]["rating_scale_min"], cfg["emotion_task"]["rating_scale_max"],
    )

    show_message(
        win,
        "Preview: STRESS task. You'll solve arithmetic problems against a countdown timer, with a leaderboard shown.\n"
        "Solve LEFT TO RIGHT, no order of operations.\n\n"
        "Press SPACE to try one example.",
        wait_key=["space", "return"],
    )
    sample_tier = cfg["stress_task"]["tiers"][0]
    expr_str, _ = generate_question(sample_tier, ctx.rng)
    get_numeric_answer(
        win, expr_str, "example", sample_tier["time_per_question_sec"],
        cfg["stress_task"]["leaderboard_names"], cfg["stress_task"]["leaderboard_scores"],
        cfg["stress_task"]["hurry_up_threshold_fraction"],
    )

    show_message(win, "That's everything. The real tasks begin now.\n\nPress SPACE to start.", wait_key=["space", "return"])
    ctx.event_logger.log("phase_end", task="familiarization")
    return win


def _raise_window_focus():
    """Best-effort: bring the just-created window to the front on macOS.

    A GUI window launched from Terminal doesn't always become the frontmost
    app automatically, which looks exactly like a hang -- the process is
    correctly blocked on a keypress, but keystrokes are still going to
    Terminal. Silently no-ops on any failure (non-macOS, no permission, etc.)
    since this is a UX nicety, not something the session should ever fail on.
    """
    import platform
    import subprocess

    if platform.system() != "Darwin":
        return
    try:
        pid = str(__import__("os").getpid())
        script = (
            'tell application "System Events" to set frontmost of '
            f'(first process whose unix id is {pid}) to true'
        )
        subprocess.run(["osascript", "-e", script], timeout=3, capture_output=True)
    except Exception:
        pass


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.devices_mode:
        config["devices"]["mode"] = args.devices_mode

    session_id = f"{args.participant_id}_{int(time.time())}"
    session_dir = REPO_ROOT / config["logging"]["session_root"] / session_id

    event_logger = EventLogger(session_dir, session_id, args.participant_id)
    rng_seed = time.time()
    rng = random.Random(rng_seed)
    window_kwargs = dict(size=(1280, 800), color=BG_COLOR, units="height", fullscr=args.fullscreen)
    ctx = SessionContext(
        config=config, event_logger=event_logger, participant_id=args.participant_id, demo_scale=args.demo_scale,
        rng=rng, window_kwargs=window_kwargs,
    )

    win = visual.Window(**window_kwargs)
    _raise_window_focus()

    try:
        print(f"Syncing devices (mode={config['devices']['mode']})...")
        sync_records = sync_all_blocking(config["devices"], config["devices"]["mode"])
        for role, records in sync_records.items():
            event_logger.log("device_sync", task="preparation", **records[0])
            if records[0]["result"] != "ok":
                print(f"WARNING: sync failed for device '{role}': {records[0]['detail']}")
        write_session_manifest(session_dir, session_id, args.participant_id, config, sync_records, args.demo_scale, rng_seed)

        try:
            player_path = resolve_external_player(config["emotion_task"])
        except ExternalPlayerNotFound as exc:
            show_message(win, f"SETUP ERROR\n\n{exc}\n\nPress SPACE to abort.", wait_key=["space", "return"])
            raise
        clips = load_emotion_manifest(str(REPO_ROOT / config["emotion_task"]["manifest_path"]))
        block_order, selected_by_group = select_task_clips(clips, config["emotion_task"], ctx.rng)

        if not args.skip_questionnaire:
            run_consent_and_questionnaire(win, ctx, config)
        if not args.skip_familiarization:
            win = run_familiarization(win, ctx, config, clips, selected_by_group, player_path)

        win = run_emotion_task(win, ctx, block_order, selected_by_group, player_path)

        show_message(win, f"Break -- take a moment to relax.", duration=ctx.scaled(config["breaks"]["after_task1_min"] * 60))

        run_stress_task(win, ctx)

        show_message(
            win,
            "CONCLUSION\n\nThat concludes the emotion and stress portion of the session. Thank you!\n\n"
            "Next: the attention/focus task (OpenMATB) runs separately as its own program.\n\n"
            "Press SPACE to end.",
            wait_key=["space", "return"],
        )
        event_logger.log("session_end", task="debrief")

    except UserQuit:
        print("Session aborted by operator (Escape pressed).")
        event_logger.log("session_aborted", task=None)
    except Exception:
        # Deliberately NOT silent: a bare `finally: core.quit()` here would
        # call sys.exit() and swallow whatever exception is propagating,
        # which is exactly what was hiding the real crash before this fix.
        import traceback

        print("\n--- SESSION CRASHED -- full traceback below ---")
        traceback.print_exc()
        event_logger.log("session_crashed", task=None, traceback=traceback.format_exc())
    finally:
        event_logger.close()
        win.close()


if __name__ == "__main__":
    main()
