"""Session runner. Orchestrates the full 63-min protocol from
Engineering_Document.md: consent -> questionnaire -> device sync ->
familiarization -> Task 1 -> break -> Task 2 -> break -> Task 3 -> debrief.

Usage:
    python -m app.main --participant-id P001
    python -m app.main --participant-id DEMO001 --demo-scale 0.2 --skip-questionnaire

--demo-scale shortens passive/macro block lengths (baseline duration, rest,
cue, fixation, video length, number of trials per block) so the whole
protocol can be walked through quickly. It deliberately does NOT shorten
interactive per-event timing (n-back SOA, stress per-question time limit) --
scaling those would make the task impossible for a human to actually try.
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
from app.tasks.attention_nback import build_grid, run_attention_task, run_trial
from app.tasks.emotion_faced import run_emotion_task
from app.tasks.stress_mat import generate_question, get_numeric_answer, run_stress_task
from app.ui.common_widgets import UserQuit, discrete_choice, numeric_entry, rating_scale_0_7, show_message

REPO_ROOT = Path(__file__).resolve().parent.parent

CONSENT_TEXT = """INFORMED CONSENT (draft -- pending IRB/PI review, see Engineering_Document.md §3.3)

Purpose: we are collecting biosignals to study emotion, stress, and attention/focus states for machine learning research.

What will happen:
  1. An attention/memory grid game.
  2. Watching short video clips and rating your emotional reaction to each.
  3. A timed mental arithmetic task with a countdown timer and leaderboard, designed to be mildly stressful.

Sensors worn (all non-invasive): two ear-EEG devices, a wristband (heart rate, motion, temperature, skin conductance), and for some participants a 32-channel EEG cap.

Duration: about 63 minutes total.

Data collected: physiological signals, task responses/accuracy, and self-report ratings. No audio/video recording of you unless separately stated.

Possible discomfort: mild sensation from electrode gel/wristband strap; some clips may evoke sadness, disgust, or fear; the arithmetic task is deliberately time-pressured.

Participation is voluntary. You may withdraw at any time without penalty, and may skip any question.

Your data is stored under an assigned participant ID, not your name.

Press SPACE to indicate you have read this and consent to participate."""


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
    show_message(win, CONSENT_TEXT, wait_key=["space", "return"])
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


def run_familiarization(win, ctx, cfg):
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

    show_message(
        win,
        "Preview: ATTENTION task. Watch the grid -- one cell lights up at a time. Here's one example trial.\n\n"
        "Press SPACE to see it.",
        wait_key=["space", "return"],
    )
    grid_size = cfg["attention_task"]["grid_size"]
    cells = build_grid(win, grid_size)
    run_trial(win, cells, pos=(grid_size * grid_size) // 2, soa=cfg["attention_task"]["soa_sec"], stim_on_sec=0.6)

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

    show_message(
        win,
        "Preview: EMOTION task. After each short clip, you'll rate how it made you feel, like this.\n\n"
        "Press SPACE to try an example rating.",
        wait_key=["space", "return"],
    )
    rating_scale_0_7(
        win, "Rate your VALENCE (example only -- not a real clip).",
        cfg["emotion_task"]["rating_scale_min"], cfg["emotion_task"]["rating_scale_max"],
    )

    show_message(win, "That's everything. The real tasks begin now.\n\nPress SPACE to start.", wait_key=["space", "return"])
    ctx.event_logger.log("phase_end", task="familiarization")


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
    rng = random.Random(args.participant_id)
    ctx = SessionContext(config=config, event_logger=event_logger, participant_id=args.participant_id, demo_scale=args.demo_scale, rng=rng)

    win = visual.Window(size=(1280, 800), color="black", units="height", fullscr=args.fullscreen)
    _raise_window_focus()

    try:
        print(f"Syncing devices (mode={config['devices']['mode']})...")
        sync_records = sync_all_blocking(config["devices"], config["devices"]["mode"])
        for role, records in sync_records.items():
            event_logger.log("device_sync", task="preparation", **records[0])
            if records[0]["result"] != "ok":
                print(f"WARNING: sync failed for device '{role}': {records[0]['detail']}")
        write_session_manifest(session_dir, session_id, args.participant_id, config, sync_records, args.demo_scale)

        if not args.skip_questionnaire:
            run_consent_and_questionnaire(win, ctx, config)
        if not args.skip_familiarization:
            run_familiarization(win, ctx, config)

        run_attention_task(win, ctx)

        show_message(win, f"Break -- take a moment to relax.", duration=ctx.scaled(config["breaks"]["after_task1_min"] * 60))

        clips = load_emotion_manifest(str(REPO_ROOT / config["emotion_task"]["manifest_path"]))
        run_emotion_task(win, ctx, clips)

        show_message(win, f"Break -- take a moment to relax.", duration=ctx.scaled(config["breaks"]["after_task2_min"] * 60))

        run_stress_task(win, ctx)

        show_message(
            win,
            "DEBRIEF\n\nThat concludes the session. Thank you for participating!\n\n"
            "The experimenter will now remove your sensors and answer any questions.\n\n"
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
