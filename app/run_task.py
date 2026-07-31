"""Standalone single-task runner, for testing one protocol module in isolation.

Sets up the same window, config, device sync, and event logging as a real
session (see app/main.py's main()) so what you see while testing one task
matches what a participant would see -- just without the surrounding
consent/questionnaire/familiarization/breaks/other-task flow.

Usage:
    python -m app.run_task --task attention --participant-id TEST001
    python -m app.run_task --task emotion --participant-id TEST001 --demo-scale 0.2
    python -m app.run_task --task stress --participant-id TEST001
    python -m app.run_task --task stress --participant-id TEST001 --skip-device-sync
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
from app.main import _raise_window_focus
from app.sync.device_sync import sync_all_blocking
from app.tasks.attention_openmatb import run_attention_task
from app.tasks.emotion_faced import ExternalPlayerNotFound, resolve_external_player, run_emotion_task, select_task_clips
from app.tasks.stress_mat import run_stress_task
from app.ui.common_widgets import BG_COLOR, UserQuit, show_message

REPO_ROOT = Path(__file__).resolve().parent.parent

TASK_LABELS = {"attention": "TASK 1: ATTENTION", "emotion": "TASK 2: EMOTION", "stress": "TASK 3: STRESS"}


def parse_args():
    parser = argparse.ArgumentParser(description="Run a single protocol task in isolation, for module-by-module testing.")
    parser.add_argument("--task", required=True, choices=sorted(TASK_LABELS))
    parser.add_argument(
        "--participant-id", required=True,
        help="Use a clearly test-only id (e.g. TEST001) to keep test runs separate from real participant data.",
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "app" / "config" / "session_config.yaml"))
    parser.add_argument(
        "--demo-scale", type=float, default=1.0,
        help="Multiplier for passive/macro durations; 1.0 = real timing (same semantics as app/main.py).",
    )
    parser.add_argument("--devices-mode", choices=["mock", "real"], default=None, help="Override devices.mode from config.")
    parser.add_argument("--skip-device-sync", action="store_true", help="Skip the device sync step entirely.")
    parser.add_argument("--fullscreen", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.devices_mode:
        config["devices"]["mode"] = args.devices_mode

    session_id = f"{args.participant_id}_{args.task}_{int(time.time())}"
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
        sync_records = {}
        if not args.skip_device_sync:
            print(f"Syncing devices (mode={config['devices']['mode']})...")
            sync_records = sync_all_blocking(config["devices"], config["devices"]["mode"])
            for role, records in sync_records.items():
                event_logger.log("device_sync", task="preparation", **records[0])
                if records[0]["result"] != "ok":
                    print(f"WARNING: sync failed for device '{role}': {records[0]['detail']}")
        write_session_manifest(session_dir, session_id, args.participant_id, config, sync_records, args.demo_scale, rng_seed)

        print(f"Running {TASK_LABELS[args.task]} in isolation (participant_id={args.participant_id})...")
        if args.task == "attention":
            win = run_attention_task(win, ctx)
        elif args.task == "emotion":
            try:
                player_path = resolve_external_player(config["emotion_task"])
            except ExternalPlayerNotFound as exc:
                show_message(win, f"SETUP ERROR\n\n{exc}\n\nPress SPACE to abort.", wait_key=["space", "return"])
                raise
            clips = load_emotion_manifest(str(REPO_ROOT / config["emotion_task"]["manifest_path"]))
            block_order, selected_by_group = select_task_clips(clips, config["emotion_task"], ctx.rng)
            win = run_emotion_task(win, ctx, block_order, selected_by_group, player_path)
        elif args.task == "stress":
            run_stress_task(win, ctx)

        event_logger.log("session_end", task=args.task)
        print(f"\n'{args.task}' task complete. Event log written to {session_dir / 'events.jsonl'}")

    except UserQuit:
        print("Task run aborted by operator (Escape pressed).")
        event_logger.log("session_aborted", task=None)
    except Exception:
        # Deliberately NOT silent -- see app/main.py's main() for the same reasoning.
        import traceback

        print("\n--- TASK RUN CRASHED -- full traceback below ---")
        traceback.print_exc()
        event_logger.log("session_crashed", task=None, traceback=traceback.format_exc())
    finally:
        event_logger.close()
        win.close()


if __name__ == "__main__":
    main()
