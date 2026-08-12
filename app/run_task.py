"""Standalone single-task runner, for testing one protocol module in isolation.

Sets up the same config/device-sync/event-logging as a real session (see
app/main.py's main()) so what you see while testing one task matches what a
participant would see -- just without the surrounding consent/questionnaire/
familiarization/breaks/other-task flow.

emotion/stress/highway/attention_focus tests launch their subprocess
(app/webui/video_player_app.py, app/webui/raindrop_app.py,
app/webui/highway_app.py, app/webui/schulte_app.py/app/webui/stroop_app.py)
directly with no parent window to hide (there's no session shell here) --
see app/tasks/emotion_web_player.py/app/tasks/stress_raindrop.py/
app/tasks/attention_highway.py/app/tasks/attention_focus.py's `win=None`
handling. attention (OpenMATB) and sart (app/tasks/attention_sart.py) are
untouched by this protocol's web migration and still need a real psychopy
window, so this is the one remaining place that creates one for either of
them -- see _run_attention_task_isolated/_run_sart_task_isolated below.

Usage:
    python -m app.run_task --task attention --participant-id TEST001
    python -m app.run_task --task emotion --participant-id TEST001 --demo-scale 0.2
    python -m app.run_task --task emotion --participant-id TEST001 --emotion-block-structure grouped_by_valence
    python -m app.run_task --task emotion --participant-id TEST001 --emotion-clip-selection random
    python -m app.run_task --task stress --participant-id TEST001
    python -m app.run_task --task stress --participant-id TEST001 --skip-device-sync
    python -m app.run_task --task highway --participant-id TEST001
    python -m app.run_task --task attention_focus --participant-id TEST001
    python -m app.run_task --task sart --participant-id TEST001
    python -m app.run_task --task sart --participant-id TEST001 --fullscreen
"""

import argparse
import random
import sys
import time
from pathlib import Path

from app.config.loader import load_config, load_emotion_manifest, load_fixed_clips
from app.context import SessionContext
from app.eventlog.event_logger import EventLogger
from app.eventlog.session_manifest import write_session_manifest
from app.sync.device_sync import sync_all_blocking
from app.tasks.attention_focus import run_attention_focus_task
from app.tasks.attention_highway import run_attention_highway_task
from app.tasks.emotion_faced import select_task_clips
from app.tasks.emotion_web_player import run_emotion_task
from app.tasks.stress_raindrop import run_stress_task
from app.ui.common_widgets import UserQuit
from app.webui.bridge import set_windows_dpi_awareness

REPO_ROOT = Path(__file__).resolve().parent.parent

TASK_LABELS = {
    "attention": "TASK 1: ATTENTION (OpenMATB, standalone -- retired in favor of SART, see README)",
    "emotion": "TASK 2: EMOTION",
    "stress": "TASK 3: STRESS",
    "highway": "HIGHWAY (not currently in app.main's flow -- see session_config.yaml's attention_highway_task comment)",
    "attention_focus": "TASK 4: ATTENTION/FOCUS (Schulte table + Stroop test)",
    "sart": "SART (Sustained Attention to Response Task -- deliberately kept separate from app.main's flow per PI request, see README)",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run a single protocol task in isolation, for module-by-module testing.")
    parser.add_argument("--task", required=True, choices=sorted(TASK_LABELS))
    parser.add_argument(
        "--participant-id", default=None,
        help="Use a clearly test-only id (e.g. TEST001) to keep test runs separate from real participant data. "
             "Falls back to session.participant_id in config if omitted; an error if neither is set.",
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "app" / "config" / "session_config.yaml"))
    parser.add_argument(
        "--demo-scale", type=float, default=1.0,
        help="Multiplier for passive/macro durations; 1.0 = real timing (same semantics as app/main.py).",
    )
    parser.add_argument("--devices-mode", choices=["mock", "real"], default=None, help="Override devices.mode from config.")
    parser.add_argument(
        "--skip-device-sync", action="store_true",
        help="Skip the device sync step entirely (ORs with devices.skip_sync in session_config.yaml).",
    )
    parser.add_argument(
        "--fullscreen", action="store_true",
        help="Only affects --task attention/sart (the two tasks that still open their own real psychopy window).",
    )
    parser.add_argument(
        "--emotion-clip-selection", choices=["fixed", "random"], default=None,
        help="Override emotion_task.clip_selection_mode from config (fixed = same predefined clips for everyone; random = legacy per-participant sampling).",
    )
    parser.add_argument(
        "--emotion-block-structure", choices=["interleaved", "grouped_by_valence"], default=None,
        help="Override emotion_task.block_structure from config (interleaved = mixed-valence blocks; grouped_by_valence = legacy FACED-style same-valence blocks).",
    )
    parser.add_argument(
        "--skip-practice", action="store_true",
        help="Only affects --task sart. Skips SART's 18-trial practice block (overrides sart_task.practice from "
             "config to False) -- for when the participant already saw a demo of every task beforehand and "
             "doesn't need a separate in-task practice round. Mirrors app/main.py's --skip-familiarization.",
    )
    return parser.parse_args()


def _run_attention_task_isolated(ctx, fullscreen: bool):
    """attention/OpenMATB is out of scope for the web migration and still
    needs a real psychopy window -- imports are local to this function so
    psychopy is only ever touched when --task attention is actually chosen."""
    from psychopy import visual

    from app.tasks.attention_openmatb import run_attention_task
    from app.ui.common_widgets import BG_COLOR, _install_mouse_click_capture, request_quit

    def _install_close_handler(win):
        def _on_close():
            request_quit()
            return True

        win.winHandle.on_close = _on_close

    window_kwargs = dict(size=(1280, 800), color=BG_COLOR, units="height", fullscr=fullscreen)
    ctx.window_kwargs = window_kwargs
    win = visual.Window(**window_kwargs)
    _install_close_handler(win)
    _install_mouse_click_capture(win)
    try:
        run_attention_task(win, ctx)
    finally:
        win.close()


def _run_sart_task_isolated(ctx, fullscreen: bool):
    """SART (app/tasks/attention_sart.py) also still needs a real psychopy
    window -- same reasoning/pattern as _run_attention_task_isolated above.
    Deliberately NOT called from session_shell.py -- per PI request
    2026-08-05, SART stays a separate standalone task, run only via
    `python -m app.run_task --task sart` (see README's "Attention task
    (SART)" section and [[project_attention_task_sart]])."""
    from psychopy import visual

    from app.tasks.attention_sart import run_sart_task
    from app.ui.common_widgets import BG_COLOR, request_quit

    def _install_close_handler(win):
        def _on_close():
            request_quit()
            return True

        win.winHandle.on_close = _on_close

    window_kwargs = dict(size=(1280, 800), color=BG_COLOR, units="height", fullscr=fullscreen)
    ctx.window_kwargs = window_kwargs
    win = visual.Window(**window_kwargs)
    _install_close_handler(win)
    try:
        run_sart_task(win, ctx)
    finally:
        win.close()


def main():
    set_windows_dpi_awareness()
    args = parse_args()
    config = load_config(args.config)
    if args.devices_mode:
        config["devices"]["mode"] = args.devices_mode
    if args.emotion_clip_selection:
        config["emotion_task"]["clip_selection_mode"] = args.emotion_clip_selection
    if args.emotion_block_structure:
        config["emotion_task"]["block_structure"] = args.emotion_block_structure
    if args.skip_practice:
        config["sart_task"]["practice"] = False

    # --participant-id (CLI) takes priority; session.participant_id (config)
    # is only the debug-run fallback -- see that key's comment in
    # session_config.yaml.
    participant_id = args.participant_id or config.get("session", {}).get("participant_id")
    if not participant_id:
        raise SystemExit(
            "No participant id given -- pass --participant-id or set session.participant_id in the config file."
        )

    session_id = f"{participant_id}_{args.task}_{int(time.time())}"
    session_dir = REPO_ROOT / config["logging"]["session_root"] / session_id

    event_logger = EventLogger(session_dir, session_id, participant_id)
    rng_seed = time.time()
    rng = random.Random(rng_seed)
    ctx = SessionContext(
        config=config, event_logger=event_logger, participant_id=participant_id, demo_scale=args.demo_scale,
        rng=rng, window_kwargs=None,
    )

    try:
        # CLI flag ORs with devices.skip_sync (config) -- see that key's
        # comment in session_config.yaml. Mirrors app/main.py's handling.
        skip_sync = args.skip_device_sync or config["devices"].get("skip_sync", False)
        sync_records = {}
        if skip_sync:
            print("Skipping device sync (devices already synced externally) -- see devices.skip_sync/--skip-device-sync.")
            event_logger.log(
                "device_sync_skipped", task="preparation",
                reason="devices pre-synced/verified via external scripts before this app started",
            )
        else:
            print(f"Syncing devices (mode={config['devices']['mode']})...")
            sync_records = sync_all_blocking(config["devices"], config["devices"]["mode"])
            for role, records in sync_records.items():
                event_logger.log("device_sync", task="preparation", **records[0])
                if records[0]["result"] != "ok":
                    print(f"WARNING: sync failed for device '{role}': {records[0]['detail']}")
        write_session_manifest(session_dir, session_id, participant_id, config, sync_records, args.demo_scale, rng_seed)

        print(f"Running {TASK_LABELS[args.task]} in isolation (participant_id={participant_id})...")
        if args.task == "attention":
            _run_attention_task_isolated(ctx, args.fullscreen)
        elif args.task == "emotion":
            clips = load_emotion_manifest(str(REPO_ROOT / config["emotion_task"]["manifest_path"]))
            fixed_clips = None
            if config["emotion_task"].get("clip_selection_mode", "random") == "fixed":
                fixed_clips = load_fixed_clips(str(REPO_ROOT / config["emotion_task"]["fixed_clips_path"]))
            blocks, _ = select_task_clips(clips, config["emotion_task"], ctx.rng, fixed_clips=fixed_clips)
            run_emotion_task(None, ctx, blocks)
        elif args.task == "stress":
            run_stress_task(None, ctx)
        elif args.task == "highway":
            run_attention_highway_task(None, ctx)
        elif args.task == "attention_focus":
            run_attention_focus_task(None, ctx)
        elif args.task == "sart":
            _run_sart_task_isolated(ctx, args.fullscreen)

        event_logger.log("session_end", task=args.task)
        print(f"\n'{args.task}' task complete. Event log written to {session_dir / 'events.jsonl'}")
        return 0

    except UserQuit:
        # Distinct exit code (2) from a crash (1) so a calling script (e.g. a
        # wrapper chaining `app.main` into `app.run_task --task sart`) can
        # tell "operator quit" apart from "it crashed" and "it finished".
        print("Task run aborted by operator (Escape pressed).")
        event_logger.log("session_aborted", task=None)
        return 2
    except Exception:
        # Deliberately NOT silent -- see app/main.py's main() for the same reasoning.
        import traceback

        print("\n--- TASK RUN CRASHED -- full traceback below ---")
        traceback.print_exc()
        event_logger.log("session_crashed", task=None, traceback=traceback.format_exc())
        return 1
    finally:
        event_logger.close()


if __name__ == "__main__":
    sys.exit(main())
