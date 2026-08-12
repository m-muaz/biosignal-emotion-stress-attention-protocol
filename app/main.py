"""Session runner. Orchestrates the emotion+stress+attention/focus portion
of the protocol: consent -> questionnaire -> device sync -> familiarization
-> emotion task -> break -> stress task -> break -> attention/focus task
(Schulte table + Stroop test) -> conclusion.

All UI (consent, questionnaire, familiarization, task instructions, break,
conclusion) is rendered by a single persistent pywebview window (see
app/tasks/session_shell.py, app/webui/shell/) -- there is no psychopy window
in this flow at all. The emotion/stress/attention-focus task experiences
themselves (app/webui/raindrop_app.py, app/webui/video_player_app.py,
app/webui/schulte_app.py, app/webui/stroop_app.py) still run as separate
subprocesses that the shell window hides itself around; see
app/webui/bridge.py for why.

SART (the attention/focus task, README "Attention task (SART)") is
deliberately NOT orchestrated here -- it's a separate standalone process run
after this script's session ends, rather than embedded in this flow. The
Schulte/Stroop attention/focus task added alongside it (per PI discussion
2026-08-04, see app/tasks/attention_focus.py) IS embedded here, since it's a
lightweight pywebview subprocess like the emotion/stress tasks rather than
SART's separate program/venv. It replaced the highway task's old slot here
(see session_config.yaml's attention_highway_task comment) -- highway is
being repurposed into a stress task instead, pending separate PI notes.
`app/run_task.py --task attention` still runs the now-retired OpenMATB path
for testing that integration in isolation.

Usage:
    python -m app.main --participant-id P001
    python -m app.main --participant-id DEMO001 --demo-scale 0.2 --skip-questionnaire

--demo-scale shortens passive/macro block lengths (baseline duration, rest,
cue, fixation, video length, number of trials per block) so the whole
protocol can be walked through quickly. It deliberately does NOT shorten
interactive per-event timing (raindrop fall speed) -- scaling that would
make the task impossible for a human to actually try.
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
from app.tasks.emotion_faced import select_task_clips
from app.tasks.session_shell import run_session
from app.ui.common_widgets import UserQuit
from app.webui.bridge import set_windows_dpi_awareness

REPO_ROOT = Path(__file__).resolve().parent.parent


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Emotion/Stress/Attention collection session.")
    parser.add_argument(
        "--participant-id", default=None,
        help="Falls back to session.participant_id in config if omitted; an error if neither is set.",
    )
    parser.add_argument("--config", default=str(REPO_ROOT / "app" / "config" / "session_config.yaml"))
    parser.add_argument("--demo-scale", type=float, default=1.0, help="Multiplier for passive/macro durations; 1.0 = real timing.")
    parser.add_argument("--devices-mode", choices=["mock", "real"], default=None, help="Override devices.mode from config.")
    parser.add_argument(
        "--skip-device-sync", action="store_true",
        help="Skip this app's own device sync step entirely -- for when hardware was already "
             "synced/verified via the external scripts before this app was started (see "
             "devices.skip_sync in session_config.yaml, which this ORs with).",
    )
    parser.add_argument("--skip-questionnaire", action="store_true", help="Skip consent/questionnaire screens (quick demo runs).")
    parser.add_argument("--skip-familiarization", action="store_true")
    return parser.parse_args()


def main():
    set_windows_dpi_awareness()
    args = parse_args()
    config = load_config(args.config)
    if args.devices_mode:
        config["devices"]["mode"] = args.devices_mode

    # --participant-id (CLI) takes priority; session.participant_id (config)
    # is only the debug-run fallback -- see that key's comment in
    # session_config.yaml.
    participant_id = args.participant_id or config.get("session", {}).get("participant_id")
    if not participant_id:
        raise SystemExit(
            "No participant id given -- pass --participant-id or set session.participant_id in the config file."
        )

    session_id = f"{participant_id}_{int(time.time())}"
    session_dir = REPO_ROOT / config["logging"]["session_root"] / session_id

    event_logger = EventLogger(session_dir, session_id, participant_id)
    rng_seed = time.time()
    rng = random.Random(rng_seed)
    ctx = SessionContext(
        config=config, event_logger=event_logger, participant_id=participant_id, demo_scale=args.demo_scale,
        rng=rng, window_kwargs=None,
    )

    try:
        # skip_device_sync: CLI flag ORs with devices.skip_sync (config) -- see
        # that key's comment in session_config.yaml for the pre-synced-
        # externally workflow this supports. Mirrors app/run_task.py's
        # existing --skip-device-sync handling.
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

        clips = load_emotion_manifest(str(REPO_ROOT / config["emotion_task"]["manifest_path"]))
        fixed_clips = None
        if config["emotion_task"].get("clip_selection_mode", "random") == "fixed":
            fixed_clips = load_fixed_clips(str(REPO_ROOT / config["emotion_task"]["fixed_clips_path"]))
        blocks, _selected_by_group = select_task_clips(clips, config["emotion_task"], ctx.rng, fixed_clips=fixed_clips)

        run_session(ctx, config, clips, blocks, args.skip_questionnaire, args.skip_familiarization)
        return 0

    except UserQuit:
        # Operator pressed Escape (or closed the window) -- an intentional
        # abort, not a crash. Distinct exit code (2) so a calling script
        # (e.g. a wrapper that chains this into `app.run_task --task sart`)
        # can tell "operator quit" apart from "it crashed" and "it finished".
        print("Session aborted by operator (Escape pressed).")
        event_logger.log("session_aborted", task=None)
        return 2

    except Exception:
        # Deliberately NOT silent -- an unexpected crash here (before/outside
        # the shell's own per-phase guarding, see app/tasks/session_shell.py)
        # should still be visible, not swallowed.
        import traceback

        print("\n--- SESSION CRASHED -- full traceback below ---")
        traceback.print_exc()
        event_logger.log("session_crashed", task=None, traceback=traceback.format_exc())
        return 1
    finally:
        event_logger.close()


if __name__ == "__main__":
    sys.exit(main())
