"""Task 3 replacement: raindrop math game, per Engineering_Document.md SS4.3.

Renders in a pywebview subprocess (app/webui/raindrop_app.py) instead of
psychopy widgets -- see app/webui/bridge.py for why this runs out-of-process.
`run_stress_task(win, ctx)` keeps the exact signature app/tasks/stress_mat.py
had, so app/main.py and app/run_task.py only need an import swap.

No rigged leaderboard here (the old MIST-style deception -- see
stress_mat.rig_rival_score -- is dropped entirely, per PI request): score is
just +1 per correct answer, no ceiling, plus an honest cross-participant high
score persisted at stress_task.high_score_path.

This module only generates each tier's question queue (via
app.tasks.stress_mat.generate_question, using ctx.rng -- so provenance stays
with the one rng_seed recorded in session_manifest.json) and hands it to the
subprocess as a pre-built bootstrap payload; the subprocess never touches
randomness itself. Baseline/arithmetic block *lengths* are demo-scaled like
every other task (ctx.scaled), but each tier's spawn_interval_sec/
fall_duration_sec -- the actual stress controller -- are never scaled, same
reasoning stress_mat.py's per-question time limit never was: shortening the
pressure for a demo run would defeat the point of a demo run that's supposed
to let a human operator actually feel the task.

Per PI request 2026-08-04, all tiers now use the SAME arithmetic complexity
(session_config.yaml's stress_task.tiers: all four operators, 2 single-digit
operands) -- difficulty comes ENTIRELY from spawn_interval_sec/
fall_duration_sec, same design philosophy as attention_highway_task's tiers.
stress_task.baseline_mode controls whether the baseline fixation repeats
before every tier ("per_trial", default) or only once at the very start
("once", for an unbroken stress ramp) -- see app/webui/raindrop/game.js.
"""

import math
import subprocess
import sys
import tempfile
from pathlib import Path

from app.tasks.stress_mat import generate_question
from app.ui.common_widgets import UserQuit
from app.webui.bridge import QUIT_EXIT_CODE, save_json_atomic

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

QUESTION_BUFFER = 5  # extra questions over the estimated need, cheap insurance against under-provisioning


def _build_tier_payload(tier_cfg: dict, rng, duration_sec: float) -> dict:
    n_questions = math.ceil(duration_sec / tier_cfg["spawn_interval_sec"]) + QUESTION_BUFFER
    questions = []
    for _ in range(n_questions):
        expr, answer = generate_question(tier_cfg, rng)
        questions.append({"expr": expr, "answer": answer})
    return {
        "spawn_interval_sec": tier_cfg["spawn_interval_sec"],
        "fall_duration_sec": tier_cfg["fall_duration_sec"],
        "questions": questions,
    }


def _launch(win, ctx, bootstrap: dict, practice: bool) -> None:
    """`win` is the pywebview session-shell window (hidden/shown around the
    subprocess so it doesn't compete for the foreground), or None when
    there's no parent window to hide (e.g. app/run_task.py's isolated
    single-task testing)."""
    if win is not None:
        win.hide()
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            bootstrap_path = Path(tmp_dir) / "raindrop_bootstrap.json"
            save_json_atomic(bootstrap_path, bootstrap)

            args = [
                sys.executable, "-m", "app.webui.raindrop_app",
                "--session-dir", str(ctx.event_logger.session_dir),
                "--session-id", ctx.event_logger.session_id,
                "--participant-id", ctx.participant_id,
                "--bootstrap", str(bootstrap_path),
            ]
            if practice:
                args.append("--practice")

            result = subprocess.run(args)
    finally:
        if win is not None:
            win.show()

    if result.returncode == QUIT_EXIT_CODE:
        raise UserQuit()
    if result.returncode != 0:
        raise RuntimeError(f"raindrop_app.py exited with code {result.returncode}")


def run_stress_task(win, ctx) -> None:
    cfg = ctx.config["stress_task"]
    baseline_duration = ctx.scaled(cfg["baseline_duration_sec"])
    arithmetic_duration = ctx.scaled(cfg["arithmetic_duration_sec"])

    tier_by_id = {tier_cfg["id"]: tier_cfg for tier_cfg in cfg["tiers"]}
    trial_order = list(cfg["trial_order"])
    try:
        tiers_payload = {
            str(tier_id): _build_tier_payload(tier_by_id[tier_id], ctx.rng, arithmetic_duration)
            for tier_id in set(trial_order)
        }
    except KeyError as e:
        raise ValueError(
            f"stress_task.trial_order references tier id {e.args[0]!r}, which isn't defined in stress_task.tiers"
        ) from e

    bootstrap = {
        "practice": False,
        "high_score_path": str((REPO_ROOT / cfg["high_score_path"]).resolve()),
        "baseline_duration_sec": baseline_duration,
        "arithmetic_duration_sec": arithmetic_duration,
        "trial_order": trial_order,
        "tiers": tiers_payload,
        # "per_trial" (default) shows a fresh baseline + a paced pause
        # before EVERY tier; "once" shows it only before the first tier and
        # then runs every tier back-to-back with no rest -- see
        # app/webui/raindrop/game.js's main() for the actual gating logic.
        "baseline_mode": cfg.get("baseline_mode", "per_trial"),
    }
    _launch(win, ctx, bootstrap, practice=False)


def run_demo(win, ctx) -> None:
    """Non-interactive demo for the session shell's familiarization flow --
    replaces the old get_numeric_answer(..., "example", ...) call (and this
    module's earlier interactive run_practice_round). Drops fall and
    auto-resolve themselves (app/webui/raindrop/game.js's bootstrap.demo
    path) for stress_task.demo_duration_sec, using tier 1's pacing; never
    touches the high-score file (--practice tags events task=
    "familiarization" and get_high_score/submit_score simply aren't called
    in demo mode)."""
    cfg = ctx.config["stress_task"]
    tier_cfg = cfg["tiers"][0]
    duration = cfg["demo_duration_sec"]
    bootstrap = {
        "demo": True,
        "practice": True,
        "high_score_path": None,
        "baseline_duration_sec": 0.0,
        "arithmetic_duration_sec": duration,
        "trial_order": [tier_cfg["id"]],
        "tiers": {str(tier_cfg["id"]): _build_tier_payload(tier_cfg, ctx.rng, duration)},
    }
    _launch(win, ctx, bootstrap, practice=True)
