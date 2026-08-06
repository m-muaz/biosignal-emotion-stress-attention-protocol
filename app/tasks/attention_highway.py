"""Highway dodge game -- a second, complementary attention/persisted-focus
task added alongside SART per PI discussion 2026-08-04 (see README
"Attention task (SART)"). Runs INSIDE the main emotion->stress->[here] flow
(app/tasks/session_shell.py), unlike SART, which stays a separate standalone
program run after this session ends.

Rationale for a second attention task: SART's data is one discrete go/no-go
response stream to a single stimulus location. This task instead requires
continuously monitoring several lanes of oncoming hazards AND the
participant's own position, and choosing a *direction* to respond with, not
just present/absent -- a genuinely different demand profile, not a
duplicate measurement. See app/webui/highway/game.js for the full mechanic
and the events it logs (obstacle_spawn/obstacle_resolved/lane_change).

Renders in a pywebview subprocess (app/webui/highway_app.py), same pattern
as app/tasks/stress_raindrop.py -- see app/webui/bridge.py for why this runs
out-of-process from psychopy.

No permadeath: a collision just gets logged and the drive continues, same
"fixed-duration, mistakes don't end the round" design as the raindrop game
-- continuous biosignal recording needs a block that runs its full length,
not one that can end early on the first mistake.

Difficulty ramps -- starts easy, gets harder over the drive, like a classic
road-crossing arcade game (per PI request 2026-08-04) -- but as discrete
TIERS (same trial_order/tiers pattern as stress_raindrop.py), each
internally constant, rather than a smooth continuous ramp: a continuous
ramp would confound "it got harder" with "attention lapsed" at every single
instant, whereas each tier's own block_start/block_end gives analysis a
known, constant-difficulty window to isolate vigilance decrement within,
separate from the deliberate jump between tiers.

app/webui/highway/game.js also guarantees at least one lane is always free
of any live hazard (pickSpawnLane()) -- a hazard is never spawned into the
one remaining clear lane while every other lane already has a live threat,
so getting boxed in with no safe move is never possible, no matter the tier.

Anti-camping nudge (per PI request 2026-08-04): with no permadeath and a
guaranteed safe lane, sitting in one lane the whole time is otherwise a
perfectly safe (if inattentive) strategy. Can't literally force a keypress,
but pickSpawnLane() ramps up the probability that the next hazard spawns
directly into the participant's current lane the longer they've stayed put
-- see camp_grace_sec/camp_ramp_sec/camp_max_probability below.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from app.ui.common_widgets import UserQuit
from app.webui.bridge import QUIT_EXIT_CODE, save_json_atomic

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Block-level self-report items (per PI discussion 2026-08-05 on repurposing
# this game as a stress task -- see this module's docstring). Deliberately a
# SEPARATE dict from app.tasks.emotion_faced.RATING_PROMPTS even where the
# construct name matches (arousal/valence) -- the wording there asks about a
# just-watched clip; this asks about a just-driven block, and reusing the
# clip wording verbatim would read oddly ("how did this clip make you feel"
# after a driving block). workload/frustration/perceived_control/engagement
# are phrased after NASA-TLX's subscales and arousal/valence after the SAM
# (Self-Assessment Manikin) two-axis model, per [[feedback_validated_instruments]]
# rather than free-hand wording -- flag any change here to the PI as a
# validity trade-off, same as any other validated-scale edit.
HIGHWAY_RATING_PROMPTS = {
    "stress": (
        "How stressed did you feel during that last attempt?",
        "Not at all stressed", "Extremely stressed",
    ),
    "workload": (
        "How mentally demanding was that last attempt?",
        "Very low", "Very high",
    ),
    "frustration": (
        "How frustrated or irritated did you feel during that drive?",
        "Not at all", "Extremely",
    ),
    "arousal": (
        "How calm or how excited/activated did that drive make you feel?",
        "Very calm", "Very excited",
    ),
    "valence": (
        "Overall, how did that drive make you feel -- unpleasant or pleasant?",
        "Very unpleasant", "Very pleasant",
    ),
    "perceived_control": (
        "How much control did you feel you had over the car during that attempt?",
        "No control at all", "Complete control",
    ),
    "engagement": (
        "How engaged or absorbed did you feel in that last attempt?",
        "Not at all engaged", "Completely engaged",
    ),
    "perceived_difficulty": (
        "How difficult did that last attempt feel?",
        "Very easy", "Very difficult",
    ),
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
            bootstrap_path = Path(tmp_dir) / "highway_bootstrap.json"
            save_json_atomic(bootstrap_path, bootstrap)

            args = [
                sys.executable, "-m", "app.webui.highway_app",
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
        raise RuntimeError(f"highway_app.py exited with code {result.returncode}")


def _build_tiers_payload(cfg: dict, ctx) -> list[dict]:
    """Resolves `attention_highway_task.trial_order` against `.tiers` into an
    ordered list ready for the frontend, same shape/validation pattern as
    stress_raindrop.py's tier_by_id lookup. Only duration_sec is demo-scaled
    -- spawn_interval_range_sec/fall_duration_sec are the actual difficulty
    controller and must stay real, same reasoning stress_raindrop.py never
    scales its own tiers' pacing.

    nominal_difficulty is passed through as its OWN field, deliberately
    separate from `id`/`label` -- per the stress-task repurposing discussion,
    the game must never collapse "which tier" and "how stressed was the
    participant" into one value; nominal_difficulty is the experimenter-set
    manipulation, logged alongside (not instead of) the behavioral/
    self-report/physiological outcomes that later analysis checks it against.
    Falls back to the tier's own id if a tier config omits it, so existing
    session_config.yaml tiers that predate this field still work."""
    tier_by_id = {tier_cfg["id"]: tier_cfg for tier_cfg in cfg["tiers"]}
    try:
        return [
            {
                "id": tier_id,
                "label": tier_by_id[tier_id]["label"],
                "duration_sec": ctx.scaled(tier_by_id[tier_id]["duration_sec"]),
                "spawn_interval_range_sec": tier_by_id[tier_id]["spawn_interval_range_sec"],
                "fall_duration_sec": tier_by_id[tier_id]["fall_duration_sec"],
                "nominal_difficulty": tier_by_id[tier_id].get("nominal_difficulty", tier_id),
                # Second difficulty axis (per playtesting feedback
                # 2026-08-05): how many hazards spawn together per wave,
                # instead of only varying how fast single hazards come --
                # see app/webui/highway/game.js's scheduleNext(). Defaults
                # to 1 (today's one-at-a-time behavior) for any tier that
                # doesn't set it.
                "concurrent_obstacles": tier_by_id[tier_id].get("concurrent_obstacles", 1),
            }
            for tier_id in cfg["trial_order"]
        ]
    except KeyError as e:
        raise ValueError(
            f"attention_highway_task.trial_order references tier id {e.args[0]!r}, "
            "which isn't defined in attention_highway_task.tiers"
        ) from e


def _build_self_report_payload(cfg: dict) -> dict | None:
    """`attention_highway_task.self_report.items` lists which of
    HIGHWAY_RATING_PROMPTS to actually ask (same pattern
    app/tasks/emotion_web_player.py's _rating_items/_bootstrap_common uses
    for the emotion task) -- absent/empty means no self-report at all
    (bootstrap.self_report stays None, and game.js's main() skips
    runSelfReport() entirely), so this can be disabled with one config edit
    without touching code."""
    items = cfg.get("self_report", {}).get("items")
    if not items:
        return None
    return {
        "items": list(items),
        "prompts": {key: HIGHWAY_RATING_PROMPTS[key] for key in items},
        "scale_min": cfg["self_report"]["scale_min"],
        "scale_max": cfg["self_report"]["scale_max"],
    }


def _telemetry_bootstrap(cfg: dict) -> dict:
    """Fields shared by the real task and the demo -- clock-sync/state-tick/
    frame-drop/near-miss knobs, all configurable and all defaulted here (not
    required in session_config.yaml) so an older config file without them
    still runs, just with these fixed defaults instead of tuned ones."""
    return {
        "state_tick_hz": cfg.get("state_tick_hz", 10),
        "rolling_window_sec": cfg.get("rolling_window_sec", 15),
        "near_miss_window_sec": cfg.get("near_miss_window_sec", 0.3),
        "frame_drop_threshold_ms": cfg.get("frame_drop_threshold_ms", 50),
        "frame_drop_min_gap_sec": cfg.get("frame_drop_min_gap_sec", 0.5),
        "sync_checkpoint_count": cfg.get("sync_checkpoint_count", 3),
        "sync_checkpoint_gap_sec": cfg.get("sync_checkpoint_gap_sec", 0.2),
    }


def run_attention_highway_task(win, ctx) -> None:
    cfg = ctx.config["attention_highway_task"]
    bootstrap = {
        "practice": False,
        "lanes": cfg["lanes"],
        "baseline_duration_sec": ctx.scaled(cfg["baseline_duration_sec"]),
        "tiers": _build_tiers_payload(cfg, ctx),
        # Anti-camping nudge -- tied to real reaction time, not demo-scaled
        # (same reasoning the tiers' own pacing isn't scaled either).
        "camp_grace_sec": cfg["camp_grace_sec"],
        "camp_ramp_sec": cfg["camp_ramp_sec"],
        "camp_max_probability": cfg["camp_max_probability"],
        "self_report": _build_self_report_payload(cfg),
        **_telemetry_bootstrap(cfg),
    }
    _launch(win, ctx, bootstrap, practice=False)


def run_demo(win, ctx) -> None:
    """Non-interactive demo for the session shell's familiarization flow --
    hazards approach and auto-dodge themselves (app/webui/highway/game.js's
    bootstrap.demo path), using tier 1's pacing, for
    attention_highway_task.demo_duration_sec, so the participant watches the
    mechanic instead of playing a real (scored) round."""
    cfg = ctx.config["attention_highway_task"]
    tier_cfg = cfg["tiers"][0]
    bootstrap = {
        "demo": True,
        "practice": True,
        "lanes": cfg["lanes"],
        "baseline_duration_sec": 0.0,
        "tiers": [{
            "id": tier_cfg["id"],
            "label": tier_cfg["label"],
            "duration_sec": cfg["demo_duration_sec"],
            "spawn_interval_range_sec": tier_cfg["spawn_interval_range_sec"],
            "fall_duration_sec": tier_cfg["fall_duration_sec"],
            "nominal_difficulty": tier_cfg.get("nominal_difficulty", tier_cfg["id"]),
            "concurrent_obstacles": tier_cfg.get("concurrent_obstacles", 1),
        }],
        "camp_grace_sec": cfg["camp_grace_sec"],
        "camp_ramp_sec": cfg["camp_ramp_sec"],
        "camp_max_probability": cfg["camp_max_probability"],
        # No self_report here -- game.js's demo branch returns before ever
        # checking bootstrap.self_report, since a practice/familiarization
        # run isn't part of the scored dataset. state_tick/frame-drop/etc.
        # ARE still included -- harmless during demo, and useful for
        # sanity-checking the new logging fields on a short run before
        # committing to a full scored session.
        **_telemetry_bootstrap(cfg),
    }
    _launch(win, ctx, bootstrap, practice=True)
