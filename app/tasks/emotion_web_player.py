"""Task 2 replacement: web-based emotion video player, per
Engineering_Document.md SS4.2.

Renders in a pywebview subprocess (app/webui/video_player_app.py) instead of
driving VLC -- see app/webui/bridge.py for why this runs out-of-process, and
app/tasks/vlc_rc_player.py's docstring for the flicker history this
replaces. `run_emotion_task(win, ctx, blocks)` mirrors
app/tasks/emotion_faced.py's function of the same name, minus the
now-unneeded player_path parameter (there's no external player to resolve
anymore). `blocks` still comes from app.tasks.emotion_faced.select_task_clips
(reused unchanged -- pure selection/layout logic, no psychopy dependency),
and EMOTION_OPTIONS/EMOTION_PROMPT/RATING_PROMPTS are reused unchanged too so
wording stays identical to today.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from app.tasks.emotion_faced import EMOTION_OPTIONS, EMOTION_PROMPT, RATING_PROMPTS
from app.ui.common_widgets import UserQuit
from app.webui.bridge import QUIT_EXIT_CODE, save_json_atomic

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# CSS-valid stand-ins for emotion_faced.py's _VALENCE_COLORS (that dict uses
# psychopy-style names like "gray40", not valid CSS) -- same fallback role as
# emotion_faced._play_placeholder: shown only if a clip's file_path doesn't
# exist on disk yet.
_PLACEHOLDER_COLORS = {"positive": "seagreen", "negative": "firebrick", "neutral": "#666666"}


def _clip_payload(clip: dict, ctx) -> dict:
    # duration_sec is the placeholder's on-screen dwell time (demo-scaled,
    # like every other passive/macro duration) and, separately, a real
    # video's safety-timeout floor (see player.js's playVideoWithSkipKeys) --
    # matches emotion_faced.py's play_clip(ctx.scaled(...)) exactly, whose
    # flat +45s buffer (added on the JS side) already absorbs demo scaling
    # for every clip in this dataset (all well under 90s).
    file_path = REPO_ROOT / clip["file_path"]
    payload = {
        "clip_id": clip["clip_id"],
        "valence_group": clip["valence_group"],
        "fine_grained_label": clip["fine_grained_label"],
        "duration_sec": ctx.scaled(clip.get("duration_sec", 90)),
        "file_path": clip["file_path"],
    }
    if file_path.exists():
        payload["placeholder"] = False
        payload["file_url"] = file_path.resolve().as_uri()
    else:
        payload["placeholder"] = True
        payload["placeholder_color"] = _PLACEHOLDER_COLORS.get(clip["valence_group"], "#666666")
    return payload


def get_familiarization_clip(clips: list, cfg: dict) -> dict:
    """Fixed neutral preview clip (cfg["familiarization_clip_id"]) instead of
    emotion_faced.pick_practice_clip's random draw from the unused pool --
    per PI request, every participant should preview the exact same clip."""
    clip_id = cfg["familiarization_clip_id"]
    for clip in clips:
        if clip["clip_id"] == clip_id:
            return clip
    raise ValueError(f"emotion_task.familiarization_clip_id {clip_id!r} not found in the manifest")


def _rating_items(cfg: dict) -> list:
    items = list(cfg["rating_items"])
    if cfg.get("dominance_enabled"):
        items.append("dominance")
    return items


def _bootstrap_common(cfg: dict) -> dict:
    rating_items = _rating_items(cfg)
    return {
        "rating_scale_min": cfg["rating_scale_min"],
        "rating_scale_max": cfg["rating_scale_max"],
        "rating_items": rating_items,
        "rating_prompts": {
            key: RATING_PROMPTS.get(key, (f"Rate your {key.upper()}.", "not at all", "extremely"))
            for key in rating_items
        },
        "emotion_options": EMOTION_OPTIONS,
        "emotion_prompt": EMOTION_PROMPT,
    }


def _launch(win, ctx, bootstrap: dict, practice: bool) -> None:
    """`win` is the pywebview session-shell window (hidden/shown around the
    subprocess), or None when there's no parent window to hide (e.g.
    app/run_task.py's isolated single-task testing)."""
    if win is not None:
        win.hide()
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            bootstrap_path = Path(tmp_dir) / "video_player_bootstrap.json"
            save_json_atomic(bootstrap_path, bootstrap)

            args = [
                sys.executable, "-m", "app.webui.video_player_app",
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
        raise RuntimeError(f"video_player_app.py exited with code {result.returncode}")


def run_emotion_task(win, ctx, blocks: list) -> None:
    # Instructions screen ("TASK 2: EMOTION...") now lives in the session
    # shell (app/webui/shell/) instead of a psychopy show_message() call --
    # the shell shows it right before calling this function.
    cfg = ctx.config["emotion_task"]
    block_baseline = cfg["block_baseline"]
    bootstrap = _bootstrap_common(cfg)
    bootstrap.update({
        "practice": False,
        "rest_duration_sec": ctx.scaled(cfg["rest_duration_sec"]),
        # 3 baseline touchpoints per block (start/mid/end), replacing the old
        # per-clip fixation -- per PI request 2026-08-05 (too many short 5s
        # snippets weren't useful for windowed DL analysis; a few longer
        # ones per block are). See player.js's runBaselinePeriod()/runBlock().
        "block_baseline_start_sec": ctx.scaled(block_baseline["start_sec"]),
        "block_baseline_mid_sec": ctx.scaled(block_baseline["mid_sec"]),
        "block_baseline_end_sec": ctx.scaled(block_baseline["end_sec"]),
        "blocks": [[_clip_payload(clip, ctx) for clip in block] for block in blocks],
    })
    _launch(win, ctx, bootstrap, practice=False)


def play_practice_clip(win, ctx, clip: dict) -> None:
    """Single-clip preview for the session shell's familiarization flow (see
    app/webui/shell/shell.js) --
    replaces the old play_clip(...) + discrete_choice + rating_scale_0_7
    call sequence with one self-contained subprocess invocation. Runs via
    runOneClip() directly (see player.js's main()), which never calls
    runBlock()/runBaselinePeriod() -- so no block-baseline fields needed here."""
    cfg = ctx.config["emotion_task"]
    bootstrap = _bootstrap_common(cfg)
    bootstrap.update({
        "practice": True,
        "rest_duration_sec": 0.0,
        "blocks": [[_clip_payload(clip, ctx)]],
    })
    _launch(win, ctx, bootstrap, practice=True)
