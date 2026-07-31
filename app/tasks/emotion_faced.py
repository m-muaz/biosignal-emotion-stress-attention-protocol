"""Task 2: video-based emotion induction, per Engineering_Document.md §4.2.

Clips are blocked by valence group (same-valence clips shown consecutively,
per FACED's design), with block order and within-block clip order both
shuffled per participant. Manifest-driven: swapping the clip set (currently
OpenLAV, previously FilmStim) is a config change (emotion_manifest.json),
not a code change -- if a clip's file_path doesn't exist on disk yet,
playback falls back to a procedurally-rendered placeholder automatically.

Clips are played in an external video player (VLC, via `--play-and-exit`)
rather than PsychoPy's own MovieStim -- MovieStim's frame-by-frame draw loop
had recurring decode-stall and early-cutoff bugs (a clip's actual decodable
length sometimes ran short of the manifest's probed duration_sec, and the
sdl2/ffpyplayer audio backend could stall a draw() call for tens of seconds
with no exception). VLC owns real playback and closes itself when the file
ends (or the participant closes it manually); we just launch it and block
until it exits. Trade-off: skip-clip/skip-block keys don't work *during*
playback anymore (VLC, not our event loop, has focus) -- only between clips.
"""

import shutil
import subprocess
import time
from pathlib import Path

from psychopy import core, event, visual

from app.ui.common_widgets import (
    BG_COLOR,
    MUTED_COLOR,
    TEXT_COLOR,
    SkipBlock,
    SkipTrial,
    UserQuit,
    check_quit,
    check_skip_block,
    check_skip_trial,
    discrete_choice,
    draw_key_hint,
    draw_progress_bar,
    fixation_cross,
    rating_scale_multi,
    show_message,
)

_VALENCE_COLORS = {"positive": "seagreen", "negative": "firebrick", "neutral": "gray40"}

_VLC_CANDIDATES = [
    r"C:\Program Files\VideoLAN\VLC\vlc.exe",
    r"C:\Program Files (x86)\VideoLAN\VLC\vlc.exe",
]


class ExternalPlayerNotFound(RuntimeError):
    pass


def resolve_external_player(cfg: dict) -> str:
    """Locate the external video player executable (config override -> PATH
    -> common Windows install paths), raising a clear, actionable error if
    none is found -- mirrors attention_openmatb.py's OpenMATBNotInstalled UX.
    """
    override = cfg.get("external_player_path")
    if override:
        if not Path(override).exists():
            raise ExternalPlayerNotFound(f"Configured emotion_task.external_player_path not found: {override}")
        return override
    found = shutil.which("vlc")
    if found:
        return found
    for candidate in _VLC_CANDIDATES:
        if Path(candidate).exists():
            return candidate
    raise ExternalPlayerNotFound(
        "No video player found for the emotion task.\n\n"
        "Install VLC (https://www.videolan.org/vlc/), or set emotion_task.external_player_path "
        "in session_config.yaml to your player's .exe path."
    )

# Each item is (prompt, left_label, right_label) -- explicit endpoint labels
# instead of the old generic "not at all"/"extremely" on every row, which
# didn't fit a bipolar item like valence (0 doesn't obviously mean "very
# unpleasant" without being told) and left the participant guessing which
# direction the scale ran.
_RATING_PROMPTS = {
    "valence": (
        "Overall, how did this clip make you feel -- unpleasant or pleasant?",
        "Very unpleasant", "Very pleasant",
    ),
    "arousal": (
        "How calm or how excited/activated did this clip make you feel?",
        "Very calm", "Very excited",
    ),
    "liking": (
        "How much did you personally like this clip?",
        "Not at all", "Extremely",
    ),
    "dominance": (
        "How much control did you feel over your own emotions while watching?",
        "No control at all", "Complete control",
    ),
}

# Fixed option list, shown after every clip regardless of valence group (see
# ui-mockup/emotion_test/emotion picker.png). Deliberately NOT derived from
# the specific clips sampled into a block (the old behavior only asked this
# for non-neutral blocks, and built the option list from whichever
# fine_grained_labels happened to be in the 3 sampled clips) -- either of
# those would let an attentive participant infer the block's valence from
# the mere presence/wording of the question, biasing their reports.
# A 3-way positive/negative/neutral self-report, exhaustive and mutually
# exclusive (no "Other" needed) -- doubles as a manipulation check against
# the clip's intended valence_group.
# Public (no leading underscore): app/main.py reuses these for the
# familiarization preview, so the practice run asks the exact same question.
EMOTION_OPTIONS = ["Positive", "Negative", "Neutral"]
EMOTION_PROMPT = "How would you categorize your overall feeling from that clip?"


def select_task_clips(clips: list[dict], cfg: dict, rng) -> tuple[list[str], dict[str, list[dict]]]:
    """Computes the exact per-participant block order + clip selection used
    by run_emotion_task. Pulled out so it can run once, before familiarization,
    letting the familiarization practice clip be reliably drawn from clips
    NOT selected here (see pick_practice_clip) instead of risking a repeat.
    """
    groups: dict[str, list[dict]] = {}
    for clip in clips:
        groups.setdefault(clip["valence_group"], []).append(clip)

    block_order = list(cfg["valence_groups"])
    rng.shuffle(block_order)

    selected_by_group: dict[str, list[dict]] = {}
    for valence_group in block_order:
        group_clips = list(groups.get(valence_group, []))
        rng.shuffle(group_clips)
        selected_by_group[valence_group] = group_clips[: cfg["clips_per_group"]]
    return block_order, selected_by_group


def pick_practice_clip(clips: list[dict], selected_by_group: dict[str, list[dict]], rng) -> dict:
    """A random clip NOT selected for the real task -- so familiarization's
    preview is a genuinely new clip, not one the participant will see (and
    have to rate for real) again a few minutes later."""
    used_ids = {clip["clip_id"] for group_clips in selected_by_group.values() for clip in group_clips}
    unused = [clip for clip in clips if clip["clip_id"] not in used_ids]
    return rng.choice(unused or clips)


def _progress_stim(win, progress_label: str | None):
    if not progress_label:
        return None
    return visual.TextStim(win, text=progress_label, height=0.03, pos=(-0.7, 0.46), color=MUTED_COLOR, alignText="left", anchorHoriz="left")


def _play_placeholder(win, clip: dict, duration: float, progress_label: str | None = None):
    color = _VALENCE_COLORS.get(clip["valence_group"], "gray40")
    rect = visual.Rect(win, width=1.2, height=0.8, fillColor=color, lineColor=None)
    label = visual.TextStim(
        win,
        text=f"[placeholder clip -- real FilmStim file not yet loaded]\n\n{clip['clip_id']}  ({clip['fine_grained_label']})",
        height=0.045,
        color="white",
    )
    progress = _progress_stim(win, progress_label)
    event.clearEvents()  # flush stale keys from the previous screen -- see fixation_cross
    clock = core.Clock()
    while clock.getTime() < duration:
        check_quit()
        check_skip_block()
        check_skip_trial()
        rect.draw()
        label.draw()
        if progress:
            progress.draw()
        draw_key_hint(win)
        win.flip()
        core.wait(0.05)



def play_clip(
    win, clip: dict, duration: float, progress_label: str | None = None, ctx=None,
    player_path: str | None = None, task_label: str = "emotion",
):
    """Plays the clip in an external player (blocking) if the file exists and
    a player was resolved; otherwise falls back to the in-window placeholder.
    Returns the (possibly new) PsychoPy Window -- callers must reassign their
    `win` reference to the return value.

    Our own window is closed before launching the player and reopened after --
    two apps each holding exclusive fullscreen on the same display at once is
    exactly what made VLC sometimes render incorrectly while our own window
    was also fullscreen (same fix, same reason, as attention_openmatb.py's
    OpenMATB subprocess: it needs sole ownership of the display while it runs).

    `duration` is only a safety timeout here (real playback runs to the
    file's actual end via VLC's --play-and-exit) plus the placeholder's
    on-screen time when there's no real file to play. `task_label` tags the
    logged events (e.g. "familiarization" for the practice clip, vs. "emotion"
    for the real task) so they're distinguishable in the event log.
    """
    file_path = Path(clip["file_path"])
    if file_path.exists() and player_path is not None:
        if ctx is not None:
            ctx.event_logger.log("clip_playback_start", task=task_label, clip_id=clip["clip_id"], file_path=str(file_path))
        window_kwargs = (ctx.window_kwargs if ctx is not None else None) or dict(size=(1280, 800), color=BG_COLOR, units="height", fullscr=False)
        win.close()
        start = time.time()
        try:
            proc = subprocess.run(
                [
                    player_path,
                    "--play-and-exit",
                    "--fullscreen",
                    "--no-video-title-show",
                    # Force a fresh, isolated process every launch. VLC's default
                    # "allow only one instance" setting otherwise hands playback off
                    # to an already-running VLC via IPC instead of the process we
                    # actually launched and are waiting on -- which is what made
                    # playback sometimes not display correctly, and sometimes never
                    # return control to us at all (we were waiting on the wrong process).
                    "--no-one-instance",
                    # Disable the "continue playback where you left off?" dialog --
                    # it blocks --play-and-exit indefinitely waiting for a click,
                    # which is the other likely cause of playback getting stuck.
                    "--qt-continue=0",
                    "--no-repeat",
                    "--no-loop",
                    "--quiet",
                    str(file_path),
                ],
                capture_output=True, text=True, timeout=duration + 45,
            )
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            returncode = -1
        win = visual.Window(**window_kwargs)
        if ctx is not None:
            ctx.event_logger.log(
                "clip_playback_end", task=task_label, clip_id=clip["clip_id"],
                returncode=returncode, wall_duration_sec=time.time() - start,
            )
        check_quit()
        return win
    _play_placeholder(win, clip, duration, progress_label)
    return win


_BLOCK_LETTERS = "ABCDEFGHIJ"


def _run_rest(win, duration: float):
    """Labeled, progress-barred rest period -- replaces a bare blank screen
    (show_message(win, "", duration=...)), which participants found confusing
    ("long blank screen... not very intuitive") since it gave no indication
    of what was happening or how much longer it would last."""
    message = visual.TextStim(win, text="Short rest.", height=0.05, pos=(0, 0.12), color=TEXT_COLOR)
    clock = core.Clock()
    while clock.getTime() < duration:
        check_quit()
        draw_progress_bar(win, (duration - clock.getTime()) / duration)
        message.draw()
        win.flip()


def run_emotion_task(win, ctx, block_order: list[str], selected_by_group: dict[str, list[dict]], player_path: str):
    """`block_order`/`selected_by_group` come from select_task_clips(), called
    once (in app/main.py, before familiarization) so the familiarization
    practice clip can reliably avoid repeating a clip used here.

    Returns the (possibly new) Window -- play_clip() closes/reopens our
    window around each external playback, so callers must reassign their
    `win` reference to the return value.
    """
    cfg = ctx.config["emotion_task"]
    fixation_duration = ctx.scaled(cfg["fixation_duration_sec"])
    rest_duration = ctx.scaled(cfg["rest_duration_sec"])
    scale_min, scale_max = cfg["rating_scale_min"], cfg["rating_scale_max"]
    dominance_enabled = cfg["dominance_enabled"]

    show_message(
        win,
        "TASK 2: EMOTION\n\n"
        "You will watch a series of short video clips, grouped into three blocks.\n\n"
        "Each clip starts with a brief '+' fixation cross, then plays with sound in its own "
        "player window. Afterwards you'll answer a couple of quick questions about how it made "
        "you feel -- whether it was positive, negative, or neutral overall, and how pleasant, "
        "arousing, and likeable it was.\n\n"
        "For every question, you can either press a number key or click the "
        "on-screen option with your mouse -- whichever is more comfortable.\n\n"
        "Some clips are lighthearted, others may be sad, tense, or unpleasant -- "
        "that's expected and part of the study.\n\n"
        "Your participation is voluntary:\n"
        "- Press N before a clip starts (or after it plays) to skip it\n"
        "- Press B to skip the rest of the current block\n"
        "- Press Esc to stop the session entirely\n\n"
        "Press SPACE to begin.",
        wait_key=["space", "return"],
        font_height=0.042,
    )

    ctx.event_logger.log("task_start", task="emotion")

    for block_index, valence_group in enumerate(block_order):
        check_quit()

        # Deliberately doesn't name the valence group (positive/negative/neutral)
        # -- telling participants what kind of clips are coming next would bias
        # their anticipatory/reported emotional response.
        show_message(
            win,
            f"VIDEO BLOCK {_BLOCK_LETTERS[block_index]}\n\n"
            f"(Block {block_index + 1} of {len(block_order)})\n\n"
            "Press SPACE to begin this block.",
            wait_key=["space", "return"],
        )

        selected = selected_by_group[valence_group]

        ctx.event_logger.log(
            "block_start", task="emotion", block_index=block_index,
            condition_label=valence_group, num_trials=len(selected),
        )

        skipped_block = False
        for trial_index, clip in enumerate(selected):
            check_quit()
            progress_label = f"Clip {trial_index + 1} of {len(selected)}"
            ctx.event_logger.log(
                "trial_start", task="emotion", block_index=block_index, trial_index=trial_index,
                condition_label=valence_group, clip_id=clip["clip_id"], fine_grained_label=clip["fine_grained_label"],
            )

            try:
                fixation_cross(win, fixation_duration, allow_skip=True, progress_label=progress_label)
                win = play_clip(
                    win, clip, ctx.scaled(clip.get("duration_sec", 90)),
                    progress_label=progress_label, ctx=ctx, player_path=player_path,
                )

                emotion_pick, emotion_rt = discrete_choice(
                    win, EMOTION_PROMPT, EMOTION_OPTIONS, allow_skip=True,
                )

                rating_items = list(cfg["rating_items"]) + (["dominance"] if dominance_enabled else [])
                rating_prompts = [
                    (item, *_RATING_PROMPTS.get(item, (f"Rate your {item.upper()}.", "not at all", "extremely")))
                    for item in rating_items
                ]
                rating_results = rating_scale_multi(win, rating_prompts, scale_min, scale_max, allow_skip=True)
                ratings: dict[str, int | None] = {item: rating_results[item][0] for item in rating_items}
                rating_rts: dict[str, float | None] = {f"{item}_rt": rating_results[item][1] for item in rating_items}

                ctx.event_logger.log(
                    "rating_response", task="emotion", block_index=block_index, trial_index=trial_index,
                    condition_label=valence_group, clip_id=clip["clip_id"],
                    emotion_pick=emotion_pick, emotion_pick_rt=emotion_rt, skipped=False,
                    **ratings, **rating_rts,
                )
            except SkipTrial:
                ctx.event_logger.log(
                    "rating_response", task="emotion", block_index=block_index, trial_index=trial_index,
                    condition_label=valence_group, clip_id=clip["clip_id"],
                    emotion_pick=None, emotion_pick_rt=None, skipped=True,
                )
                continue
            except SkipBlock:
                ctx.event_logger.log(
                    "rating_response", task="emotion", block_index=block_index, trial_index=trial_index,
                    condition_label=valence_group, clip_id=clip["clip_id"],
                    emotion_pick=None, emotion_pick_rt=None, skipped=True,
                )
                skipped_block = True
                break

            _run_rest(win, rest_duration)

        ctx.event_logger.log(
            "block_end", task="emotion", block_index=block_index, condition_label=valence_group, skipped=skipped_block,
        )

    ctx.event_logger.log("task_end", task="emotion")
    return win
