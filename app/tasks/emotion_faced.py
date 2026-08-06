"""Task 2: video-based emotion induction, per Engineering_Document.md §4.2.

Block layout is config-driven (emotion_task.block_structure): by default
("interleaved", per PI request 2026-08-02) each block mixes valence groups,
with clip order fully randomized across the pooled selection. The original
FACED-style layout -- clips blocked by valence group, same-valence clips
shown consecutively -- is still available via block_structure:
grouped_by_valence. Clip selection is likewise config-driven
(emotion_task.clip_selection_mode): by default ("fixed") the same predefined
clip_ids (scripts/select_fixed_emotion_clips.py) are shown to every
participant, with only presentation order randomized; "random" restores the
old per-participant random sampling from the full manifest pool. See
select_task_clips() for details. Manifest-driven: swapping the clip set
(currently OpenLAV, previously FilmStim) is a config change
(emotion_manifest.json), not a code change -- if a clip's file_path doesn't
exist on disk yet, playback falls back to a procedurally-rendered
placeholder automatically.

Clips are played in an external video player (VLC) rather than PsychoPy's
own MovieStim -- MovieStim's frame-by-frame draw loop had recurring
decode-stall and early-cutoff bugs (a clip's actual decodable length
sometimes ran short of the manifest's probed duration_sec, and the
sdl2/ffpyplayer audio backend could stall a draw() call for tens of seconds
with no exception). VLC owns real playback; we just launch it and wait for
it to finish. Trade-off: skip-clip/skip-block keys don't work *during*
playback anymore (VLC, not our event loop, has focus) -- only between clips.

play_clip()'s `transition_mode` controls how VLC playback is driven (see
play_clip's docstring and app/tasks/vlc_rc_player.py for the full story):
"persistent" (default) reuses one long-lived VLC process across every clip,
driven via its RC interface, with automatic fallback to "hide" for the rest
of the session on any error; "hide" spawns a fresh VLC process per clip but
just hides (not closes) our own window meanwhile; "close_reopen" is the
original behavior (fully closes/recreates our window per clip too).
"""

import shutil
import subprocess
import time
from pathlib import Path

from psychopy import core, event, visual

from app.tasks.vlc_rc_player import PersistentVlcSession, VlcSessionError
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
RATING_PROMPTS = {
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


def _chunk_evenly(items: list, num_chunks: int) -> list[list]:
    """Splits `items` into `num_chunks` chunks as evenly as possible (any
    remainder distributed one-per-chunk to the earliest chunks), preserving
    `items`' existing order."""
    n = len(items)
    base, remainder = divmod(n, num_chunks)
    chunks, start = [], 0
    for i in range(num_chunks):
        size = base + (1 if i < remainder else 0)
        chunks.append(items[start : start + size])
        start += size
    return chunks


def select_task_clips(
    clips: list[dict], cfg: dict, rng, fixed_clips: dict[str, list[str]] | None = None,
) -> tuple[list[list[dict]], dict[str, list[dict]]]:
    """Computes the exact per-participant clip selection + block layout used
    by run_emotion_task. Pulled out so it can run once, before familiarization,
    letting the familiarization practice clip be reliably drawn from clips
    NOT selected here (see pick_practice_clip) instead of risking a repeat.

    `selected_by_group` is always {valence_group: [clips shown this session]}.
    `clip_selection_mode` (cfg) controls how those clips are chosen:
    - "fixed" (default): the same predefined clip_ids (`fixed_clips`, from
      scripts/select_fixed_emotion_clips.py's output, or hand-picked via
      scripts/add_manual_clips.py) are used for every participant -- only
      presentation order is randomized. Falls back to random sampling for
      any valence_group `fixed_clips` doesn't cover.
    - "random": the old behavior -- clips_per_group clips are randomly
      sampled per participant from the full manifest pool of that group.

    `clips_per_group` (cfg) is how many clips to select for each valence
    group. Either a single int applied to every group (legacy), or a dict
    ({valence_group: count}, e.g. {"positive": 10, "negative": 10,
    "neutral": 10}) so each group's count can be set independently --
    e.g. to draw down a group that participants report as too intense/long
    without touching the others.

    `block_structure` (cfg) controls how the selected clips are laid out into
    the blocks returned as `blocks` (a list of blocks, each a list of clips in
    presentation order):
    - "interleaved" (default): each valence group's selected clips are
      distributed round-robin across `num_blocks` (cfg) blocks (each group's
      clips already shuffled above, so which specific clips land in which
      block is still randomized), then every block is shuffled internally --
      so a block mixes positive/negative/neutral clips in random order, per
      PI request, while keeping roughly equal per-valence counts in every
      block (not just an equal-size pool split, which could by chance skew
      a block toward one valence). The round-robin start position rotates
      one valence group at a time so the "extra" clip from an uneven
      division (e.g. 10 clips / 3 blocks = 4/3/3) doesn't stack onto the
      same block every group -- with equal clips_per_group across groups
      this makes block sizes come out equal too. `num_blocks` defaults to
      len(valence_groups) if unset, but is independent of it -- e.g. 30
      selected clips (10/group) can be laid out as 3 blocks of 10, 5 of 6,
      6 of 5, etc.
    - "grouped_by_valence": the original FACED-style layout -- one block per
      valence group, each block's clips all sharing that group's valence
      (block order across groups is still shuffled). `num_blocks` is not
      used in this mode -- block count is always len(valence_groups).
    """
    groups: dict[str, list[dict]] = {}
    for clip in clips:
        groups.setdefault(clip["valence_group"], []).append(clip)

    selection_mode = cfg.get("clip_selection_mode", "random")
    n_per_group_cfg = cfg["clips_per_group"]

    selected_by_group: dict[str, list[dict]] = {}
    for valence_group in cfg["valence_groups"]:
        n_per_group = (
            n_per_group_cfg[valence_group] if isinstance(n_per_group_cfg, dict) else n_per_group_cfg
        )
        group_clips = list(groups.get(valence_group, []))
        fixed_ids = (fixed_clips or {}).get(valence_group)
        if selection_mode == "fixed" and fixed_ids:
            wanted = set(fixed_ids[:n_per_group])
            chosen = [clip for clip in group_clips if clip["clip_id"] in wanted]
        else:
            rng.shuffle(group_clips)
            chosen = group_clips[:n_per_group]
        rng.shuffle(chosen)  # presentation order is always randomized, even for a fixed clip set
        selected_by_group[valence_group] = chosen

    block_structure = cfg.get("block_structure", "grouped_by_valence")
    if block_structure == "interleaved":
        num_blocks = cfg.get("num_blocks", len(cfg["valence_groups"]))
        blocks: list[list[dict]] = [[] for _ in range(num_blocks)]
        for group_index, group_clips in enumerate(selected_by_group.values()):
            # Round-robin each group's (already-shuffled) clips across blocks,
            # rotating the start offset per group so a group's "extra" clip
            # from an uneven division doesn't always land in block 0 -- see
            # docstring above for why this balances both per-valence counts
            # and overall block size.
            for i, clip in enumerate(group_clips):
                blocks[(i + group_index) % num_blocks].append(clip)
        for block in blocks:
            rng.shuffle(block)  # mix valence groups' clips within the block, not group-then-group
    else:
        block_order = list(cfg["valence_groups"])
        rng.shuffle(block_order)
        blocks = [selected_by_group[valence_group] for valence_group in block_order]
    return blocks, selected_by_group


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



# Module-level so run_familiarization's single practice clip and
# run_emotion_task's whole clip loop can share (and amortize the ~0.5s
# startup cost of) the same persistent VLC process -- see
# close_persistent_session(), which must be called once the task is done.
_persistent_session: PersistentVlcSession | None = None
_persistent_disabled = False  # set True after any persistent-mode failure so later clips this run don't keep retrying a broken session


def close_persistent_session() -> None:
    """Stops the persistent VLC session, if one was ever started. Safe/no-op
    to call even if "persistent" mode was never used or already failed --
    callers (app/main.py, app/run_task.py) should call this once after the
    emotion task (and familiarization, if it played the practice clip
    first) are both done, including on the error path, so a leftover
    vlc.exe doesn't linger past the session."""
    global _persistent_session, _persistent_disabled
    if _persistent_session is not None:
        _persistent_session.stop()
        _persistent_session = None
    _persistent_disabled = False


def play_clip(
    win, clip: dict, duration: float, progress_label: str | None = None, ctx=None,
    player_path: str | None = None, task_label: str = "emotion", transition_mode: str = "persistent",
):
    """Plays the clip in an external player (blocking) if the file exists and
    a player was resolved; otherwise falls back to the in-window placeholder.
    Returns the (possibly new) PsychoPy Window -- callers must reassign their
    `win` reference to the return value.

    `transition_mode` controls how VLC playback is driven:
    - "persistent" (default): reuses one long-lived VLC process (see
      app/tasks/vlc_rc_player.py) across every clip this session, driven via
      its RC interface instead of spawning a fresh vlc.exe per clip -- this
      removes the per-clip process-spawn + decoder-init + fullscreen-entry
      cost, which is the main remaining source of flicker once "hide" (below)
      is already in place. On ANY error (VLC fails to start, dies mid-play,
      RC socket breaks, etc.) this automatically and permanently falls back
      to "hide" for the rest of the session -- no participant-visible crash,
      just reduced smoothness for the remaining clips.
    - "hide": spawns a fresh VLC process per clip (--play-and-exit), but just
      hides our own window meanwhile (win.winHandle.set_visible(False) -- a
      pyglet "fullscreen" window on Windows is a borderless topmost window,
      not real exclusive-mode fullscreen, so hiding it fully releases the
      topmost/foreground spot to VLC without the cost of tearing down and
      recreating the GL context/canvas). Restoring it after
      (set_visible(True)) is near-instant, versus visibly recreating the
      whole window.
    - "close_reopen": the original behavior -- fully win.close()s and
      recreates a new visual.Window() around each clip, on top of spawning a
      fresh VLC process per clip. Kept as a last-resort fallback
      (emotion_task.clip_transition_mode in session_config.yaml, or
      --emotion-clip-transition in app/run_task.py) in case even "hide" ever
      resurfaces the VLC rendering glitches that motivated closing the
      window in the first place (same reason attention_openmatb.py's
      OpenMATB subprocess needs sole ownership of the display while it runs).

    `duration` is only a safety timeout here (real playback runs to the
    file's actual end) plus the placeholder's on-screen time when there's no
    real file to play. `task_label` tags the logged events (e.g.
    "familiarization" for the practice clip, vs. "emotion" for the real
    task) so they're distinguishable in the event log.
    """
    global _persistent_session, _persistent_disabled
    file_path = Path(clip["file_path"])
    if file_path.exists() and player_path is not None:
        if ctx is not None:
            ctx.event_logger.log("clip_playback_start", task=task_label, clip_id=clip["clip_id"], file_path=str(file_path))
        window_kwargs = (ctx.window_kwargs if ctx is not None else None) or dict(size=(1280, 800), color=BG_COLOR, units="height", fullscr=False)

        effective_mode = transition_mode
        if effective_mode == "persistent" and _persistent_disabled:
            effective_mode = "hide"  # already failed once this session -- don't keep retrying a broken VLC session

        if effective_mode == "persistent":
            win.winHandle.set_visible(False)
            start = time.time()
            try:
                if _persistent_session is None:
                    _persistent_session = PersistentVlcSession()
                    _persistent_session.start(player_path)
                _persistent_session.play(file_path, timeout_sec=duration + 45)
                win.winHandle.set_visible(True)
                if ctx is not None:
                    ctx.event_logger.log(
                        "clip_playback_end", task=task_label, clip_id=clip["clip_id"],
                        returncode=0, wall_duration_sec=time.time() - start,
                    )
                check_quit()
                return win
            except VlcSessionError as exc:
                if ctx is not None:
                    ctx.event_logger.log(
                        "clip_playback_error", task=task_label, clip_id=clip["clip_id"], error=str(exc),
                    )
                if _persistent_session is not None:
                    _persistent_session.stop()
                    _persistent_session = None
                _persistent_disabled = True
                # win is already hidden (not closed) above -- "hide" is the
                # correct fallback entry point for this clip and every one
                # after it this session (persistent won't be retried, see
                # _persistent_disabled check above).
                effective_mode = "hide"

        if effective_mode == "hide":
            win.winHandle.set_visible(False)
        else:
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
        if effective_mode == "hide":
            win.winHandle.set_visible(True)
        else:
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


def run_emotion_task(win, ctx, blocks: list[list[dict]], player_path: str):
    """`blocks` comes from select_task_clips(), called once (in app/main.py,
    before familiarization) so the familiarization practice clip can reliably
    avoid repeating a clip used here. Each block is a list of clips in
    presentation order -- with block_structure: interleaved (default) a
    block can mix valence groups, so the per-trial valence_group is read off
    each clip rather than assumed constant for the whole block.

    Returns the (possibly new) Window -- play_clip() closes/reopens our
    window around each external playback, so callers must reassign their
    `win` reference to the return value.
    """
    cfg = ctx.config["emotion_task"]
    fixation_duration = ctx.scaled(cfg["fixation_duration_sec"])
    rest_duration = ctx.scaled(cfg["rest_duration_sec"])
    scale_min, scale_max = cfg["rating_scale_min"], cfg["rating_scale_max"]
    dominance_enabled = cfg["dominance_enabled"]
    transition_mode = cfg.get("clip_transition_mode", "persistent")

    show_message(
        win,
        "TASK 2: EMOTION\n\n"
        f"You will watch a series of short video clips, grouped into {len(blocks)} blocks.\n\n"
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

    for block_index, selected in enumerate(blocks):
        check_quit()

        # Deliberately doesn't name the valence group(s) in this block
        # (positive/negative/neutral) -- telling participants what kind of
        # clips are coming next would bias their anticipatory/reported
        # emotional response.
        show_message(
            win,
            f"VIDEO BLOCK {_BLOCK_LETTERS[block_index]}\n\n"
            f"(Block {block_index + 1} of {len(blocks)})\n\n"
            "Press SPACE to begin this block.",
            wait_key=["space", "return"],
        )

        condition_labels = sorted({clip["valence_group"] for clip in selected})
        ctx.event_logger.log(
            "block_start", task="emotion", block_index=block_index,
            condition_labels=condition_labels, num_trials=len(selected),
        )

        skipped_block = False
        for trial_index, clip in enumerate(selected):
            check_quit()
            valence_group = clip["valence_group"]
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
                    transition_mode=transition_mode,
                )

                emotion_pick, emotion_rt = discrete_choice(
                    win, EMOTION_PROMPT, EMOTION_OPTIONS, allow_skip=True, require_confirm=True,
                )

                rating_items = list(cfg["rating_items"]) + (["dominance"] if dominance_enabled else [])
                rating_prompts = [
                    (item, *RATING_PROMPTS.get(item, (f"Rate your {item.upper()}.", "not at all", "extremely")))
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
            "block_end", task="emotion", block_index=block_index, condition_labels=condition_labels, skipped=skipped_block,
        )

    ctx.event_logger.log("task_end", task="emotion")
    return win
