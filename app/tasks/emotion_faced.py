"""Task 2: FilmStim-based emotion induction, per Engineering_Document.md §4.2.

Clips are blocked by valence group (same-valence clips shown consecutively,
per FACED's design), with block order and within-block clip order both
shuffled per participant. Manifest-driven: swapping placeholder clips for
real FilmStim exports is a config change (emotion_manifest.json), not a
code change -- if a clip's file_path doesn't exist on disk yet, playback
falls back to a procedurally-rendered placeholder automatically.
"""

from pathlib import Path

from psychopy import core, visual

from app.ui.common_widgets import check_quit, discrete_choice, fixation_cross, rating_scale_0_7, show_message

_VALENCE_COLORS = {"positive": "seagreen", "negative": "firebrick", "neutral": "gray40"}


def _play_placeholder(win, clip: dict, duration: float):
    color = _VALENCE_COLORS.get(clip["valence_group"], "gray40")
    rect = visual.Rect(win, width=1.2, height=0.8, fillColor=color, lineColor=None)
    label = visual.TextStim(
        win,
        text=f"[placeholder clip -- real FilmStim file not yet loaded]\n\n{clip['clip_id']}  ({clip['fine_grained_label']})",
        height=0.045,
        color="white",
    )
    clock = core.Clock()
    while clock.getTime() < duration:
        check_quit()
        rect.draw()
        label.draw()
        win.flip()
        core.wait(0.05)


def play_clip(win, clip: dict, duration: float):
    file_path = Path(clip["file_path"])
    if file_path.exists():
        try:
            movie = visual.MovieStim(win, str(file_path), noAudio=False)
            clock = core.Clock()
            while clock.getTime() < duration:
                check_quit()
                movie.draw()
                win.flip()
            return
        except Exception:
            pass  # any load/decode failure -> fall through to the placeholder
    _play_placeholder(win, clip, duration)


def run_emotion_task(win, ctx, clips: list[dict]):
    cfg = ctx.config["emotion_task"]
    fixation_duration = ctx.scaled(cfg["fixation_duration_sec"])
    rest_duration = ctx.scaled(cfg["rest_duration_sec"])
    scale_min, scale_max = cfg["rating_scale_min"], cfg["rating_scale_max"]
    dominance_enabled = cfg["dominance_enabled"]

    show_message(
        win,
        "TASK 2: EMOTION\n\n"
        "You will watch a series of short video clips.\n"
        "After each clip, you'll rate your emotional reaction.\n\n"
        "Press SPACE to begin.",
        wait_key=["space", "return"],
    )

    groups: dict[str, list[dict]] = {}
    for clip in clips:
        groups.setdefault(clip["valence_group"], []).append(clip)

    block_order = list(cfg["valence_groups"])
    ctx.rng.shuffle(block_order)

    ctx.event_logger.log("task_start", task="emotion")

    for block_index, valence_group in enumerate(block_order):
        check_quit()
        group_clips = list(groups.get(valence_group, []))
        ctx.rng.shuffle(group_clips)
        selected = group_clips[: cfg["clips_per_group"]]

        fine_labels = sorted({c["fine_grained_label"] for c in selected})
        options = fine_labels + ["Other / none of these"] if valence_group != "neutral" else None

        ctx.event_logger.log(
            "block_start", task="emotion", block_index=block_index,
            condition_label=valence_group, num_trials=len(selected),
        )

        for trial_index, clip in enumerate(selected):
            check_quit()
            ctx.event_logger.log(
                "trial_start", task="emotion", block_index=block_index, trial_index=trial_index,
                condition_label=valence_group, clip_id=clip["clip_id"], fine_grained_label=clip["fine_grained_label"],
            )

            fixation_cross(win, fixation_duration)
            play_clip(win, clip, ctx.scaled(clip.get("duration_sec", 90)))

            emotion_pick, emotion_rt = None, None
            if options:
                emotion_pick, emotion_rt = discrete_choice(win, "Which emotion best matches what you felt?", options)

            ratings: dict[str, int | None] = {}
            rating_rts: dict[str, float | None] = {}
            for item in cfg["rating_items"]:
                value, rt = rating_scale_0_7(win, f"Rate your {item.upper()} while watching that clip.", scale_min, scale_max)
                ratings[item] = value
                rating_rts[f"{item}_rt"] = rt
            if dominance_enabled:
                value, rt = rating_scale_0_7(win, "Rate your DOMINANCE (sense of control) while watching that clip.", scale_min, scale_max)
                ratings["dominance"] = value
                rating_rts["dominance_rt"] = rt

            ctx.event_logger.log(
                "rating_response", task="emotion", block_index=block_index, trial_index=trial_index,
                condition_label=valence_group, clip_id=clip["clip_id"],
                emotion_pick=emotion_pick, emotion_pick_rt=emotion_rt,
                **ratings, **rating_rts,
            )

            show_message(win, "", duration=rest_duration)

        ctx.event_logger.log("block_end", task="emotion", block_index=block_index, condition_label=valence_group)

    ctx.event_logger.log("task_end", task="emotion")
