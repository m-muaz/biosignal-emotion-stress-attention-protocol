"""Task: SART (Sustained Attention to Response Task, Robertson et al. 1997)
-- participants press SPACE (or any key) for every digit 1-9 EXCEPT the
omit number(s) (default: 3), and withhold on those.

Ported 2026-08-05 from the standalone tryout at
C:\\Users\\UT_Wireless\\Documents\\attention-task-tryouts\\sart_cstothart\\
(python_sart.py/sart_config.py, Cary Stothart, MIT license -- see
ATTRIBUTION at the bottom) into this project's ctx/event_logger
conventions, per PI request 2026-08-05. Deliberately NOT a full rewrite --
the core trial-timing algorithm (250ms digit + 900ms mask, fixed 1.15s/
trial SOA, 18-trial practice block with feedback, real block = reps x 45
trials) is unchanged from the original. What changed, and why:

  - gui.Dlg participant-info popup + manual tab-separated .txt output ->
    ctx.event_logger (task_start/task_end, block_start/block_end,
    baseline_start/baseline_end, trial_start, response events -- the same
    events.jsonl every other task in this project writes to). Matches this
    project's actual convention (participant_id already comes from
    --participant-id, not an in-task popup) rather than the original
    script's one-off intake dialog.
  - No Escape-to-quit ANYWHERE in the original script (confirmed: the
    README's old SART section said so explicitly, and there's a
    participant-reported bug to match -- "no way to quit it while
    playing"). Added via app/ui/common_widgets.py's check_quit()/UserQuit,
    same mechanism every other task in this project uses. See _run_trial's
    escape handling below for why it's folded into the existing response
    keypress read rather than a second event.getKeys() call (a second call
    would drain the buffer and silently eat a real space-bar response).
  - Instruction/"press b to continue" screens -> common_widgets.
    show_message()/fixation_cross(), which already have Escape handling
    built in, instead of hand-rolled `while 'b' not in event.getKeys()`
    loops.
  - Added a pre-task physiological baseline period (fixation cross,
    sart_task.baseline_duration_sec) -- not in the original at all -- same
    distributed-baseline treatment as every other task (see
    session_config.yaml's sart_task comment, [[project_session_timing_budget_2026-08-05]]).
    Per PI request 2026-08-12, this baseline now also shows a low-key mm:ss
    countdown (fixation_cross's show_timer, common_widgets.py) so the
    participant has feedback that the wait is progressing -- mirrors the
    web-based emotion/stress players' baseline countdown (player.js's
    runFixation/formatCountdown), which this task's psychopy window doesn't
    otherwise share any UI code with.
  - Simplified the original's trial-sequence construction: its non-fixed
    path effectively only used reps + PsychoPy's own TrialHandler(...,
    method='random') to reshuffle the 45-trial (9 numbers x N font sizes)
    factorial list each rep -- the elaborate seqList-building loop in the
    original only mattered for its fixed=True path. Reimplemented as a
    plain shuffle-per-rep using ctx.rng (so this draw is covered by the
    one rng_seed already recorded in session_manifest.json, instead of the
    original's unseeded stdlib `random` module).

Per PI request 2026-08-05, this stays a SEPARATE standalone task -- run via
`python -m app.run_task --task sart`, NOT wired into session_shell.py/
app/main.py's session flow (unlike Schulte/Stroop) -- see README's
"Attention task (SART)" section and [[project_attention_task_sart]].

ATTRIBUTION: original algorithm/timing/instruction wording by Cary Stothart
(cary.stothart@gmail.com), MIT License --
https://github.com/cstothart/sustained-attention-to-response-task.
Citation: Stothart, C. (2015). Python SART (Version 2) [software].
"""

from __future__ import annotations

from psychopy import core, event, visual

from app.ui.common_widgets import TEXT_COLOR, UserQuit, check_quit, fixation_cross, show_message

_NUMBERS = list(range(1, 10))
# The original script's window used units='cm' with monitor="testMonitor"
# (sart_cstothart/python_sart.py:142-143) -- PsychoPy's built-in default
# calibration (30cm wide, 1024x768px, see `psychopy.monitors.Monitor
# ("testMonitor").getWidth()/.getSizePix()`), which renders 'cm' sizes as a
# FIXED absolute pixel count (1024/30 = 34.13 px/cm) independent of the
# actual runtime screen resolution. A first attempt at this port approximated
# that ratio as a fraction of this project's units="height" window (1.0 ==
# full window height, see app/run_task.py's window_kwargs) -- but "height"
# units scale UP with actual screen resolution while the original's
# calibrated-cm sizing does not, so on any screen taller than the ~800px that
# approximation assumed, digits rendered proportionally bigger than the
# original ever did (reported: still too big, worse on higher-res screens).
# Also caught in the same pass: circle_stim's pos=(0, -0.2) -- a barely
# visible 2mm nudge in the original's cm units -- was rendering as 20% of the
# window height off-center under "height" units; and correct_stim/
# incorrect_stim had no explicit height, silently defaulting to psychopy's
# own units="height" default (0.2 = 20% of window height) instead of the
# original's units='cm' default (1.0cm, ~34px).
#
# Fixed by giving every SART-specific stim (not the shared window) an
# explicit units='pix' override with the exact pixel-equivalent of the
# original's cm values via the same testMonitor ratio -- this reproduces the
# original's absolute on-screen rendering exactly, independent of this
# project's window-level units="height" default (which stays in place for
# common_widgets' show_message/fixation_cross layout, untouched here).
# 'cm' units themselves aren't usable directly: this project's window is
# created with no `monitor=` kwarg (app/run_task.py), and PsychoPy raises
# ValueError ("Monitor __blank__ has no known size in pixels") for 'cm'-unit
# stims without one.
_CM_TO_PIX = 1024 / 30  # testMonitor's calibration ratio, ~34.13 px/cm
_REAL_FONT_SIZES = [round(cm * _CM_TO_PIX, 2) for cm in (1.20, 1.80, 2.35, 2.50, 3.00)]
_PRACTICE_FONT_SIZES = [round(cm * _CM_TO_PIX, 2) for cm in (1.20, 3.00)]
_PRACTICE_REPS = 1  # 1 rep x 9 numbers x 2 font sizes = 18 practice trials, matches the original


def _resolve_omit_numbers(rng, omit_num, omit_count) -> list[int]:
    """Mirrors the original's resolve_omit_numbers() -- either a hardcoded
    omit_num (int or list, sart_task.omit_num), or omit_count numbers chosen
    at random from 1-9 (sart_task.omit_count, takes priority if set). Uses
    ctx.rng (not the original's unseeded stdlib `random`) so this draw is
    covered by the session's one recorded rng_seed."""
    if omit_count is not None:
        if omit_count < 1 or omit_count >= len(_NUMBERS):
            raise ValueError(f"sart_task.omit_count must be between 1 and {len(_NUMBERS) - 1}.")
        return sorted(rng.sample(_NUMBERS, omit_count))
    if isinstance(omit_num, (list, tuple, set)):
        return sorted(set(omit_num))
    return [omit_num]


def _format_number_list(numbers: list[int]) -> str:
    strs = [str(n) for n in numbers]
    if len(strs) == 1:
        return strs[0]
    if len(strs) == 2:
        return f"{strs[0]} or {strs[1]}"
    return ", ".join(strs[:-1]) + f", or {strs[-1]}"


def _build_trial_sequence(rng, font_sizes: list[float], reps: int, fixed_order: bool) -> list[tuple[int, float]]:
    """Returns a flat list of (number, font_size) -- reps independent passes
    through the full 9-numbers x len(font_sizes) factorial set. Random order
    (fixed_order=False, this project's default) reshuffles each pass
    independently; fixed order repeats the same (font_size-major) ordering
    every pass."""
    base = [(n, f) for f in font_sizes for n in _NUMBERS]
    sequence = []
    for _ in range(reps):
        rep_trials = list(base)
        if not fixed_order:
            rng.shuffle(rep_trials)
        sequence.extend(rep_trials)
    return sequence


def _run_trial(
    win, ctx, x_stim, circle_stim, num_stim, correct_stim, incorrect_stim, clock,
    number: int, font_size: float, omit_nums: list[int], feedback: bool,
    block_index: int, trial_index: int, practice: bool,
) -> None:
    is_omit = number in omit_nums
    ctx.event_logger.log(
        "trial_start", task="sart", block_index=block_index, trial_index=trial_index,
        practice=practice, number=number, font_size=font_size, is_omit=is_omit, omit_numbers=omit_nums,
    )

    check_quit()
    num_stim.setHeight(font_size)
    num_stim.setText(str(number))
    num_stim.draw()
    event.clearEvents()
    clock.reset()

    # Fixed 1.15s SOA (250ms digit + 900ms mask), unchanged from the
    # original -- deliberately NOT demo-scaled, same reasoning
    # stress_task/attention_highway_task's own per-trial pacing isn't
    # scaled either (scaling interactive per-event timing would make the
    # task either trivial or literally impossible for a human to play).
    stim_start = core.getTime()
    win.flip()
    x_stim.draw()
    circle_stim.draw()
    wait_time = 0.25 - (core.getTime() - stim_start)
    if wait_time > 0:
        core.wait(wait_time, hogCPUperiod=wait_time)
    mask_start = core.getTime()
    win.flip()
    wait_time = 0.90 - (core.getTime() - mask_start)
    if wait_time > 0:
        core.wait(wait_time, hogCPUperiod=wait_time)
    win.flip()

    # ONE read of the key buffer, not two -- event.getKeys() drains
    # whatever's queued, so a second call (e.g. a separate check_quit()
    # here) would see nothing and silently swallow a real space-bar
    # response. Escape is checked against THIS same read instead.
    all_keys = event.getKeys(timeStamped=clock)
    if any(key == "escape" for key, _rt in all_keys):
        raise UserQuit()
    responded = len(all_keys) > 0
    rt = all_keys[0][1] if responded else None
    # Correct = pressed on a non-omit number, OR withheld on an omit number.
    accurate = (responded and not is_omit) or (not responded and is_omit)

    ctx.event_logger.log(
        "response", task="sart", block_index=block_index, trial_index=trial_index,
        practice=practice, number=number, is_omit=is_omit, responded=responded, rt=rt, accurate=accurate,
    )

    if feedback:
        (correct_stim if accurate else incorrect_stim).draw()
        fb_start = core.getTime()
        win.flip()
        wait_time = 0.90 - (core.getTime() - fb_start)
        if wait_time > 0:
            core.wait(wait_time, hogCPUperiod=wait_time)
        win.flip()

    check_quit()


def _run_block(
    win, ctx, omit_nums: list[int], reps: int, font_sizes: list[float], fixed_order: bool,
    feedback: bool, block_index: int, practice: bool,
) -> None:
    sequence = _build_trial_sequence(ctx.rng, font_sizes, reps, fixed_order)
    ctx.event_logger.log(
        "block_start", task="sart", block_index=block_index, practice=practice, num_trials=len(sequence),
    )

    # units='pix' on every stim below -- see _CM_TO_PIX's comment for why:
    # the shared window stays units="height" for common_widgets, these
    # override it locally to reproduce the original script's absolute
    # (resolution-independent, calibration-based) pixel sizing.
    x_stim = visual.TextStim(win, text="X", height=3.35 * _CM_TO_PIX, color=TEXT_COLOR, units="pix")
    circle_stim = visual.Circle(
        win, radius=1.50 * _CM_TO_PIX, lineWidth=8, lineColor=TEXT_COLOR,
        pos=(0, -0.2 * _CM_TO_PIX), units="pix",
    )
    num_stim = visual.TextStim(win, font="Arial", color=TEXT_COLOR, pos=(0, 0), units="pix")
    # height=1.0cm equivalent -- the original never set an explicit height on
    # these either, but its units='cm' default (1.0cm) differs from what
    # these would silently default to here (units='pix' defaults to 20px,
    # units="height" -- this project's window default -- to 0.2, i.e. 20% of
    # the window height). Set explicitly so it isn't left to chance either way.
    correct_stim = visual.TextStim(win, text="CORRECT", color="green", font="Arial", pos=(0, 0), units="pix", height=1.0 * _CM_TO_PIX)
    incorrect_stim = visual.TextStim(win, text="INCORRECT", color="red", font="Arial", pos=(0, 0), units="pix", height=1.0 * _CM_TO_PIX)
    clock = core.Clock()

    try:
        for trial_index, (number, font_size) in enumerate(sequence):
            _run_trial(
                win, ctx, x_stim, circle_stim, num_stim, correct_stim, incorrect_stim, clock,
                number, font_size, omit_nums, feedback, block_index, trial_index, practice,
            )
    finally:
        # Logged even on a mid-block UserQuit -- so it's visible in
        # events.jsonl that this block ended early, not just silently cut off.
        ctx.event_logger.log("block_end", task="sart", block_index=block_index, practice=practice)


def run_sart_task(win, ctx) -> None:
    cfg = ctx.config["sart_task"]
    omit_nums = _resolve_omit_numbers(ctx.rng, cfg.get("omit_num"), cfg.get("omit_count"))
    fixed_order = cfg.get("fixed_order", False)
    number_word = "number" if len(omit_nums) == 1 else "numbers"
    omit_text = _format_number_list(omit_nums)

    show_message(
        win,
        "TASK: SART (SUSTAINED ATTENTION)\n\n"
        f"A series of numbers will be presented to you. For every number EXCEPT "
        f"the {number_word} {omit_text}, press the SPACE bar as quickly as you "
        f"can. If you see the {number_word} {omit_text}, do NOT press the space "
        "bar or any other key.\n\n"
        "Please give equal importance to both accuracy and speed.\n\n"
        "Press SPACE to begin.",
        wait_key=["space", "return"],
    )

    # Pre-task physiological baseline -- not in the original script at all,
    # added per PI request 2026-08-05 (see module docstring).
    ctx.event_logger.log(
        "baseline_start", task="sart", position="pre_task", duration_sec=cfg["baseline_duration_sec"],
    )
    fixation_cross(win, ctx.scaled(cfg["baseline_duration_sec"]), progress_label="Baseline recording", show_timer=True)
    ctx.event_logger.log("baseline_end", task="sart", position="pre_task")

    ctx.event_logger.log("task_start", task="sart", omit_numbers=omit_nums, reps=cfg["reps"], fixed_order=fixed_order)

    if cfg.get("practice", True):
        show_message(
            win,
            "We will now do some practice trials to familiarize you with the task.\n\n"
            f"Remember, press the space bar when you see any number except the "
            f"{number_word} {omit_text}.\n\nPress SPACE to start the practice.",
            wait_key=["space", "return"],
        )
        _run_block(
            win, ctx, omit_nums, reps=_PRACTICE_REPS, font_sizes=_PRACTICE_FONT_SIZES,
            fixed_order=fixed_order, feedback=True, block_index=0, practice=True,
        )

    show_message(
        win,
        "We will now start the actual task.\n\n"
        "Remember, give equal importance to both accuracy and speed.\n\n"
        "Press SPACE to start the actual task.",
        wait_key=["space", "return"],
    )
    _run_block(
        win, ctx, omit_nums, reps=cfg["reps"], font_sizes=_REAL_FONT_SIZES,
        fixed_order=fixed_order, feedback=False, block_index=1, practice=False,
    )

    ctx.event_logger.log("task_end", task="sart")
