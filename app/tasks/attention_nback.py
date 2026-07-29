"""Task 1: spatial n-back (attention), per Engineering_Document.md §4.1.

Capped at 2-back deliberately -- higher n shifts the task from
attention-dominant to memory-dominant, which is the opposite of what this
task is for. 0-back uses a fixed target cell rather than "0 steps back",
since positions[i-0] == positions[i] is degenerate.
"""

from psychopy import core, event, visual

from app.ui.common_widgets import UserQuit, check_quit, fixation_cross, show_message


def build_grid(win, grid_size: int, cell_size: float = 0.14, gap: float = 0.05):
    total = grid_size * cell_size + (grid_size - 1) * gap
    start = -total / 2 + cell_size / 2
    cells = []
    for row in range(grid_size):
        for col in range(grid_size):
            x = start + col * (cell_size + gap)
            y = start + row * (cell_size + gap)
            cells.append(
                visual.Rect(win, width=cell_size, height=cell_size, pos=(x, y), fillColor="gray20", lineColor="white")
            )
    return cells


def draw_grid(win, cells, highlight_idx: int | None):
    for i, rect in enumerate(cells):
        rect.fillColor = "yellow" if i == highlight_idx else "gray20"
        rect.draw()
    win.flip()


def generate_sequence(n: int, num_trials: int, grid_size: int, target_prob: float, rng):
    """Returns (positions, is_target, fixed_target_or_None)."""
    num_positions = grid_size * grid_size
    positions: list[int] = []
    is_target: list[bool] = []

    if n == 0:
        fixed_target = num_positions // 2
        for _ in range(num_trials):
            make_target = rng.random() < target_prob
            if make_target:
                pos = fixed_target
            else:
                pos = rng.randrange(num_positions)
                while pos == fixed_target:
                    pos = rng.randrange(num_positions)
            positions.append(pos)
            is_target.append(make_target)
        return positions, is_target, fixed_target

    for i in range(num_trials):
        if i < n:
            positions.append(rng.randrange(num_positions))
            is_target.append(False)  # no valid n-back reference yet, not scorable
            continue
        ref = positions[i - n]
        make_target = rng.random() < target_prob
        if make_target:
            pos = ref
        else:
            pos = rng.randrange(num_positions)
            while pos == ref:
                pos = rng.randrange(num_positions)
        positions.append(pos)
        is_target.append(make_target)
    return positions, is_target, None


def run_trial(win, cells, pos: int, soa: float, stim_on_sec: float):
    trial_clock = core.Clock()
    draw_grid(win, cells, pos)
    event.clearEvents()
    stim_off_done = False
    response_key = None
    rt = None
    while trial_clock.getTime() < soa:
        t = trial_clock.getTime()
        if not stim_off_done and t >= stim_on_sec:
            draw_grid(win, cells, None)
            stim_off_done = True
        for key, kt in event.getKeys(keyList=["space", "escape"], timeStamped=trial_clock):
            if key == "escape":
                raise UserQuit()
            if response_key is None:
                response_key, rt = key, kt
        core.wait(0.005)
    return response_key, rt


def _run_block(win, ctx, cells, n, positions, is_target, block_index, task_label, soa, stim_on_sec):
    ctx.event_logger.log(
        "block_start", task=task_label, block_index=block_index,
        condition_label=f"{n}-back", num_trials=len(positions),
    )
    correct_count = 0
    for trial_idx, (pos, target) in enumerate(zip(positions, is_target)):
        check_quit()
        ctx.event_logger.log(
            "trial_start", task=task_label, block_index=block_index, trial_index=trial_idx,
            condition_label=f"{n}-back", position=pos, is_target=target,
        )
        response_key, rt = run_trial(win, cells, pos, soa, stim_on_sec)
        responded = response_key is not None
        correct = responded == target
        correct_count += int(correct)
        ctx.event_logger.log(
            "response", task=task_label, block_index=block_index, trial_index=trial_idx,
            condition_label=f"{n}-back", position=pos, is_target=target,
            responded=responded, rt=rt, correct=correct,
        )
    ctx.event_logger.log(
        "block_end", task=task_label, block_index=block_index, condition_label=f"{n}-back",
        accuracy=correct_count / len(positions) if positions else None,
    )


def run_attention_task(win, ctx):
    cfg = ctx.config["attention_task"]
    grid_size = cfg["grid_size"]
    conditions = cfg["conditions"]
    trials_per_condition = cfg["trials_per_condition"]
    soa = cfg["soa_sec"]
    block_duration = ctx.scaled(cfg["block_duration_sec"])
    cue_duration = ctx.scaled(cfg["cue_duration_sec"])
    rest_duration = ctx.scaled(cfg["rest_duration_sec"])
    target_prob = cfg["target_prob"]
    stim_on_sec = min(0.6, soa * 0.3)

    show_message(
        win,
        "TASK 1: ATTENTION\n\n"
        "Watch the grid. One cell lights up at a time.\n\n"
        "For 1-back and 2-back rounds: press SPACE as soon as the lit cell repeats "
        "from N steps back (you'll be told which N before each round).\n\n"
        "For 0-back rounds: press SPACE whenever the CENTER cell lights up.\n\n"
        "Press SPACE to begin.",
        wait_key=["space", "return"],
    )

    cells = build_grid(win, grid_size)
    ctx.event_logger.log("task_start", task="attention")

    # Untimed practice block (1-back), not part of the 24-trial dataset.
    show_message(win, "Practice round (not scored) -- 1-back.\n\nPress SPACE to begin.", wait_key=["space", "return"])
    practice_n = 1
    practice_positions, practice_targets, _ = generate_sequence(
        practice_n, cfg["practice_trials"], grid_size, target_prob, ctx.rng
    )
    _run_block(win, ctx, cells, practice_n, practice_positions, practice_targets, -1, "attention_practice", soa, stim_on_sec)

    block_plan = list(conditions) * trials_per_condition
    ctx.rng.shuffle(block_plan)

    for block_index, n in enumerate(block_plan):
        check_quit()
        fixation_cross(win, ctx.scaled(1.0))
        show_message(win, f"Get ready: {n}-back", duration=cue_duration)
        num_trials = max(1, int(block_duration / soa))
        positions, is_target, _ = generate_sequence(n, num_trials, grid_size, target_prob, ctx.rng)
        _run_block(win, ctx, cells, n, positions, is_target, block_index, "attention", soa, stim_on_sec)
        show_message(win, "Rest", duration=rest_duration)

    ctx.event_logger.log("task_end", task="attention")
