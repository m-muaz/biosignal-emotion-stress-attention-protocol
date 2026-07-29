"""Task 3: tiered MAT-style mental arithmetic (stress), per Engineering_Document.md §4.3.

Baseline/arithmetic block *lengths* are scaled for demo runs (ctx.scaled), but
the per-question time limit is never scaled -- shortening the time-pressure
window would defeat the point of a demo run that's supposed to let a human
operator actually feel what the task is like.

Tier is logged on the backend only and never shown on screen, so post-hoc
analysis can either collapse all tiers into a binary stress/no-stress label
or treat them as ordinal stress-intensity levels.
"""

from psychopy import core, event, visual

from app.ui.common_widgets import UserQuit, check_quit, draw_countdown_bar, draw_leaderboard, show_message


def generate_question(tier_cfg: dict, rng) -> tuple[str, int]:
    """Builds a left-to-right arithmetic expression with an exact integer answer.

    Division is kept exact by padding the running value up to the nearest
    multiple of the chosen divisor with an extra '+' step when needed --
    every token appended always matches the running `value`, so the
    displayed expression and the logged correct answer can never drift
    apart.
    """
    ops = tier_cfg["operators"]
    n = tier_cfg["num_operands"]
    max_val = 10 ** tier_cfg["max_digits"] - 1

    first = rng.randint(1, max_val)
    tokens = [str(first)]
    value = first

    for _ in range(n - 1):
        op = rng.choice(ops)
        if op == "/":
            divisor = rng.choice([2, 3, 4, 5])
            if value == 0:
                operand, new_value = divisor, 0
            else:
                remainder = value % divisor
                if remainder != 0:
                    pad = divisor - remainder
                    tokens += ["+", str(pad)]
                    value += pad
                operand, new_value = divisor, value // divisor
            tokens += ["/", str(operand)]
            value = new_value
        elif op == "*":
            operand = rng.randint(2, 9)
            tokens += ["*", str(operand)]
            value *= operand
        elif op == "+":
            operand = rng.randint(1, max_val)
            tokens += ["+", str(operand)]
            value += operand
        else:  # "-"
            operand = rng.randint(1, max_val)
            tokens += ["-", str(operand)]
            value -= operand

    return " ".join(tokens), value


def get_numeric_answer(win, expr_str, question_num, time_limit, leaderboard_names, leaderboard_scores, hurry_fraction):
    """Renders the countdown/leaderboard/hurry-up UI and collects a typed answer.

    Returns (answer:int|None, rt:float|None, raw_buffer:str, timed_out:bool).
    """
    clock = core.Clock()
    buffer = ""
    event.clearEvents()

    while True:
        elapsed = clock.getTime()
        remaining = time_limit - elapsed
        if remaining <= 0:
            return None, None, buffer, True

        frac_remaining = remaining / time_limit
        hurry = frac_remaining <= hurry_fraction

        draw_countdown_bar(win, frac_remaining, hurry)
        draw_leaderboard(win, leaderboard_names, leaderboard_scores)
        prompt = visual.TextStim(win, text=f"Question {question_num}\n\n{expr_str} = {buffer}", height=0.08, pos=(-0.3, 0))
        prompt.draw()
        win.flip()

        for key, kt in event.getKeys(
            keyList=[str(d) for d in range(10)] + ["minus", "return", "backspace", "escape"],
            timeStamped=clock,
        ):
            if key == "escape":
                raise UserQuit()
            if key == "return":
                if buffer:
                    try:
                        return int(buffer), kt, buffer, False
                    except ValueError:
                        return None, kt, buffer, False
            elif key == "backspace":
                buffer = buffer[:-1]
            elif key == "minus":
                if buffer == "":
                    buffer = "-"
            else:
                buffer += key


def run_baseline_block(win, ctx, tier_id: int, block_index: int, duration: float):
    ctx.event_logger.log("block_start", task="stress", block_index=block_index, condition_label="baseline", tier=None)
    show_message(win, "Baseline: please sit quietly with your eyes closed until you hear the tone.", duration=duration)
    ctx.event_logger.log("block_end", task="stress", block_index=block_index, condition_label="baseline", tier=None)


def run_arithmetic_block(win, ctx, tier_cfg: dict, block_index: int, duration: float, leaderboard_names, leaderboard_scores, hurry_fraction: float):
    tier_id = tier_cfg["id"]
    ctx.event_logger.log("block_start", task="stress", block_index=block_index, condition_label=f"tier_{tier_id}", tier=tier_id)

    block_clock = core.Clock()
    question_num = 0
    correct_count = 0

    while block_clock.getTime() < duration:
        check_quit()
        question_num += 1
        expr_str, correct_answer = generate_question(tier_cfg, ctx.rng)
        ctx.event_logger.log(
            "trial_start", task="stress", block_index=block_index, trial_index=question_num,
            condition_label=f"tier_{tier_id}", tier=tier_id, expression=expr_str, correct_answer=correct_answer,
        )
        answer, rt, raw_buffer, timed_out = get_numeric_answer(
            win, expr_str, question_num, tier_cfg["time_per_question_sec"],
            leaderboard_names, leaderboard_scores, hurry_fraction,
        )
        correct = (answer == correct_answer)
        correct_count += int(correct)
        ctx.event_logger.log(
            "response", task="stress", block_index=block_index, trial_index=question_num,
            condition_label=f"tier_{tier_id}", tier=tier_id, expression=expr_str, correct_answer=correct_answer,
            participant_answer=answer, raw_input=raw_buffer, rt=rt, correct=correct, timed_out=timed_out,
        )

    ctx.event_logger.log(
        "block_end", task="stress", block_index=block_index, condition_label=f"tier_{tier_id}", tier=tier_id,
        accuracy=correct_count / question_num if question_num else None,
    )


def run_stress_task(win, ctx):
    cfg = ctx.config["stress_task"]
    baseline_duration = ctx.scaled(cfg["baseline_duration_sec"])
    arithmetic_duration = ctx.scaled(cfg["arithmetic_duration_sec"])
    hurry_fraction = cfg["hurry_up_threshold_fraction"]
    leaderboard_names = cfg["leaderboard_names"]
    leaderboard_scores = cfg["leaderboard_scores"]
    tiers = list(cfg["tiers"])

    show_message(
        win,
        "TASK 3: MENTAL ARITHMETIC\n\n"
        "You'll see arithmetic problems. Solve LEFT TO RIGHT (no order of operations) "
        "and type the answer, then press ENTER.\n\n"
        "Answer as many as you can before time runs out on each question. "
        "A leaderboard and countdown timer will be visible -- do your best!\n\n"
        "Press SPACE to begin.",
        wait_key=["space", "return"],
    )

    ctx.rng.shuffle(tiers)
    ctx.event_logger.log("task_start", task="stress")

    for pair_index, tier_cfg in enumerate(tiers):
        check_quit()
        block_index = pair_index * 2
        run_baseline_block(win, ctx, tier_cfg["id"], block_index, baseline_duration)
        run_arithmetic_block(
            win, ctx, tier_cfg, block_index + 1, arithmetic_duration,
            leaderboard_names, leaderboard_scores, hurry_fraction,
        )

    ctx.event_logger.log("task_end", task="stress")
