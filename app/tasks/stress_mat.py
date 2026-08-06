"""Task 3: tiered MAT-style mental arithmetic (stress), per Engineering_Document.md §4.3.

Baseline/arithmetic block *lengths* are scaled for demo runs (ctx.scaled), but
the per-question time limit is never scaled -- shortening the time-pressure
window would defeat the point of a demo run that's supposed to let a human
operator actually feel what the task is like.

Tier is logged on the backend only and never shown on screen, so post-hoc
analysis can either collapse all tiers into a binary stress/no-stress label
or treat them as ordinal stress-intensity levels.

Participation is voluntary, so SKIP_BLOCK_KEY/SKIP_TRIAL_KEY (see
app.ui.common_widgets; B/N by default) let a block or single question be
ended early without invoking a full session-abort (Escape) -- every skip is
still logged (event_type "response"/"block_end" with skipped=True) so it's
visible in post-hoc analysis rather than silently missing data. draw_key_hint
keeps the controls visible on every task screen, not just the one-time
instructions.
"""

from psychopy import core, event, visual

from app.ui.common_widgets import (
    MUTED_COLOR,
    SKIP_BLOCK_KEY,
    SKIP_TRIAL_KEY,
    TEXT_COLOR,
    SkipBlock,
    UserQuit,
    check_quit,
    check_skip_block,
    draw_countdown_bar,
    draw_key_hint,
    draw_leaderboard,
    draw_progress_bar,
    show_message,
)


def generate_question(tier_cfg: dict, rng) -> tuple[str, int]:
    """Builds a left-to-right arithmetic expression with an exact integer answer.

    Division must land on a whole number, so when the running value isn't
    already a multiple of the chosen divisor, it needs padding up first.
    That padding is folded invisibly into the immediately preceding term
    whenever that's algebraically safe (the preceding term was itself added
    or subtracted, or it's the very first number) -- e.g. a "35 / 2" needing
    +1 becomes "36 / 2" rather than the more confusing "35 + 1 / 2". It's
    only shown as an explicit extra step when the preceding term came from a
    "*" or "/", where folding it in would silently change the answer.
    """
    ops = tier_cfg["operators"]
    n = tier_cfg["num_operands"]
    max_val = 10 ** tier_cfg["max_digits"] - 1

    first = rng.randint(1, max_val)
    tokens = [str(first)]
    value = first
    last_op = None  # operator behind the current last token; None = it's still just `first`

    def fold_pad(pad):
        """Try to absorb `pad` into the preceding term in place (tokens/value
        already updated on success). Returns False if that isn't safe here,
        leaving it to the caller to append an explicit "+ pad" step instead."""
        nonlocal value
        if last_op not in (None, "+", "-"):
            return False
        prior = int(tokens[-1])
        if last_op == "-":
            new_operand = prior - pad
            if new_operand > 0:
                tokens[-1] = str(new_operand)
            elif new_operand < 0:
                tokens[-2] = "+"
                tokens[-1] = str(-new_operand)
            else:
                del tokens[-2:]
        else:  # None (very first term) or "+"
            tokens[-1] = str(prior + pad)
        value += pad
        return True

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
                    if not fold_pad(pad):
                        tokens += ["+", str(pad)]
                        value += pad
                operand, new_value = divisor, value // divisor
            tokens += ["/", str(operand)]
            value = new_value
            last_op = "/"
        elif op == "*":
            operand = rng.randint(2, 9)
            tokens += ["*", str(operand)]
            value *= operand
            last_op = "*"
        elif op == "+":
            operand = rng.randint(1, max_val)
            tokens += ["+", str(operand)]
            value += operand
            last_op = "+"
        else:  # "-"
            operand = rng.randint(1, max_val)
            tokens += ["-", str(operand)]
            value -= operand
            last_op = "-"

    return " ".join(tokens), value


def rig_rival_score(participant_score: float, gap: float, floor: float, cap: float) -> int:
    """MIST-style bogus performance feedback (Dedovic et al. 2005, Montreal
    Imaging Stress Task): the rival's displayed score is dynamically pinned
    just above the participant's own live score, rather than a fixed number,
    so the social-evaluative-threat manipulation stays effective regardless
    of how well the participant is actually doing.
    """
    return int(min(cap, max(floor, participant_score + gap)))


def build_leaderboard(participant_score: float, leaderboard_cfg: dict):
    """Assembles the (names, scores) rows shown this question: the rigged
    rival (see rig_rival_score), the participant's own real/live score, and
    the fixed filler names -- all re-sorted by score every call so the
    on-screen ranking always matches the numbers shown.
    """
    rival_score = rig_rival_score(
        participant_score,
        leaderboard_cfg["rigged_gap"],
        leaderboard_cfg["rigged_floor"],
        leaderboard_cfg["rigged_cap"],
    )
    names = [leaderboard_cfg["rival_name"], leaderboard_cfg["participant_label"], *leaderboard_cfg["fixed_names"]]
    scores = [rival_score, int(round(participant_score)), *leaderboard_cfg["fixed_scores"]]
    order = sorted(range(len(names)), key=lambda i: -scores[i])
    return [names[i] for i in order], [scores[i] for i in order], rival_score


def get_numeric_answer(win, expr_str, question_num, time_limit, participant_score, leaderboard_cfg, hurry_fraction):
    """Renders the countdown/leaderboard/hurry-up UI and collects a typed answer.

    participant_score is the participant's own live score (as of the start of
    this question) used to build the rigged leaderboard -- see build_leaderboard.

    Returns (answer:int|None, rt:float|None, raw_buffer:str, timed_out:bool, skipped:bool, rival_score:int).
    Raises SkipBlock if SKIP_BLOCK_KEY is pressed, to end the whole block early.
    """
    clock = core.Clock()
    buffer = ""
    event.clearEvents()
    names, scores, rival_score = build_leaderboard(participant_score, leaderboard_cfg)

    while True:
        # Also catches a window-close click, not just Escape -- see
        # check_quit/request_quit in common_widgets.py. This loop otherwise
        # bypasses check_quit() (it does its own escape handling below,
        # scoped to this question's keyList), so without this the close
        # button wouldn't be noticed until the current question timed out.
        check_quit()

        elapsed = clock.getTime()
        remaining = time_limit - elapsed
        if remaining <= 0:
            return None, None, buffer, True, False, rival_score

        frac_remaining = remaining / time_limit
        hurry = frac_remaining <= hurry_fraction

        draw_countdown_bar(win, frac_remaining, hurry)
        draw_leaderboard(win, names, scores, highlight_name=leaderboard_cfg["participant_label"])

        label = visual.TextStim(
            win, text=f"Question {question_num}", height=0.045,
            pos=(-0.42, 0.08), color=MUTED_COLOR, alignText="left", anchorHoriz="left",
        )
        label.draw()

        expr = visual.TextStim(
            win, text=f"{expr_str} = {buffer}", height=0.09, bold=True,
            pos=(-0.42, -0.02), wrapWidth=0.62, color=TEXT_COLOR,
            alignText="left", anchorHoriz="left", anchorVert="top",
        )
        expr.draw()
        draw_key_hint(win, include_skip_trial=True)
        win.flip()

        for key, kt in event.getKeys(
            keyList=[str(d) for d in range(10)]
            + ["minus", "return", "backspace", "escape", SKIP_BLOCK_KEY, SKIP_TRIAL_KEY],
            timeStamped=clock,
        ):
            if key == "escape":
                raise UserQuit()
            if key == SKIP_BLOCK_KEY:
                raise SkipBlock()
            if key == SKIP_TRIAL_KEY:
                return None, kt, buffer, False, True, rival_score
            if key == "return":
                if buffer:
                    try:
                        return int(buffer), kt, buffer, False, False, rival_score
                    except ValueError:
                        return None, kt, buffer, False, False, rival_score
            elif key == "backspace":
                buffer = buffer[:-1]
            elif key == "minus":
                if buffer == "":
                    buffer = "-"
            else:
                buffer += key


def run_baseline_block(win, ctx, tier_id: int, block_index: int, duration: float):
    ctx.event_logger.log("block_start", task="stress", block_index=block_index, condition_label="baseline", tier=None)

    clock = core.Clock()
    message = visual.TextStim(
        win, text="Baseline: please sit quietly and relax.\n\nThe next phase begins when the bar below runs out.",
        height=0.05, wrapWidth=1.4, pos=(0, 0.1), color=TEXT_COLOR,
    )
    skipped = False
    try:
        while clock.getTime() < duration:
            check_quit()
            check_skip_block()
            remaining = duration - clock.getTime()
            draw_progress_bar(win, remaining / duration)
            message.draw()
            draw_key_hint(win, include_skip_trial=False)
            win.flip()
    except SkipBlock:
        skipped = True

    ctx.event_logger.log(
        "block_end", task="stress", block_index=block_index, condition_label="baseline", tier=None, skipped=skipped,
    )


def run_arithmetic_block(win, ctx, tier_cfg: dict, block_index: int, duration: float, leaderboard_cfg: dict, hurry_fraction: float):
    tier_id = tier_cfg["id"]
    ctx.event_logger.log("block_start", task="stress", block_index=block_index, condition_label=f"tier_{tier_id}", tier=tier_id)

    block_clock = core.Clock()
    question_num = 0
    correct_count = 0
    skipped_block = False

    while block_clock.getTime() < duration:
        check_quit()
        check_skip_block()
        question_num += 1
        # Participant's live score (as of the start of this question, i.e.
        # not counting the answer they're about to give) drives the rigged
        # leaderboard for this question -- see build_leaderboard/rig_rival_score.
        participant_score = 100.0 * correct_count / (question_num - 1) if question_num > 1 else 0.0
        expr_str, correct_answer = generate_question(tier_cfg, ctx.rng)
        ctx.event_logger.log(
            "trial_start", task="stress", block_index=block_index, trial_index=question_num,
            condition_label=f"tier_{tier_id}", tier=tier_id, expression=expr_str, correct_answer=correct_answer,
            leaderboard_participant_score_shown=round(participant_score),
        )
        try:
            answer, rt, raw_buffer, timed_out, skipped, rival_score = get_numeric_answer(
                win, expr_str, question_num, tier_cfg["time_per_question_sec"],
                participant_score, leaderboard_cfg, hurry_fraction,
            )
        except SkipBlock:
            ctx.event_logger.log(
                "response", task="stress", block_index=block_index, trial_index=question_num,
                condition_label=f"tier_{tier_id}", tier=tier_id, expression=expr_str, correct_answer=correct_answer,
                participant_answer=None, raw_input="", rt=None, correct=False, timed_out=False, skipped=True,
            )
            skipped_block = True
            break

        correct = (answer == correct_answer)
        correct_count += int(correct)
        ctx.event_logger.log(
            "response", task="stress", block_index=block_index, trial_index=question_num,
            condition_label=f"tier_{tier_id}", tier=tier_id, expression=expr_str, correct_answer=correct_answer,
            participant_answer=answer, raw_input=raw_buffer, rt=rt, correct=correct, timed_out=timed_out, skipped=skipped,
            leaderboard_rival_score_shown=rival_score,
        )

    ctx.event_logger.log(
        "block_end", task="stress", block_index=block_index, condition_label=f"tier_{tier_id}", tier=tier_id,
        accuracy=correct_count / question_num if question_num else None, skipped=skipped_block,
    )


def run_stress_task(win, ctx):
    cfg = ctx.config["stress_task"]
    baseline_duration = ctx.scaled(cfg["baseline_duration_sec"])
    arithmetic_duration = ctx.scaled(cfg["arithmetic_duration_sec"])
    hurry_fraction = cfg["hurry_up_threshold_fraction"]
    leaderboard_cfg = {
        "rival_name": cfg["leaderboard_rival_name"],
        "participant_label": cfg["leaderboard_participant_label"],
        "fixed_names": cfg["leaderboard_fixed_names"],
        "fixed_scores": cfg["leaderboard_fixed_scores"],
        "rigged_gap": cfg["leaderboard_rigged_gap"],
        "rigged_floor": cfg["leaderboard_rigged_floor"],
        "rigged_cap": cfg["leaderboard_rigged_cap"],
    }

    tier_by_id = {tier_cfg["id"]: tier_cfg for tier_cfg in cfg["tiers"]}
    try:
        trials = [tier_by_id[tier_id] for tier_id in cfg["trial_order"]]
    except KeyError as e:
        raise ValueError(
            f"stress_task.trial_order references tier id {e.args[0]!r}, which isn't defined in stress_task.tiers"
        ) from e
    n_trials = len(trials)

    show_message(
        win,
        "TASK 3: MENTAL ARITHMETIC\n\n"
        "You'll see arithmetic problems. Solve each question "
        "and type the answer, then press ENTER.\n\n"
        "Answer as many as you can before time runs out on each question. "
        "A leaderboard and countdown timer will be visible! -- DO YOUR BEST ✺◟(＾∇＾)◞✺\n\n"
        f"You will complete {n_trials} rounds, each starting with a short rest "
        "period followed by arithmetic questions.\n\n"
        "Your participation is voluntary:\n"
        "- Press N at any time to skip a question \n"
        "- Press B to skip the current block \n"
        "- Press Esc to stop the session entirely. "
        # "These key reminders stay on screen throughout the task.\n\n"
        "Press SPACE to begin.",
        wait_key=["space", "return"],
        font_height=0.035,
    )

    ctx.event_logger.log("task_start", task="stress")

    for pair_index, tier_cfg in enumerate(trials):
        check_quit()
        block_index = pair_index * 2
        run_baseline_block(win, ctx, tier_cfg["id"], block_index, baseline_duration)
        run_arithmetic_block(
            win, ctx, tier_cfg, block_index + 1, arithmetic_duration,
            leaderboard_cfg, hurry_fraction,
        )

    ctx.event_logger.log("task_end", task="stress")
