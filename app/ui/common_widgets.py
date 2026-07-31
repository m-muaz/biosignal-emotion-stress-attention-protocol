from psychopy import core, event, visual

# Shared dark theme. BG_COLOR is applied to the psychopy Window itself
# (see app/main.py); everything else here is consumed by the drawing
# helpers below.
BG_COLOR = "#1b1e2b"
PANEL_COLOR = "#2a2f45"
BORDER_COLOR = "#454b68"
TEXT_COLOR = "#eceff4"
MUTED_COLOR = "#9096b3"
ACCENT_COLOR = "#7fd8a6"
WARN_COLOR = "#e8846b"
GOLD_COLOR = "#f2c94c"

# "HURRY UP!" flashes on/off every this many seconds (full cycle = 2x this),
# i.e. ~1.7 Hz -- see draw_countdown_bar.
HURRY_BLINK_HALF_PERIOD_SEC = 0.3


class UserQuit(Exception):
    """Raised when the operator presses Escape to abort a session early."""


def check_quit() -> None:
    if event.getKeys(keyList=["escape"]):
        raise UserQuit()


# Skip keys let a block or single question be ended early -- task participation
# is voluntary, so participants (or the operator on their behalf) need a way to
# move on from a block/question without invoking a full session-abort (Escape).
# Letters chosen don't collide with any task's participant-facing input (digits,
# minus, return, backspace). Available in real sessions and standalone debug
# runs alike -- both go through the same task functions.
SKIP_BLOCK_KEY = "b"
SKIP_TRIAL_KEY = "n"


class SkipBlock(Exception):
    """Raised when SKIP_BLOCK_KEY is pressed to end the current block early.

    Deliberately NOT wired into check_quit(), since check_quit() is called
    from shared helpers used by every task -- only callers that actually
    catch SkipBlock (currently the stress task's block runners) should opt
    in, via check_skip_block().
    """


def check_skip_block() -> None:
    if event.getKeys(keyList=[SKIP_BLOCK_KEY]):
        raise SkipBlock()


class SkipTrial(Exception):
    """Raised when SKIP_TRIAL_KEY is pressed to end the current trial early.

    Mirrors SkipBlock (see check_skip_trial()) -- deliberately NOT wired into
    check_quit() for the same reason: only callers that actually catch
    SkipTrial (currently the emotion task) should opt in.
    """


def check_skip_trial() -> None:
    if event.getKeys(keyList=[SKIP_TRIAL_KEY]):
        raise SkipTrial()


def draw_key_hint(win, include_skip_trial: bool = True, pos=(0, -0.46)):
    """Small persistent reminder of the skip/quit keys, drawn on every task
    screen that offers them -- task participation is voluntary, so the
    controls need to stay visible rather than living only in the
    instructions screen the participant saw once at the start.
    """
    parts = []
    if include_skip_trial:
        parts.append(f"{SKIP_TRIAL_KEY.upper()} = skip question")
    parts.append(f"{SKIP_BLOCK_KEY.upper()} = skip block")
    parts.append("ESC = quit")
    hint = visual.TextStim(win, text="   |   ".join(parts), height=0.03, pos=pos, color=MUTED_COLOR)
    hint.draw()


def show_message(win, text, duration=None, wait_key=None, font_height=0.05):
    stim = visual.TextStim(win, text=text, height=font_height, wrapWidth=1.4, color=TEXT_COLOR)
    stim.draw()
    win.flip()
    check_quit()
    if wait_key:
        keys = event.waitKeys(keyList=list(wait_key) + ["escape"])
        if keys and "escape" in keys:
            raise UserQuit()
    elif duration is not None:
        core.wait(duration)


def show_title_screen(win, heading: str, subtext: str = "", wait_key=("space", "return")):
    """Minimal splash screen: one big heading line + an optional smaller
    subtext line -- for screens that just need to orient the participant
    (e.g. the very first screen), where a full paragraph at any single font
    size either overflows the window's visible bounds or is too small to read.
    """
    heading_stim = visual.TextStim(win, text=heading, height=0.09, wrapWidth=1.6, bold=True, pos=(0, 0.1), color=TEXT_COLOR)
    heading_stim.draw()
    if subtext:
        visual.TextStim(win, text=subtext, height=0.04, wrapWidth=1.4, pos=(0, -0.15), color=MUTED_COLOR).draw()
    win.flip()
    check_quit()
    keys = event.waitKeys(keyList=list(wait_key) + ["escape"])
    if keys and "escape" in keys:
        raise UserQuit()


def fixation_cross(win, duration, allow_skip: bool = False, progress_label: str | None = None):
    """If allow_skip, SKIP_BLOCK_KEY/SKIP_TRIAL_KEY are live and raise
    SkipBlock/SkipTrial -- off by default so callers without skip semantics
    (e.g. the attention task) behave exactly as before. progress_label, if
    given, draws a small persistent corner label (e.g. "Clip 2 of 3")."""
    cross = visual.TextStim(win, text="+", height=0.1, color=TEXT_COLOR)
    label = (
        visual.TextStim(win, text=progress_label, height=0.03, pos=(-0.7, 0.46), color=MUTED_COLOR, alignText="left", anchorHoriz="left")
        if progress_label else None
    )
    if not allow_skip:
        cross.draw()
        if label:
            label.draw()
        win.flip()
        check_quit()
        core.wait(duration)
        return

    # Flush any keys queued from the previous screen (e.g. digits pressed
    # during video playback, which isn't listening for them) -- otherwise
    # they get silently consumed the instant this loop starts polling,
    # which is exactly what made the next screen (usually discrete_choice)
    # appear to "skip" or auto-answer.
    event.clearEvents()
    clock = core.Clock()
    while clock.getTime() < duration:
        check_quit()
        check_skip_block()
        check_skip_trial()
        cross.draw()
        if label:
            label.draw()
        draw_key_hint(win)
        win.flip()


def draw_countdown_bar(win, fraction_remaining: float, hurry_up: bool = False):
    """Draws a track + shrinking fill bar so the full time budget stays visible
    as a reference, not just the remaining sliver (which is hard to read once small).

    Positioned inside the window's visible height-units bounds (y in [-0.5, 0.5]) --
    the previous version placed it at y=0.85, entirely off-canvas.
    """
    fraction_remaining = max(0.0, min(1.0, fraction_remaining))
    track_width = 1.1
    bar_y = 0.42
    left_edge = -track_width / 2

    track = visual.Rect(
        win, width=track_width, height=0.05, pos=(0, bar_y),
        fillColor=PANEL_COLOR, lineColor=BORDER_COLOR, lineWidth=1,
    )
    track.draw()

    fill_width = max(track_width * fraction_remaining, 0.001)
    fill_color = WARN_COLOR if hurry_up else ACCENT_COLOR
    fill = visual.Rect(
        win, width=fill_width, height=0.05, pos=(left_edge + fill_width / 2, bar_y),
        fillColor=fill_color, lineColor=None,
    )
    fill.draw()

    if hurry_up:
        # Blinks (rather than staying static) so the time pressure keeps registering
        # perceptually instead of fading into the background -- kept under 3 flashes/sec
        # (WCAG general flash threshold) since this runs for the length of a whole block.
        blink_on = int(core.getTime() / HURRY_BLINK_HALF_PERIOD_SEC) % 2 == 0
        if blink_on:
            hurry_text = visual.TextStim(win, text="HURRY UP!", height=0.05, pos=(0, 0.33), color=WARN_COLOR, bold=True)
            hurry_text.draw()


def draw_progress_bar(win, fraction_remaining: float):
    """Calm variant of draw_countdown_bar (never enters the hurry-up/warn state) --
    for passive countdowns like a baseline rest period, where the participant
    just needs a visible "how much longer" cue instead of an audio tone."""
    draw_countdown_bar(win, fraction_remaining, hurry_up=False)


def draw_leaderboard(win, names, scores, pos=(0.3, 0.42), center_vertically=True):
    """Draws the leaderboard as a card anchored top-left at `pos`.

    `pos` and the card width are chosen to stay within the window's visible
    horizontal bounds (x in roughly [-0.8, 0.8] at 1280x800) -- the previous
    version anchored at x=0.75 with left-aligned text, which ran off the right
    edge of the screen for every row.

    By default the card is vertically centered on the window (`pos`'s y is
    ignored) regardless of how many rows it has -- pass center_vertically=False
    to anchor the top edge at pos's y instead.
    """
    px, py = pos
    line_height = 0.048
    panel_width = 0.42
    n_lines = len(names) + 1
    panel_height = n_lines * line_height + 0.035
    if center_vertically:
        py = panel_height / 2

    panel = visual.Rect(
        win, width=panel_width, height=panel_height,
        pos=(px + panel_width / 2 - 0.02, py - panel_height / 2 + 0.018),
        fillColor=PANEL_COLOR, lineColor=BORDER_COLOR, lineWidth=1,
    )
    panel.draw()

    header = visual.TextStim(
        win, text="LEADERBOARD", height=0.036, pos=(px, py), color=GOLD_COLOR,
        bold=True, alignText="left", anchorHoriz="left", anchorVert="top",
    )
    header.draw()

    lines = "\n".join(f"{i + 1}. {n:<8}{s}" for i, (n, s) in enumerate(zip(names, scores)))
    body = visual.TextStim(
        win, text=lines, height=0.032, pos=(px, py - line_height), color=TEXT_COLOR,
        alignText="left", anchorHoriz="left", anchorVert="top",
    )
    body.draw()


def _rating_box_layout(scale_min: int, scale_max: int):
    """x-offsets (in height units, relative to row center) for one 0..scale_max
    box row, plus the shared box size. Shared by rating_scale_multi's drawing
    and hit-testing so the two never disagree about where a box actually is.
    """
    n = scale_max - scale_min + 1
    box_size = 0.062
    gap = 0.02
    total_width = n * box_size + (n - 1) * gap
    start_x = -total_width / 2 + box_size / 2
    return [(scale_min + i, start_x + i * (box_size + gap)) for i in range(n)], box_size


def _draw_rating_row(win, y, layout, box_size, selected_value, hovered_value, active: bool, left_label: str, right_label: str):
    for value, x in layout:
        is_selected = selected_value == value
        is_hovered = active and not is_selected and hovered_value == value
        fill = ACCENT_COLOR if is_selected else (BORDER_COLOR if is_hovered else PANEL_COLOR)
        visual.Rect(win, width=box_size, height=box_size, pos=(x, y), fillColor=fill, lineColor=BORDER_COLOR, lineWidth=1).draw()
        visual.TextStim(win, text=str(value), height=0.032, pos=(x, y), color=BG_COLOR if is_selected else TEXT_COLOR).draw()

    label_color = TEXT_COLOR if active else MUTED_COLOR
    first_x, last_x = layout[0][1], layout[-1][1]
    visual.TextStim(win, text=left_label, height=0.026, pos=(first_x - box_size * 1.5, y), color=label_color, alignText="right", anchorHoriz="right").draw()
    visual.TextStim(win, text=right_label, height=0.026, pos=(last_x + box_size * 1.5, y), color=label_color, alignText="left", anchorHoriz="left").draw()


def _hit_test_row(mouse, y, layout, box_size) -> int | None:
    mx, my = mouse.getPos()
    half = box_size / 2
    if not (y - half <= my <= y + half):
        return None
    for value, x in layout:
        if x - half <= mx <= x + half:
            return value
    return None


def rating_scale_multi(
    win, items: list[tuple[str, str] | tuple[str, str, str, str]], scale_min: int = 0, scale_max: int = 7,
    timeout: float | None = None, allow_skip: bool = False,
):
    """Draws every item in `items` as a stacked rating row on one screen (per
    ui-mockup/emotion_test/valence_liking_rating.png) instead of one screen
    per item. Rows are answered top-to-bottom: only the first unanswered row
    is live (highlighted boxes, responds to input); answered rows show their
    locked-in value, later rows are dimmed until it's their turn. Each row
    takes either a number key (scale_min..scale_max) or a mouse click on a box.

    Each item is (item_key, prompt) or (item_key, prompt, left_label,
    right_label) -- the last two are the low/high endpoint labels shown at
    either end of that row (e.g. "Very unpleasant"/"Very pleasant" for a
    bipolar valence item, vs. the generic "not at all"/"extremely" default),
    since a single generic anchor pair doesn't fit every question and leaves
    the participant guessing which direction is which.

    If allow_skip, SKIP_BLOCK_KEY/SKIP_TRIAL_KEY are also live and raise
    SkipBlock/SkipTrial -- off by default (e.g. rating_scale_0_7's other
    callers, like the consent questionnaire, have no skip semantics).

    Returns {item_key: (value:int|None, rt:float|None)}.
    """
    norm_items = [(it[0], it[1], it[2] if len(it) > 2 else "not at all", it[3] if len(it) > 3 else "extremely") for it in items]

    values: dict[str, int | None] = {key: None for key, _, _, _ in norm_items}
    rts: dict[str, float | None] = {key: None for key, _, _, _ in norm_items}
    layout, box_size = _rating_box_layout(scale_min, scale_max)

    row_gap = 0.16
    top_y = (len(norm_items) - 1) * row_gap / 2 + 0.06
    prompt_stims = [
        visual.TextStim(win, text=prompt, height=0.04, wrapWidth=1.4, bold=True, pos=(0, top_y - i * row_gap + 0.055), color=TEXT_COLOR)
        for i, (_, prompt, _, _) in enumerate(norm_items)
    ]
    hint = visual.TextStim(
        win, text="Click a box, or press a number key, on each row to respond.",
        height=0.028, pos=(0, top_y - len(norm_items) * row_gap + 0.02), color=MUTED_COLOR,
    )

    mouse = event.Mouse(win=win)
    clock = core.Clock()
    key_list = [str(i) for i in range(scale_min, scale_max + 1)]
    # Seed with the mouse's current button state (not False) so a click
    # already in progress when this screen appears isn't misread as a fresh
    # click on whatever happens to be under the cursor.
    prev_pressed = mouse.getPressed()[0]
    # See fixation_cross's comment: flush stale keys from whatever screen was
    # up before this one, so they can't get instantly (and invisibly)
    # consumed as soon as this loop starts polling.
    event.clearEvents()

    while True:
        check_quit()
        if allow_skip:
            check_skip_block()
            check_skip_trial()
        active_index = next((i for i, (key, _, _, _) in enumerate(norm_items) if values[key] is None), None)
        if active_index is None:
            return {key: (values[key], rts[key]) for key, _, _, _ in norm_items}

        active_key = norm_items[active_index][0]
        active_y = top_y - active_index * row_gap
        hovered_value = _hit_test_row(mouse, active_y, layout, box_size)

        for i, (key, _, left_label, right_label) in enumerate(norm_items):
            prompt_stims[i].draw()
            _draw_rating_row(
                win, top_y - i * row_gap, layout, box_size, values[key], hovered_value,
                active=(i == active_index), left_label=left_label, right_label=right_label,
            )
        hint.draw()
        if allow_skip:
            draw_key_hint(win, pos=(0, -0.42))
        win.flip()

        if timeout is not None and clock.getTime() > timeout:
            return {key: (values[key], rts[key]) for key, _, _, _ in norm_items}

        keys = event.getKeys(keyList=key_list, timeStamped=clock)
        if keys:
            key_pressed, rt = keys[0]
            values[active_key] = int(key_pressed)
            rts[active_key] = rt
            continue

        pressed = mouse.getPressed()[0]
        if pressed and not prev_pressed and hovered_value is not None:
            values[active_key] = hovered_value
            rts[active_key] = clock.getTime()
        prev_pressed = pressed


def rating_scale_0_7(
    win, prompt: str, scale_min: int = 0, scale_max: int = 7, timeout: float | None = None,
    left_label: str = "not at all", right_label: str = "extremely",
):
    """0..7 rating via number key or mouse click on a box. Returns (value:int|None, rt:float|None)."""
    return rating_scale_multi(win, [("_value", prompt, left_label, right_label)], scale_min, scale_max, timeout)["_value"]


def numeric_entry(win, prompt: str, allow_negative: bool = False, allow_decimal: bool = True):
    """Untimed numeric text entry (digits, backspace, ENTER to submit). Returns (value, rt)."""
    buffer = ""
    key_list = [str(d) for d in range(10)] + ["return", "backspace", "escape"]
    if allow_negative:
        key_list.append("minus")
    if allow_decimal:
        key_list.append("period")

    clock = core.Clock()
    while True:
        stim = visual.TextStim(win, text=f"{prompt}\n\n{buffer}", height=0.05, wrapWidth=1.4, color=TEXT_COLOR)
        stim.draw()
        win.flip()
        check_quit()
        keys = event.waitKeys(keyList=key_list, timeStamped=clock)
        if not keys:
            continue
        key, rt = keys[0]
        if key == "escape":
            raise UserQuit()
        if key == "return":
            if not buffer:
                continue
            try:
                value = float(buffer) if "." in buffer else int(buffer)
            except ValueError:
                value = None
            return value, rt
        elif key == "backspace":
            buffer = buffer[:-1]
        elif key == "minus":
            if buffer == "":
                buffer = "-"
        elif key == "period":
            if "." not in buffer:
                buffer += "."
        else:
            buffer += key


def discrete_choice(win, prompt: str, options: list[str], timeout: float | None = None, allow_skip: bool = False):
    """Participant presses 1..len(options), or clicks the option with the mouse.
    If allow_skip, SKIP_BLOCK_KEY/SKIP_TRIAL_KEY are also live and raise
    SkipBlock/SkipTrial -- off by default since most callers (e.g. the consent
    questionnaire) have no skip semantics for those keys.
    Returns (choice:str|None, rt:float|None)."""
    title = visual.TextStim(win, text=prompt, height=0.05, wrapWidth=1.4, bold=True, pos=(0, 0.32), color=TEXT_COLOR)
    start_y = 0.15
    # Space rows to fit within a safe vertical band regardless of option
    # count, rather than a fixed 0.09 -- with enough options (e.g. the fixed
    # 8-item emotion list), fixed spacing ran the last row off the bottom of
    # the visible window (y < -0.5) and into draw_key_hint's row at y=-0.46.
    # bottom_y=-0.34 leaves clearance above that hint row.
    bottom_y = -0.34
    n = len(options)
    line_height = min(0.09, (start_y - bottom_y) / (n - 1)) if n > 1 else 0.09

    mouse = event.Mouse(win=win)
    clock = core.Clock()
    key_list = [str(i + 1) for i in range(len(options))]
    # Seed with the mouse's current button state (not False) so a click
    # already in progress when this screen appears isn't misread as a fresh
    # click on whatever happens to be under the cursor.
    prev_pressed = mouse.getPressed()[0]
    # See fixation_cross's comment: flush stale keys from whatever screen was
    # up before this one, so they can't get instantly (and invisibly)
    # consumed as soon as this loop starts polling.
    event.clearEvents()

    while True:
        check_quit()
        if allow_skip:
            check_skip_block()
            check_skip_trial()
        mx, my = mouse.getPos()
        hovered = None
        for i in range(len(options)):
            y = start_y - i * line_height
            if abs(my - y) <= line_height / 2 and -0.6 <= mx <= 0.6:
                hovered = i

        title.draw()
        for i, opt in enumerate(options):
            y = start_y - i * line_height
            color = ACCENT_COLOR if i == hovered else TEXT_COLOR
            visual.TextStim(
                win, text=f"{i + 1}.  {opt}", height=0.045, pos=(-0.5, y), color=color,
                alignText="left", anchorHoriz="left",
            ).draw()
        if allow_skip:
            draw_key_hint(win)
        win.flip()

        if timeout is not None and clock.getTime() > timeout:
            return None, None

        keys = event.getKeys(keyList=key_list, timeStamped=clock)
        if keys:
            key, rt = keys[0]
            return options[int(key) - 1], rt

        pressed = mouse.getPressed()[0]
        if pressed and not prev_pressed and hovered is not None:
            return options[hovered], clock.getTime()
        prev_pressed = pressed
