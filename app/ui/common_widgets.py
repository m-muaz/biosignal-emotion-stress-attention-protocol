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
    """Raised when the operator presses Escape (or clicks the window's close
    button -- see request_quit) to abort a session early."""


# Set by app.main._install_close_handler's on_close callback when the OS
# window-close button is clicked. Can't raise UserQuit directly from there:
# on Windows that callback runs inside a ctypes WNDPROC trampoline invoked
# from the Win32 message pump, not a normal Python call stack, so a raised
# exception is caught and silently discarded ("Exception ignored on calling
# ctypes callback function") rather than propagating out to main()'s
# try/except. Instead we set this flag and let the next check_quit() poll
# (called every frame throughout the app, from plain Python code) raise it.
_close_requested = False


def request_quit() -> None:
    global _close_requested
    _close_requested = True


def check_quit() -> None:
    if _close_requested or event.getKeys(keyList=["escape"]):
        raise UserQuit()


# Set by _install_mouse_click_capture's on_mouse_press callback whenever the
# left mouse button goes down. Widgets used to detect a click by sampling
# mouse.getPressed() once per win.flip() cycle and looking for a 0->1 edge
# between consecutive samples -- but a fast click (press+release) can
# complete entirely within a single frame-to-frame gap, in which case both
# samples read "not pressed" and the click is silently lost (worse the
# slower/laggier that gap is -- reported as "doesn't get selected on the
# first try"). Pyglet delivers on_mouse_press independent of our poll rate,
# dispatched via win.dispatch_events() -- which event.getKeys() already
# calls every frame (see _install_close_handler's comment) -- so a press set
# here can't be dropped between polls the way a getPressed() sample can.
_click_pending = False


def _install_mouse_click_capture(win) -> None:
    from pyglet.window import mouse as pyglet_mouse

    def _on_mouse_press(x, y, button, modifiers):
        global _click_pending
        if button == pyglet_mouse.LEFT:
            _click_pending = True

    win.winHandle.on_mouse_press = _on_mouse_press


def consume_click() -> bool:
    """True at most once per physical left-click, no matter how slow the
    polling loop calling this is -- see _install_mouse_click_capture."""
    global _click_pending
    if _click_pending:
        _click_pending = False
        return True
    return False


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


def draw_leaderboard(win, names, scores, pos=(0.3, 0.42), center_vertically=True, highlight_name=None):
    """Draws the leaderboard as a card anchored top-left at `pos`.

    `pos` and the card width are chosen to stay within the window's visible
    horizontal bounds (x in roughly [-0.8, 0.8] at 1280x800) -- the previous
    version anchored at x=0.75 with left-aligned text, which ran off the right
    edge of the screen for every row.

    By default the card is vertically centered on the window (`pos`'s y is
    ignored) regardless of how many rows it has -- pass center_vertically=False
    to anchor the top edge at pos's y instead.

    highlight_name, if given, renders that row (the participant's own, in the
    stress task's rigged leaderboard) in ACCENT_COLOR so it's distinguishable
    from the surrounding names at a glance.
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

    # Drawn one row per TextStim (rather than a single joined-lines block, as
    # before) so highlight_name's row can be colored independently.
    for i, (n, s) in enumerate(zip(names, scores)):
        row_color = ACCENT_COLOR if n == highlight_name else TEXT_COLOR
        row = visual.TextStim(
            win, text=f"{i + 1}. {n:<8}{s}", height=0.032, pos=(px, py - line_height * (i + 1)),
            color=row_color, bold=(n == highlight_name),
            alignText="left", anchorHoriz="left", anchorVert="top",
        )
        row.draw()


def _rating_box_layout(scale_min: int, scale_max: int, box_size: float = 0.062):
    """x-offsets (in height units, relative to row center) for one 0..scale_max
    box row, plus the shared box size. Shared by rating_scale_multi's drawing
    and hit-testing so the two never disagree about where a box actually is.
    """
    n = scale_max - scale_min + 1
    gap = 0.02 * (box_size / 0.062)
    total_width = n * box_size + (n - 1) * gap
    start_x = -total_width / 2 + box_size / 2
    return [(scale_min + i, start_x + i * (box_size + gap)) for i in range(n)], box_size


def _draw_rating_row(win, y, layout, box_size, selected_value, hovered_value, active: bool, left_label: str, right_label: str, font_scale: float = 1.0):
    value_height = 0.032 * font_scale
    label_height = 0.026 * font_scale
    for value, x in layout:
        is_selected = selected_value == value
        is_hovered = active and not is_selected and hovered_value == value
        fill = ACCENT_COLOR if is_selected else (BORDER_COLOR if is_hovered else PANEL_COLOR)
        visual.Rect(win, width=box_size, height=box_size, pos=(x, y), fillColor=fill, lineColor=BORDER_COLOR, lineWidth=1).draw()
        visual.TextStim(win, text=str(value), height=value_height, pos=(x, y), color=BG_COLOR if is_selected else TEXT_COLOR).draw()

    label_color = TEXT_COLOR if active else MUTED_COLOR
    first_x, last_x = layout[0][1], layout[-1][1]
    visual.TextStim(win, text=left_label, height=label_height, pos=(first_x - box_size * 1.5, y), color=label_color, alignText="right", anchorHoriz="right").draw()
    visual.TextStim(win, text=right_label, height=label_height, pos=(last_x + box_size * 1.5, y), color=label_color, alignText="left", anchorHoriz="left").draw()


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
    n = len(norm_items)

    values: dict[str, int | None] = {key: None for key, _, _, _ in norm_items}
    rts: dict[str, float | None] = {key: None for key, _, _, _ in norm_items}

    # Rows are normally DEFAULT_ROW_GAP apart, centered on CENTER_Y -- this
    # matches the original fixed layout, which rating_scale_0_7's many
    # single-row callers (valence/arousal/liking etc.) depend on keeping.
    # Once enough rows are given that spacing would push rows outside the
    # visible window (e.g. PANAS's 10 items previously rendered at
    # y > 0.5, entirely off-canvas -- the reported "mood questions don't
    # show properly" bug), row_gap shrinks -- and box/font size along with
    # it -- just enough to fit everything within MAX_HALF_SPAN of center.
    DEFAULT_ROW_GAP = 0.16
    CENTER_Y = 0.06
    # 0.40 left the topmost row's bold prompt text overflowing past the
    # window's top edge by a hair at max shrink (font_scale floored at 0.6) --
    # 0.37 leaves enough margin for the prompt text height/offset above the
    # top row and the hint line below the bottom row to stay fully on-screen.
    MAX_HALF_SPAN = 0.37
    desired_half_span = (n - 1) * DEFAULT_ROW_GAP / 2
    if n > 1 and desired_half_span > MAX_HALF_SPAN:
        row_gap = (2 * MAX_HALF_SPAN) / (n - 1)
    else:
        row_gap = DEFAULT_ROW_GAP
    font_scale = max(0.6, min(1.0, row_gap / DEFAULT_ROW_GAP))
    top_y = CENTER_Y + (n - 1) * row_gap / 2

    layout, box_size = _rating_box_layout(scale_min, scale_max, box_size=0.062 * font_scale)
    prompt_stims = [
        visual.TextStim(win, text=prompt, height=0.04 * font_scale, wrapWidth=1.4, bold=True, pos=(0, top_y - i * row_gap + 0.055 * font_scale), color=TEXT_COLOR)
        for i, (_, prompt, _, _) in enumerate(norm_items)
    ]
    hint = visual.TextStim(
        win, text="Click a box, or press a number key, on each row to respond.",
        height=0.028, pos=(0, top_y - n * row_gap + 0.02), color=MUTED_COLOR,
    )
    done_hint = visual.TextStim(
        win, text="All set -- click any answer above to change it, or press SPACE/ENTER to continue.",
        height=0.028, wrapWidth=1.4, pos=(0, top_y - n * row_gap + 0.02), color=MUTED_COLOR,
    )

    mouse = event.Mouse(win=win)
    clock = core.Clock()
    key_list = [str(i) for i in range(scale_min, scale_max + 1)]
    _install_mouse_click_capture(win)
    # See fixation_cross's comment: flush stale keys from whatever screen was
    # up before this one, so they can't get instantly (and invisibly)
    # consumed as soon as this loop starts polling. consume_click() likewise
    # drops any click carried over from the previous screen.
    event.clearEvents()
    consume_click()

    while True:
        check_quit()
        if allow_skip:
            check_skip_block()
            check_skip_trial()
        active_index = next((i for i, (key, _, _, _) in enumerate(norm_items) if values[key] is None), None)
        all_answered = active_index is None

        # Hit-test every row, not just the active one -- previously only the
        # active row's y was tested, so once the screen moved past a row
        # there was no way to click back and change that answer.
        row_hover = {i: _hit_test_row(mouse, top_y - i * row_gap, layout, box_size) for i in range(n)}

        for i, (key, _, left_label, right_label) in enumerate(norm_items):
            prompt_stims[i].draw()
            # A row is editable (accepts hover/click) if it's the current
            # active row, or if it's already been answered -- i.e. any
            # answered row can be revisited and changed at any time, not
            # just the one row the participant is currently on.
            editable = (i == active_index) or (values[key] is not None)
            _draw_rating_row(
                win, top_y - i * row_gap, layout, box_size, values[key], row_hover[i],
                active=editable, left_label=left_label, right_label=right_label, font_scale=font_scale,
            )
        (done_hint if all_answered else hint).draw()
        if allow_skip:
            draw_key_hint(win, pos=(0, -0.42))
        win.flip()

        if timeout is not None and clock.getTime() > timeout:
            return {key: (values[key], rts[key]) for key, _, _, _ in norm_items}

        if all_answered:
            # Every row has an answer, but the participant may still want to
            # go back and change one before moving on -- don't return until
            # they explicitly confirm, so a click on an earlier row (below)
            # actually has a chance to be seen.
            if event.getKeys(keyList=["space", "return"]):
                return {key: (values[key], rts[key]) for key, _, _, _ in norm_items}
        else:
            keys = event.getKeys(keyList=key_list, timeStamped=clock)
            if keys:
                key_pressed, rt = keys[0]
                active_key = norm_items[active_index][0]
                values[active_key] = int(key_pressed)
                rts[active_key] = rt
                continue

        if consume_click():
            for i, (key, _, _, _) in enumerate(norm_items):
                hv = row_hover[i]
                if hv is None:
                    continue
                if i == active_index or values[key] is not None:
                    values[key] = hv
                    rts[key] = clock.getTime()
                    break


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


_TEXT_FIELD_KEYS = [chr(c) for c in range(ord("a"), ord("z") + 1)] + ["space", "apostrophe", "minus", "backspace"]


def _text_field_char(key: str) -> str | None:
    if key == "space":
        return " "
    if key == "apostrophe":
        return "'"
    if key == "minus":
        return "-"
    if len(key) == 1:
        return key
    return None


def _field_visible(field, values) -> bool:
    """A field with no show_if is always visible. A field with
    show_if=(dep_field_id, required_value) is only visible once that other
    field has been answered with exactly that value -- e.g. a "please
    specify" text field that only appears if gender == 'Prefer to
    self-describe'. Requires the dependency to appear earlier in `fields`."""
    show_if = field[4]
    if show_if is None:
        return True
    dep_id, dep_value = show_if
    return values.get(dep_id) == dep_value


def demographics_form(win, fields):
    """Draws several demographic fields on one screen and collects them all
    together, instead of one screen per question -- shortens a small
    demographic block (e.g. gender, education, native language) to a single
    fill-in screen per PI request, rather than one screen per item.

    Each field is (field_id, prompt, kind, options, show_if):
      kind="choice" -- options is a list of strings; answered by pressing the
      corresponding number key (1..len(options)), shown inline.
      kind="dropdown" -- options is a list of strings; the field starts as a
      closed box ("Select... (v)"), click (or a number key, which both
      opens and answers in one step) to pick. Used instead of "choice" when
      the option list is long enough that showing it inline would overflow
      the screen (e.g. a language list) -- see education/native_language.
      kind="text" -- options is ignored; free-text entry (letters, space,
      apostrophe, hyphen, backspace), submitted with ENTER (empty allowed, so
      a field can be left blank rather than forcing an answer). Drawn with a
      highlighted border box while active.
      show_if -- None, or (dependency_field_id, required_value): see
      _field_visible. A hidden field is simply never shown and never needs
      answering; its returned value stays None.

    Fields are answered top-to-bottom, mirroring rating_scale_multi: only the
    first unanswered *visible* field is active (live/highlighted); answered
    fields show their locked-in value, later fields stay dimmed until their
    turn. The visible set is recomputed every frame, so a conditional field
    (e.g. gender's self-describe follow-up) appears immediately once its
    dependency is answered.

    Returns {field_id: (value:str|None, rt:float|None)} for every field
    (including hidden/not-applicable ones, whose value is just None).
    """
    values: dict[str, str | None] = {fid: None for fid, *_ in fields}
    rts: dict[str, float | None] = {fid: None for fid, *_ in fields}
    text_buffers: dict[str, str] = {fid: "" for fid, _, kind, *_ in fields if kind == "text"}
    dropdown_open: dict[str, bool] = {fid: False for fid, _, kind, *_ in fields if kind == "dropdown"}
    # Scroll offset (index of the first option shown) per dropdown field --
    # long option lists (e.g. native_language's 13 entries) previously drew
    # every option stacked with no limit, so entries past the window edge
    # were rendered but unreachable by mouse. Only MAX_VISIBLE_OPTIONS show
    # at once; mouse wheel / up-down arrows move the window over the list.
    dropdown_scroll: dict[str, int] = {fid: 0 for fid, _, kind, *_ in fields if kind == "dropdown"}
    MAX_VISIBLE_OPTIONS = 6

    row_gap = 0.16
    box_w, box_h, opt_h = 0.5, 0.06, 0.044
    field_x = -0.6
    clock = core.Clock()
    mouse = event.Mouse(win=win)
    _install_mouse_click_capture(win)
    # See fixation_cross's comment: flush stale keys from the previous screen
    # so they can't be instantly (and invisibly) consumed as soon as this
    # loop starts polling. consume_click() likewise drops any click carried
    # over from the previous screen.
    event.clearEvents()
    consume_click()

    while True:
        check_quit()
        visible = [f for f in fields if _field_visible(f, values)]
        top_y = (len(visible) - 1) * row_gap / 2 + 0.1
        active_index = next((i for i, (fid, *_rest) in enumerate(visible) if values[fid] is None), None)
        if active_index is None:
            return {fid: (values[fid], rts[fid]) for fid, *_ in fields}

        active_id, _active_prompt, active_kind, active_options, _ = visible[active_index]
        mx, my = mouse.getPos()
        # See _install_mouse_click_capture: a getPressed() edge-sample can
        # miss a fast click entirely; consume_click() can't.
        clicked = consume_click()
        active_dropdown_option_rows = []  # (option, y) for the active field's open dropdown, if any
        active_dropdown_scroll_hint = None  # (has_more_above, has_more_below, box_y), if the active dropdown is open
        wheel_dx, wheel_dy = mouse.getWheelRel()

        for i, (fid, prompt, kind, options, _show_if) in enumerate(visible):
            y = top_y - i * row_gap
            is_active = i == active_index
            is_answered = values[fid] is not None

            visual.TextStim(
                win, text=prompt, height=0.038, bold=True, pos=(field_x, y + 0.045),
                color=(TEXT_COLOR if (is_active or is_answered) else MUTED_COLOR),
                alignText="left", anchorHoriz="left",
            ).draw()

            if is_answered:
                shown = values[fid] if values[fid] else "(skipped)"
                visual.TextStim(
                    win, text=shown, height=0.034, pos=(field_x, y), color=ACCENT_COLOR,
                    alignText="left", anchorHoriz="left",
                ).draw()
                continue

            if kind == "choice":
                opts_text = "   ".join(f"{j + 1}) {opt}" for j, opt in enumerate(options))
                visual.TextStim(
                    win, text=opts_text, height=0.03, wrapWidth=1.5, pos=(field_x, y),
                    color=(TEXT_COLOR if is_active else MUTED_COLOR),
                    alignText="left", anchorHoriz="left",
                ).draw()

            elif kind == "dropdown":
                box_y = y - 0.01
                is_open = is_active and dropdown_open.get(fid, False)
                visual.Rect(
                    win, width=box_w, height=box_h, pos=(field_x + box_w / 2, box_y),
                    fillColor=PANEL_COLOR, lineColor=(ACCENT_COLOR if is_active else BORDER_COLOR),
                    lineWidth=2 if is_active else 1,
                ).draw()
                visual.TextStim(
                    win, text="Select...  ▾", height=0.03, pos=(field_x + 0.015, box_y),
                    color=(TEXT_COLOR if is_active else MUTED_COLOR), alignText="left", anchorHoriz="left",
                ).draw()
                if is_active and (clicked and abs(mx - (field_x + box_w / 2)) <= box_w / 2 and abs(my - box_y) <= box_h / 2):
                    dropdown_open[fid] = not dropdown_open[fid]
                    if dropdown_open[fid]:
                        dropdown_scroll[fid] = 0
                if is_open:
                    max_scroll = max(0, len(options) - MAX_VISIBLE_OPTIONS)
                    if wheel_dy:
                        # Scrolling up (positive wheel_dy) reveals earlier
                        # (lower-index) options, so it decreases the offset.
                        dropdown_scroll[fid] -= int(wheel_dy)
                    dropdown_scroll[fid] = max(0, min(max_scroll, dropdown_scroll[fid]))
                    scroll = dropdown_scroll[fid]
                    visible_options = options[scroll:scroll + MAX_VISIBLE_OPTIONS]
                    # Option rows are collected here and drawn LAST (after
                    # every other row, below) so the overlay always renders
                    # on top of whatever field rows sit underneath it.
                    for j, opt in enumerate(visible_options):
                        oy = box_y - box_h / 2 - opt_h / 2 - j * opt_h
                        active_dropdown_option_rows.append((opt, oy))
                    active_dropdown_scroll_hint = (scroll > 0, scroll + MAX_VISIBLE_OPTIONS < len(options), box_y)

            else:  # text, unanswered -- drawn with a highlighted box so it's
                # obviously an active input, per the "highlighted text box"
                # request for the language 'Other' follow-up (applied to all
                # active text fields for visual consistency).
                buffer = text_buffers[fid]
                cursor = "_" if is_active and int(core.getTime() / 0.4) % 2 == 0 else ""
                if is_active:
                    visual.Rect(
                        win, width=box_w, height=box_h, pos=(field_x + box_w / 2, y - 0.01),
                        fillColor=PANEL_COLOR, lineColor=ACCENT_COLOR, lineWidth=2,
                    ).draw()
                visual.TextStim(
                    win, text=f"{buffer}{cursor}", height=0.034, pos=(field_x + 0.015, y - 0.01),
                    color=(TEXT_COLOR if is_active else MUTED_COLOR),
                    alignText="left", anchorHoriz="left",
                ).draw()

        hint = {
            "choice": "Press a number key to answer.",
            "dropdown": "Click the box (or press a number key) to choose.",
            "text": "Type your answer, then press ENTER (leave blank + ENTER to skip).",
        }[active_kind]
        if active_kind == "dropdown" and dropdown_open.get(active_id, False) and len(active_options) > MAX_VISIBLE_OPTIONS:
            hint += " Scroll or press ↑/↓ for more options."
        visual.TextStim(win, text=hint, height=0.028, pos=(0, top_y - len(visible) * row_gap + 0.02), color=MUTED_COLOR).draw()

        # Drawn last so the open dropdown's option list overlays every other
        # row instead of being hidden behind them.
        for opt, oy in active_dropdown_option_rows:
            hovered = abs(mx - (field_x + box_w / 2)) <= box_w / 2 and abs(my - oy) <= opt_h / 2
            visual.Rect(
                win, width=box_w, height=opt_h, pos=(field_x + box_w / 2, oy),
                fillColor=(ACCENT_COLOR if hovered else PANEL_COLOR), lineColor=BORDER_COLOR, lineWidth=1,
            ).draw()
            visual.TextStim(
                win, text=opt, height=0.028, pos=(field_x + 0.015, oy),
                color=(BG_COLOR if hovered else TEXT_COLOR), alignText="left", anchorHoriz="left",
            ).draw()
        if active_dropdown_scroll_hint is not None:
            has_more_above, has_more_below, box_y = active_dropdown_scroll_hint
            arrow_x = field_x + box_w - 0.02
            if has_more_above:
                visual.TextStim(
                    win, text="▲", height=0.024, pos=(arrow_x, box_y - box_h / 2 - opt_h * 0.4),
                    color=MUTED_COLOR, alignText="center", anchorHoriz="center",
                ).draw()
            if has_more_below:
                last_row_y = box_y - box_h / 2 - opt_h * (0.5 + (len(active_dropdown_option_rows) - 1))
                visual.TextStim(
                    win, text="▼", height=0.024, pos=(arrow_x, last_row_y - opt_h * 0.4),
                    color=MUTED_COLOR, alignText="center", anchorHoriz="center",
                ).draw()

        win.flip()

        if active_kind == "choice":
            # Capped at 9: psychopy's digit keys are single characters
            # ("1".."9"), so options past the 9th are mouse/click-only --
            # doesn't come up for choice fields today (max 5 options) but
            # guards against silently-dead key names if that changes.
            key_list = [str(i + 1) for i in range(min(len(active_options), 9))]
            keys = event.getKeys(keyList=key_list, timeStamped=clock)
            if keys:
                key, rt = keys[0]
                values[active_id] = active_options[int(key) - 1]
                rts[active_id] = rt

        elif active_kind == "dropdown":
            key_list = [str(i + 1) for i in range(min(len(active_options), 9))]
            keys = event.getKeys(keyList=key_list + ["up", "down"], timeStamped=clock)
            if keys:
                key, rt = keys[0]
                if key == "up":
                    dropdown_scroll[active_id] -= 1
                elif key == "down":
                    dropdown_scroll[active_id] += 1
                else:
                    values[active_id] = active_options[int(key) - 1]
                    rts[active_id] = rt
                    dropdown_open[active_id] = False
            elif clicked and dropdown_open.get(active_id, False):
                for opt, oy in active_dropdown_option_rows:
                    if abs(mx - (field_x + box_w / 2)) <= box_w / 2 and abs(my - oy) <= opt_h / 2:
                        values[active_id] = opt
                        rts[active_id] = clock.getTime()
                        dropdown_open[active_id] = False
                        break

        else:
            keys = event.getKeys(keyList=_TEXT_FIELD_KEYS + ["return"], timeStamped=clock)
            for key, rt in keys:
                if key == "return":
                    values[active_id] = text_buffers[active_id]
                    rts[active_id] = rt
                    break
                if key == "backspace":
                    text_buffers[active_id] = text_buffers[active_id][:-1]
                else:
                    ch = _text_field_char(key)
                    if ch:
                        text_buffers[active_id] += ch


def discrete_choice(
    win, prompt: str, options: list[str], timeout: float | None = None, allow_skip: bool = False,
    require_confirm: bool = False,
):
    """Participant presses 1..len(options), or clicks the option with the mouse.
    If allow_skip, SKIP_BLOCK_KEY/SKIP_TRIAL_KEY are also live and raise
    SkipBlock/SkipTrial -- off by default since most callers (e.g. the consent
    questionnaire) have no skip semantics for those keys.

    If require_confirm, picking an option (key or click) only highlights it
    as the pending choice instead of returning immediately -- picking a
    different option changes the pending choice, and ENTER/SPACE confirms
    and returns it. Off by default (immediate select-and-return) to leave
    existing callers (age range, handedness, etc.) unchanged; the emotion
    category screen turns it on so a mis-click/mis-press can be corrected
    before it's locked in, rather than instantly advancing the screen.

    Returns (choice:str|None, rt:float|None).
    """
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
    _install_mouse_click_capture(win)
    # See fixation_cross's comment: flush stale keys from whatever screen was
    # up before this one, so they can't get instantly (and invisibly)
    # consumed as soon as this loop starts polling. consume_click() likewise
    # drops any click carried over from the previous screen.
    event.clearEvents()
    consume_click()

    selected: int | None = None
    hint = visual.TextStim(
        win, text="Press ENTER/SPACE to confirm.", height=0.03, pos=(0, bottom_y - 0.08), color=MUTED_COLOR,
    )

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
            if i == selected:
                color = ACCENT_COLOR
            elif i == hovered:
                color = TEXT_COLOR if require_confirm else ACCENT_COLOR
            else:
                color = TEXT_COLOR
            visual.TextStim(
                win, text=f"{i + 1}.  {opt}", height=0.045, pos=(-0.5, y), color=color,
                bold=(i == selected), alignText="left", anchorHoriz="left",
            ).draw()
        if require_confirm and selected is not None:
            hint.draw()
        if allow_skip:
            draw_key_hint(win)
        win.flip()

        if timeout is not None and clock.getTime() > timeout:
            return None, None

        if require_confirm and selected is not None:
            confirm_keys = event.getKeys(keyList=["space", "return"], timeStamped=clock)
            if confirm_keys:
                _, rt = confirm_keys[0]
                return options[selected], rt

        keys = event.getKeys(keyList=key_list, timeStamped=clock)
        if keys:
            key, rt = keys[0]
            if require_confirm:
                selected = int(key) - 1
            else:
                return options[int(key) - 1], rt
            continue

        # See _install_mouse_click_capture: a getPressed() edge-sample can
        # miss a fast click entirely; consume_click() can't.
        if consume_click() and hovered is not None:
            if require_confirm:
                selected = hovered
            else:
                return options[hovered], clock.getTime()
