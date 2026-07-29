from psychopy import core, event, visual


class UserQuit(Exception):
    """Raised when the operator presses Escape to abort a session early."""


def check_quit() -> None:
    if event.getKeys(keyList=["escape"]):
        raise UserQuit()


def show_message(win, text, duration=None, wait_key=None, font_height=0.06):
    stim = visual.TextStim(win, text=text, height=font_height, wrapWidth=1.6, color="white")
    stim.draw()
    win.flip()
    check_quit()
    if wait_key:
        keys = event.waitKeys(keyList=list(wait_key) + ["escape"])
        if keys and "escape" in keys:
            raise UserQuit()
    elif duration is not None:
        core.wait(duration)


def fixation_cross(win, duration):
    cross = visual.TextStim(win, text="+", height=0.12, color="white")
    cross.draw()
    win.flip()
    check_quit()
    core.wait(duration)


def draw_countdown_bar(win, fraction_remaining: float, hurry_up: bool = False):
    fraction_remaining = max(0.0, min(1.0, fraction_remaining))
    full_width = 1.6
    width = max(full_width * fraction_remaining, 0.001)
    color = "red" if hurry_up else "palegreen"
    bar = visual.Rect(
        win,
        width=width,
        height=0.06,
        pos=(-0.8 + width / 2, 0.85),
        fillColor=color,
        lineColor=None,
    )
    bar.draw()
    if hurry_up:
        hurry_text = visual.TextStim(win, text="HURRY UP!", height=0.07, pos=(0, 0.7), color="red", bold=True)
        hurry_text.draw()


def draw_leaderboard(win, names, scores, pos=(0.75, 0.0)):
    lines = "\n".join(f"{i + 1}. {n:<10} {s}" for i, (n, s) in enumerate(zip(names, scores)))
    text = visual.TextStim(
        win,
        text=f"LEADERBOARD\n{lines}",
        height=0.04,
        pos=pos,
        color="yellow",
        alignText="left",
        anchorHoriz="left",
    )
    text.draw()


def rating_scale_0_7(win, prompt: str, scale_min: int = 0, scale_max: int = 7, timeout: float | None = None):
    """0..7 discrete rating via number keys. Returns (value:int|None, rt:float|None)."""
    key_list = [str(i) for i in range(scale_min, scale_max + 1)]
    stim = visual.TextStim(
        win,
        text=f"{prompt}\n\n{scale_min} (not at all)  ...  {scale_max} (extremely)\n\nPress a number key {scale_min}-{scale_max}",
        height=0.06,
        wrapWidth=1.6,
        color="white",
    )
    clock = core.Clock()
    stim.draw()
    win.flip()
    clock.reset()
    keys = event.waitKeys(keyList=key_list + ["escape"], timeStamped=clock, maxWait=timeout if timeout else float("inf"))
    if not keys:
        return None, None
    key, rt = keys[0]
    if key == "escape":
        raise UserQuit()
    return int(key), rt


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
        stim = visual.TextStim(win, text=f"{prompt}\n\n{buffer}", height=0.06, wrapWidth=1.6, color="white")
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


def discrete_choice(win, prompt: str, options: list[str], timeout: float | None = None):
    """Participant presses 1..len(options). Returns (choice:str|None, rt:float|None)."""
    lines = "\n".join(f"{i + 1}. {opt}" for i, opt in enumerate(options))
    stim = visual.TextStim(
        win,
        text=f"{prompt}\n\n{lines}",
        height=0.06,
        wrapWidth=1.6,
        color="white",
    )
    clock = core.Clock()
    stim.draw()
    win.flip()
    clock.reset()
    valid_keys = [str(i + 1) for i in range(len(options))]
    keys = event.waitKeys(keyList=valid_keys + ["escape"], timeStamped=clock, maxWait=timeout if timeout else float("inf"))
    if not keys:
        return None, None
    key, rt = keys[0]
    if key == "escape":
        raise UserQuit()
    return options[int(key) - 1], rt
