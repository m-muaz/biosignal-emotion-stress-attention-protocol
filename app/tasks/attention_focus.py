"""Schulte table + Stroop color-word test -- two selective-attention/
interference-control games added per PI discussion 2026-08-04, taking over
attention_highway_task's slot in app/tasks/session_shell.py's flow (see
session_config.yaml's attention_highway_task comment for why that module
itself is left untouched rather than deleted).

Modeled on https://www.freefocusgames.com/games/schulte-table and
.../stroop-effect-test, adapted for lab data collection:
  - Schulte: click the numbers 1..N (N = grid_size^2) in ascending order,
    scattered across a grid. Explicitly UNTIMED -- no countdown, no response
    deadline -- completion time is the dependent measure, not a race against
    a clock (per PI request 2026-08-04).
  - Stroop: a color word rendered in an "ink" color; click the on-screen
    color swatch matching the INK, not the word. Congruent (word == ink) vs
    incongruent (word != ink) trials are interleaved; reaction time is the
    dependent measure, so this is self-paced too (no response deadline).
    Response is a click on a fixed-position, unlabeled color swatch (not a
    text button) -- deliberately avoids adding a SECOND word-reading
    conflict at the response stage, and keeping the swatches in the same
    on-screen position every trial minimizes the Fitts'-Law reach-time
    variance a click response (vs. a keypress) otherwise adds to the RT
    measurement.

Neither game uses difficulty tiers, unlike stress_task/attention_highway_task
-- per PI request 2026-08-04, both run at one fixed, comparable difficulty,
since the point here is measuring attention/focus under a known paradigm,
not a difficulty ramp. (Every knob is still config-overridable in
session_config.yaml's attention_focus_task, so a tiered version could be
added later without a code change to this file.)

Session structure: attention_focus_task.trials_per_game trials of EACH game
(2 * trials_per_game total blocks), each trial = one baseline (fixation) +
one game round. The across-trial ORDER (which game comes up in which slot)
is shuffled once per session via ctx.rng (build_sequence(), called once by
app/tasks/session_shell.py at session start) -- so which game a participant
sees at trial N isn't confounded with a fixed sequence position, same
reasoning the emotion task randomizes clip order while keeping clip
SELECTION fixed. Each individual game's CONTENT (Schulte's number layout,
Stroop's stimulus list) is also pre-generated at that same point, via the
same ctx.rng, so all of this session's randomness draws stay traceable to
the one rng_seed already recorded in session_manifest.json.

Renders in a pywebview subprocess per trial (app/webui/schulte_app.py /
app/webui/stroop_app.py), same out-of-process-from-psychopy architecture as
every other web-based task -- see app/webui/bridge.py for why.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

from app.ui.common_widgets import UserQuit
from app.webui.bridge import QUIT_EXIT_CODE, save_json_atomic

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def _launch(module: str, win, ctx, bootstrap: dict, practice: bool) -> None:
    """`win` is the pywebview session-shell window (hidden/shown around the
    subprocess so it doesn't compete for the foreground), or None when
    there's no parent window to hide (e.g. app/run_task.py's isolated
    single-task testing). `module` is the dotted subprocess entry point
    (app.webui.schulte_app or app.webui.stroop_app)."""
    if win is not None:
        win.hide()
    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            bootstrap_path = Path(tmp_dir) / "bootstrap.json"
            save_json_atomic(bootstrap_path, bootstrap)

            args = [
                sys.executable, "-m", module,
                "--session-dir", str(ctx.event_logger.session_dir),
                "--session-id", ctx.event_logger.session_id,
                "--participant-id", ctx.participant_id,
                "--bootstrap", str(bootstrap_path),
            ]
            if practice:
                args.append("--practice")

            result = subprocess.run(args)
    finally:
        if win is not None:
            win.show()

    if result.returncode == QUIT_EXIT_CODE:
        raise UserQuit()
    if result.returncode != 0:
        raise RuntimeError(f"{module} exited with code {result.returncode}")


def _build_schulte_bootstrap(
    schulte_cfg: dict, rng, baseline_duration_sec: float, trial_index: int | None, total_trials: int | None,
    grid_size: int | None = None, practice: bool = False,
) -> dict:
    grid_size = grid_size if grid_size is not None else schulte_cfg["grid_size"]
    numbers = list(range(1, grid_size * grid_size + 1))
    rng.shuffle(numbers)  # cell i (row-major) holds numbers[i]
    return {
        "practice": practice,
        "demo": practice,
        "grid_size": grid_size,
        "layout": numbers,
        "show_target_hint": schulte_cfg.get("show_target_hint", True),
        "baseline_duration_sec": baseline_duration_sec,
        "trial_index": trial_index,
        "total_trials": total_trials,
    }


def _build_stroop_bootstrap(
    stroop_cfg: dict, rng, baseline_duration_sec: float, trial_index: int | None, total_trials: int | None,
    n_trials: int | None = None, practice: bool = False,
) -> dict:
    colors = stroop_cfg["colors"]
    n_trials = n_trials if n_trials is not None else stroop_cfg["trials_per_block"]
    n_congruent = round(n_trials * stroop_cfg.get("congruent_fraction", 0.5))
    congruent_flags = [True] * n_congruent + [False] * (n_trials - n_congruent)
    rng.shuffle(congruent_flags)

    stimuli = []
    for congruent in congruent_flags:
        ink = rng.choice(colors)
        if congruent:
            word = ink
        else:
            word = rng.choice([c for c in colors if c["name"] != ink["name"]])
        stimuli.append({"word": word["name"], "ink_color": ink["name"], "congruent": congruent})

    return {
        "practice": practice,
        "demo": practice,
        "colors": colors,
        "stimuli": stimuli,
        "isi_sec": stroop_cfg.get("isi_sec", 0.5),
        "baseline_duration_sec": baseline_duration_sec,
        "trial_index": trial_index,
        "total_trials": total_trials,
    }


def build_sequence(ctx) -> list[dict]:
    """Shuffles which game comes up in each of the 2*trials_per_game slots,
    and pre-generates every trial's full content (Schulte layout / Stroop
    stimulus list) up front -- called once by app/tasks/session_shell.py at
    session start (same point emotion/stress pre-generate their own
    content), so all the randomness for this task traces to that one call.
    Returns a list of {"game", "trial_index", "bootstrap"} dicts, one per
    trial, in presentation order."""
    cfg = ctx.config["attention_focus_task"]
    n = cfg["trials_per_game"]
    games = ["schulte"] * n + ["stroop"] * n
    ctx.rng.shuffle(games)

    baseline_duration_sec = ctx.scaled(cfg["baseline_duration_sec"])
    total_trials = len(games)
    trials = []
    for trial_index, game in enumerate(games):
        if game == "schulte":
            bootstrap = _build_schulte_bootstrap(
                cfg["schulte"], ctx.rng, baseline_duration_sec, trial_index, total_trials,
            )
        else:
            bootstrap = _build_stroop_bootstrap(
                cfg["stroop"], ctx.rng, baseline_duration_sec, trial_index, total_trials,
            )
        trials.append({"game": game, "trial_index": trial_index, "bootstrap": bootstrap})
    return trials


def run_trial(win, ctx, trial_spec: dict) -> None:
    """Runs one trial (baseline + one game round) from a `trials` entry
    returned by build_sequence()."""
    if trial_spec["game"] == "schulte":
        _launch("app.webui.schulte_app", win, ctx, trial_spec["bootstrap"], practice=False)
    else:
        _launch("app.webui.stroop_app", win, ctx, trial_spec["bootstrap"], practice=False)


def run_attention_focus_task(win, ctx) -> None:
    """Runs the full trial sequence back-to-back -- used by
    `python -m app.run_task --task attention_focus` for isolated testing;
    app/tasks/session_shell.py instead calls build_sequence() once up front
    and run_trial() per trial, interleaving each with the shell's own
    per-trial instruction screen (see app/webui/shell/shell.js)."""
    for trial_spec in build_sequence(ctx):
        run_trial(win, ctx, trial_spec)


def run_demo_schulte(win, ctx) -> None:
    """Non-interactive Schulte preview for the session shell's
    familiarization flow -- a SMALLER grid (schulte.demo_grid_size) that
    solves itself (app/webui/schulte/game.js's bootstrap.demo path) so the
    participant watches the click-in-order mechanic. No baseline shown."""
    cfg = ctx.config["attention_focus_task"]
    bootstrap = _build_schulte_bootstrap(
        cfg["schulte"], ctx.rng, baseline_duration_sec=0.0, trial_index=None, total_trials=None,
        grid_size=cfg["schulte"]["demo_grid_size"], practice=True,
    )
    _launch("app.webui.schulte_app", win, ctx, bootstrap, practice=True)


def run_demo_stroop(win, ctx) -> None:
    """Non-interactive Stroop preview -- demo_trials auto-answered stimuli
    (app/webui/stroop/game.js's bootstrap.demo path) so the participant
    watches the click-the-ink-color mechanic. No baseline shown."""
    cfg = ctx.config["attention_focus_task"]
    bootstrap = _build_stroop_bootstrap(
        cfg["stroop"], ctx.rng, baseline_duration_sec=0.0, trial_index=None, total_trials=None,
        n_trials=cfg["stroop"]["demo_trials"], practice=True,
    )
    _launch("app.webui.stroop_app", win, ctx, bootstrap, practice=True)
