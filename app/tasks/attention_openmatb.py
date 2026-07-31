"""Task 1: attention/focus, via OpenMATB (juliencegarra/OpenMATB) run as a subprocess.

Replaces the old spatial n-back (see attention_nback.py, retired -- its
discrete-trial structure couldn't record *continuous* attention/focus).
OpenMATB is a separate pyglet application (not PsychoPy) with its own window
and blocking event loop, so it can't be embedded in our PsychoPy window --
instead we generate a scenario + config.ini, launch it as a subprocess with
our own window closed, and reopen our window once it exits.

Only `sysmon` (system monitoring) and `resman` (resource management) are
used: both are entirely keyboard-driven (F1-F6, NUM_1-NUM_8). `track`
(tracking) requires a physical joystick with no keyboard/mouse fallback
anywhere in OpenMATB's code, and this protocol has no joystick.

OpenMATB logs its own CSV (sessions/<date>/<id>_<timestamp>.csv) timestamped
with per-process perf_counter(), not wall clock, so it can't be merged
directly with our host-clock event log. Instead -- same philosophy as the
BLE-synced wearables (Engineering_Document.md §2/§5.1) -- we log task_start/
task_end on the host clock around the subprocess call and copy OpenMATB's
own CSV into the session dir for provenance; fine-grained alignment can add
task_start + scenario_time as an approximation, since exact sub-second sync
here isn't the precision-critical path (that's the wearables' BLE sync).
"""

from __future__ import annotations

import platform
import shutil
import subprocess
import time
from pathlib import Path

from psychopy import visual

from app.ui.common_widgets import BG_COLOR, fixation_cross, show_message

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# (gauge_kind, gauge_index) -- the 6 sysmon gauges a failure can be triggered on.
_SYSMON_GAUGES = [("scales", "1"), ("scales", "2"), ("scales", "3"), ("scales", "4"), ("lights", "1"), ("lights", "2")]


class OpenMATBNotInstalled(RuntimeError):
    pass


def _resolve_install_path(openmatb_cfg: dict) -> Path:
    raw = openmatb_cfg.get("install_path", "vendor/OpenMATB")
    path = Path(raw)
    if not path.is_absolute():
        path = REPO_ROOT / path
    if not (path / "main.py").exists():
        raise OpenMATBNotInstalled(
            f"OpenMATB not found at {path}.\n\n"
            f'Clone it there: git clone https://github.com/juliencegarra/OpenMATB "{path}"\n\n'
            "See README.md 'Attention task (OpenMATB)' for the rest of setup."
        )
    return path


def _resolve_venv_python(install_path: Path) -> Path:
    """OpenMATB needs pyglet>=2.1,<3 -- a backwards-incompatible rewrite that
    breaks PsychoPy's pyglet==1.4.11-pinned window backend on Windows -- so it
    must run in its own venv, never this project's conda env. Matches
    OpenMATB's own README convention (a `.venv` inside its checkout).
    """
    is_windows = platform.system() == "Windows"
    venv_python = install_path / ".venv" / ("Scripts" if is_windows else "bin") / ("python.exe" if is_windows else "python3")
    if not venv_python.exists():
        activate_hint = r".venv\Scripts\python.exe -m pip install -r requirements.txt" if is_windows else \
            ".venv/bin/python3 -m pip install -r requirements.txt"
        raise OpenMATBNotInstalled(
            f"OpenMATB's isolated venv not found at {install_path / '.venv'}.\n\n"
            "OpenMATB requires pyglet 2.x, which conflicts with PsychoPy's pinned "
            "pyglet 1.4.11 -- it must run in its own environment, not this "
            f"project's conda env. Create it once, from {install_path}:\n\n"
            f"  python -m venv .venv\n  {activate_hint}\n\n"
            "See README.md 'Attention task (OpenMATB)'."
        )
    return venv_python


def _hms(seconds: float) -> str:
    """OpenMATB scenario time format: H:MM:SS, integer seconds (core/event.py)."""
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def _generate_sysmon_failures(rng, duration_sec: int, interval_range: tuple[float, float], alerttimeout_ms: int):
    """Pseudo-random failure onsets across the block, one gauge at a time.

    A gauge is never re-triggered until its previous failure would have
    auto-cleared (alerttimeout) -- avoids stacking unresolved failures on
    the same gauge, which the participant has no way to "catch up" on.
    """
    alert_sec = alerttimeout_ms / 1000
    lo, hi = interval_range
    next_free = {g: 0.0 for g in _SYSMON_GAUGES}
    events: list[tuple[float, str, str]] = []
    t = rng.uniform(lo, hi)
    while t < duration_sec - alert_sec:
        available = [g for g in _SYSMON_GAUGES if next_free[g] <= t]
        if not available:
            t += 1.0
            continue
        gauge = rng.choice(available)
        kind, idx = gauge
        events.append((t, f"{kind}-{idx}-failure", "True"))
        next_free[gauge] = t + alert_sec
        t += rng.uniform(lo, hi)
    return events


def _build_scenario_text(rng, duration_sec: int, openmatb_cfg: dict) -> str:
    interval_range = tuple(openmatb_cfg.get("sysmon_failure_interval_range_sec", [10, 20]))
    alerttimeout_ms = openmatb_cfg.get("sysmon_alerttimeout_ms", 10000)
    feedback_ms = openmatb_cfg.get("sysmon_feedback_duration_ms", 1500)

    # t=0 rows are listed params-then-start (list order, not just time, breaks
    # ties for simultaneous events -- see core/scenario.py's get_event_at_scenario_time).
    rows: list[tuple[float, str, str, str | None]] = [
        (0, "sysmon", "alerttimeout", str(alerttimeout_ms)),
        (0, "sysmon", "feedbackduration", str(feedback_ms)),
        (0, "sysmon", "start", None),
        (0, "resman", "start", None),
    ]
    for t, param, value in _generate_sysmon_failures(rng, duration_sec, interval_range, alerttimeout_ms):
        rows.append((t, "sysmon", param, value))
    rows.append((duration_sec, "sysmon", "stop", None))
    rows.append((duration_sec, "resman", "stop", None))
    rows.sort(key=lambda row: row[0])

    lines = []
    for t, plugin, cmd, value in rows:
        hms = _hms(t)
        lines.append(f"{hms};{plugin};{cmd}" if value is None else f"{hms};{plugin};{cmd};{value}")
    return "\n".join(lines) + "\n"


def _write_config_ini(install_path: Path, openmatb_cfg: dict, scenario_rel: str) -> None:
    # Written by hand (not configparser.write(), which pads "key = value") --
    # OpenMATB's own main.py locates the language line via a literal
    # "language=" substring match, before its config parser is even loaded.
    lines = [
        "[Openmatb]",
        f"language={openmatb_cfg.get('language', 'en_EN')}",
        f"screen_index={openmatb_cfg.get('screen_index', 0)}",
        "font_name=",
        f"fullscreen={'True' if openmatb_cfg.get('fullscreen', True) else 'False'}",
        f"scenario_path={scenario_rel}",
        "display_session_number=True",
        "hide_on_pause=False",
        "highlight_aoi=False",
        "top_bounds=[0.35, 0.85]",
        "bottom_bounds=[0.30, 0.85]",
        "",
    ]
    (install_path / "config.ini").write_text("\n".join(lines), encoding="utf-8")


def _run_subprocess(install_path: Path, venv_python: Path, duration_sec: float) -> tuple[int, str, str]:
    try:
        proc = subprocess.run(
            [str(venv_python), "main.py"],
            cwd=str(install_path),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=duration_sec + 90,
        )
        return proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as exc:
        return -1, exc.stdout or "", (exc.stderr or "") + "\nTIMEOUT: OpenMATB subprocess did not exit in time."


def _read_scenario_errors(install_path: Path) -> str | None:
    path = install_path / "last_scenario_errors.log"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="replace")


def _copy_latest_log(install_path: Path, since_epoch: float, session_dir: Path, dest_name: str) -> Path | None:
    sessions_dir = install_path / "sessions"
    if not sessions_dir.exists():
        return None
    candidates = [p for p in sessions_dir.rglob("*.csv") if p.stat().st_mtime >= since_epoch - 5]
    if not candidates:
        return None
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    dest = Path(session_dir) / dest_name
    shutil.copy2(latest, dest)
    return dest


def _run_openmatb_block(
    ctx, install_path: Path, venv_python: Path, openmatb_cfg: dict, duration_sec: float, task_label: str, run_label: str
) -> None:
    duration_sec = int(round(duration_sec))
    scenario_text = _build_scenario_text(ctx.rng, duration_sec, openmatb_cfg)
    scenario_rel = f"generated/{ctx.participant_id}_{task_label}_{run_label}.txt"
    scenario_abs = install_path / "includes" / "scenarios" / scenario_rel
    scenario_abs.parent.mkdir(parents=True, exist_ok=True)
    scenario_abs.write_text(scenario_text, encoding="utf-8")
    _write_config_ini(install_path, openmatb_cfg, scenario_rel)

    session_dir = ctx.event_logger.session_dir
    (session_dir / f"openmatb_{run_label}_scenario.txt").write_text(scenario_text, encoding="utf-8")

    start_host_time = time.time()
    ctx.event_logger.log(
        "openmatb_subprocess_start", task=task_label, run_label=run_label,
        duration_sec=duration_sec, scenario_path=str(scenario_abs),
    )

    returncode, _stdout, stderr = _run_subprocess(install_path, venv_python, duration_sec)

    ctx.event_logger.log(
        "openmatb_subprocess_end", task=task_label, run_label=run_label,
        returncode=returncode, wall_duration_sec=time.time() - start_host_time,
    )
    if returncode != 0:
        ctx.event_logger.log("openmatb_subprocess_error", task=task_label, run_label=run_label, stderr=stderr[-4000:])

    errors_text = _read_scenario_errors(install_path)
    if errors_text and errors_text.strip() != "No error":
        ctx.event_logger.log("openmatb_scenario_errors", task=task_label, run_label=run_label, detail=errors_text.strip())

    dest = _copy_latest_log(install_path, start_host_time, session_dir, f"openmatb_{run_label}_log.csv")
    ctx.event_logger.log(
        "openmatb_log_copied", task=task_label, run_label=run_label,
        source_found=dest is not None, dest_path=str(dest) if dest else None,
    )

    scenario_abs.unlink(missing_ok=True)


def run_attention_task(win, ctx):
    """Runs the attention/focus task. Closes `win` and returns a NEW Window
    (OpenMATB owns its own window/GL context while its subprocess is alive) --
    callers must reassign their `win` reference to the return value.
    """
    cfg = ctx.config["attention_task"]
    openmatb_cfg = cfg["openmatb"]

    try:
        install_path = _resolve_install_path(openmatb_cfg)
        venv_python = _resolve_venv_python(install_path)
    except OpenMATBNotInstalled as exc:
        show_message(win, f"SETUP ERROR\n\n{exc}\n\nPress SPACE to abort.", wait_key=["space", "return"])
        raise

    window_kwargs = ctx.window_kwargs or dict(size=(1280, 800), color=BG_COLOR, units="height", fullscr=False)

    show_message(
        win,
        "TASK 1: ATTENTION\n\n"
        "The screen will switch to a full-screen monitoring & resource-management display "
        "(OpenMATB, a widely-used attention/workload task battery).\n\n"
        "MONITORING: six gauges. Press F1-F4 the moment one of the four moving-arrow scales "
        "drifts out of its normal (middle) zone; press F5/F6 the moment the matching light "
        "changes from its normal state.\n\n"
        "RESOURCE MANAGEMENT: two fuel tanks drain over time. Use the NUM keys to toggle pumps "
        "on/off and keep both tanks near their target level.\n\n"
        "Both run continuously and at the same time -- keep an eye on all of it.\n\n"
        "Press SPACE to begin.",
        wait_key=["space", "return"],
    )
    ctx.event_logger.log("task_start", task="attention")

    show_message(win, "Practice round (not scored) -- about a minute.\n\nPress SPACE to begin.", wait_key=["space", "return"])
    fixation_cross(win, ctx.scaled(cfg["cue_duration_sec"]))
    win.close()
    _run_openmatb_block(
        ctx, install_path, venv_python, openmatb_cfg, ctx.scaled(cfg["practice_duration_sec"]), "attention_practice", "practice"
    )

    win = visual.Window(**window_kwargs)
    minutes = int(cfg["block_duration_sec"] // 60)
    show_message(
        win, f"Practice complete. The real (scored) task begins now -- about {minutes} minutes.\n\nPress SPACE to begin.",
        wait_key=["space", "return"],
    )
    fixation_cross(win, ctx.scaled(cfg["cue_duration_sec"]))
    win.close()
    _run_openmatb_block(ctx, install_path, venv_python, openmatb_cfg, ctx.scaled(cfg["block_duration_sec"]), "attention", "main")

    win = visual.Window(**window_kwargs)
    ctx.event_logger.log("task_end", task="attention")
    return win
