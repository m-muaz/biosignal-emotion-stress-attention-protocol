// Schulte table front end. Numbers/layout come entirely from Python
// (app/tasks/attention_focus.py's build_sequence(), using ctx.rng -- so
// provenance stays with the one rng_seed recorded in session_manifest.json)
// -- this file only renders, collects clicks, and logs events back through
// the pywebview bridge (see app/webui/bridge.py).
//
// Explicitly UNTIMED: no countdown, no response deadline. Completion time
// is the dependent measure (reported on the end screen after the fact), not
// a race against a visible clock -- per PI request 2026-08-04.
(() => {
  let bootstrap = null;
  let gridSize = 5;
  let n = 25;
  let layout = [];
  let nextTarget = 1;
  let misclicks = 0;
  let startTime = 0;
  let quitting = false;
  let resolveTrial = null;
  let trialSkipped = false;
  let lastCompletionTimeSec = null;
  let lastMisclicks = 0;

  const el = (id) => document.getElementById(id);

  // See app/webui/shared/clock_sync.js -- shared with every other webui
  // task and the session shell (browser performance.now() tagging on every
  // event, plus sync_checkpoint round trips at session start/end -- see
  // app/eventlog/event_logger.py's timestamp_monotonic and
  // app/webui/bridge.py's sync_checkpoint()).
  const logEvent = ClockSync.wrapLogEvent((eventType, fields) => window.pywebview.api.log_event(eventType, fields));

  function showScreen(name) {
    document.querySelectorAll(".screen").forEach((s) => s.classList.add("hidden"));
    el(name).classList.remove("hidden");
  }

  // Text instructions shown before every trial (and before the demo) --
  // per feedback 2026-08-04, this needs to work whether the subprocess was
  // launched through the session shell (which shows its own instructions
  // beforehand) OR standalone (python -m app.run_task), which has no shell
  // screens at all. Escape here quits before anything's even started; B
  // isn't offered (nothing to skip yet).
  function showInstructions(text, hint) {
    showScreen("instructions-screen");
    el("instructions-text").textContent = text;
    el("instructions-hint").textContent = hint;
    return new Promise((resolve) => {
      function onKey(e) {
        if (e.key === "Escape") { quitting = true; finish(); }
        else if (e.key === " " || e.key === "Enter") finish();
      }
      function finish() {
        document.removeEventListener("keydown", onKey);
        resolve();
      }
      document.addEventListener("keydown", onKey);
    });
  }

  async function main() {
    bootstrap = await window.pywebview.api.get_bootstrap();
    gridSize = bootstrap.grid_size;
    layout = bootstrap.layout;
    n = gridSize * gridSize;

    if (bootstrap.demo) {
      // Non-interactive preview for familiarization: a smaller grid solves
      // itself so the participant watches the click-in-order mechanic --
      // same role as every other task's demo mode.
      el("hud").classList.add("hidden");
      logEvent("task_start", {});
      await showInstructions(
        "PREVIEW -- SCHULTE TABLE\n\nWatch: the grid solves itself automatically so you can see how the game works.",
        "Press SPACE to watch.",
      );
      if (!quitting) await runTrial(0, /* autoSolve */ true);
      logEvent("task_end", {});
      if (quitting) window.pywebview.api.request_quit();
      else window.pywebview.api.close_window();
      return;
    }

    el("trial-hint").textContent = `Attention trial ${bootstrap.trial_index + 1} of ${bootstrap.total_trials}`;
    // Skipped in the demo branch above, same reasoning as every other
    // task's demo mode -- a practice/familiarization run isn't part of the
    // scored dataset.
    await ClockSync.runCheckpoints(logEvent, "session_start");
    logEvent("task_start", {});
    await showInstructions(
      `ATTENTION TRIAL ${bootstrap.trial_index + 1} of ${bootstrap.total_trials} -- SCHULTE TABLE\n\n` +
      `Click the numbers 1 to ${n}, in order, wherever they appear on the grid.\n\n` +
      "There's no time limit -- take the time you need.\n\n" +
      "B = skip this trial   |   Esc = quit",
      "Press SPACE to begin.",
    );

    if (!quitting) {
      await runBaseline(0);
      if (!quitting) await runTrial(1, /* autoSolve */ false);
    }

    logEvent("task_end", {});
    await ClockSync.runCheckpoints(logEvent, "session_end");
    if (quitting) window.pywebview.api.request_quit();
    else showEndScreen();
  }

  // Shared by baseline: resolves when `durationMs` elapses, OR the
  // participant presses B (skip block) or Escape (full quit) -- same
  // convention as every other task's waitForBlock().
  function waitForBlock(durationMs) {
    return new Promise((resolve) => {
      let skipped = false;
      const timer = setTimeout(finish, durationMs);
      function onKey(e) {
        if (e.key === "Escape") { quitting = true; finish(); }
        else if (e.key.toLowerCase() === "b") { skipped = true; finish(); }
      }
      function finish() {
        clearTimeout(timer);
        document.removeEventListener("keydown", onKey);
        resolve(skipped);
      }
      document.addEventListener("keydown", onKey);
    });
  }

  async function runBaseline(blockIndex) {
    if (bootstrap.baseline_duration_sec <= 0) return;
    showScreen("rest-screen");
    logEvent("block_start", { block_index: blockIndex, condition_label: "baseline" });

    const durationSec = bootstrap.baseline_duration_sec;
    const fill = el("rest-progress");
    fill.style.transition = "none";
    fill.style.transform = "scaleX(1)";
    void fill.offsetWidth; // force reflow so the transition below actually animates
    fill.style.transition = `transform ${durationSec}s linear`;
    fill.style.transform = "scaleX(0)";

    const skipped = await waitForBlock(durationSec * 1000);
    logEvent("block_end", { block_index: blockIndex, condition_label: "baseline", skipped });
  }

  function buildGrid() {
    const grid = el("grid");
    grid.innerHTML = "";
    grid.style.gridTemplateColumns = `repeat(${gridSize}, 1fr)`;
    grid.style.gridTemplateRows = `repeat(${gridSize}, 1fr)`;
    layout.forEach((number) => {
      const cell = document.createElement("button");
      cell.className = "schulte-cell";
      cell.textContent = String(number);
      cell.dataset.number = String(number);
      cell.addEventListener("click", () => handleCellClick(cell, number));
      grid.appendChild(cell);
    });
  }

  function updateNextHint() {
    if (bootstrap.show_target_hint === false) {
      el("next-hint").classList.add("hidden");
      return;
    }
    el("next-number").textContent = String(nextTarget);
  }

  function handleCellClick(cellEl, number) {
    if (cellEl.classList.contains("found")) {
      logEvent("cell_click", { number_clicked: number, expected_number: nextTarget, correct: null, repeat: true });
      return;
    }

    const correct = number === nextTarget;
    logEvent("cell_click", {
      number_clicked: number, expected_number: nextTarget, correct,
      rt_ms: performance.now() - startTime,
    });

    if (correct) {
      cellEl.classList.add("found");
      flash(cellEl, "correct-flash");
      nextTarget += 1;
      updateNextHint();
      if (nextTarget > n && resolveTrial) resolveTrial();
    } else {
      misclicks += 1;
      flash(cellEl, "wrong-flash");
    }
  }

  function flash(cellEl, className) {
    cellEl.classList.add(className);
    setTimeout(() => cellEl.classList.remove(className), 250);
  }

  // Demo mode only: clicks the next correct cell on a human-like delay so a
  // participant just watching can see the mechanic without clicking.
  function autoSolveLoop() {
    function clickNext() {
      if (quitting || nextTarget > n) return;
      const cell = document.querySelector(`.schulte-cell[data-number="${nextTarget}"]`);
      if (cell) cell.click();
      if (nextTarget <= n) setTimeout(clickNext, 450 + Math.random() * 350);
    }
    setTimeout(clickNext, 600);
  }

  async function runTrial(blockIndex, autoSolve) {
    showScreen("game-screen");
    buildGrid();
    nextTarget = 1;
    misclicks = 0;
    trialSkipped = false;
    updateNextHint();
    logEvent("block_start", { block_index: blockIndex, condition_label: "schulte_trial", grid_size: gridSize });
    startTime = performance.now();

    function onKeyLocal(e) {
      if (e.key === "Escape") { quitting = true; trialSkipped = true; if (resolveTrial) resolveTrial(); }
      else if (e.key.toLowerCase() === "b") { trialSkipped = true; if (resolveTrial) resolveTrial(); }
    }
    document.addEventListener("keydown", onKeyLocal);

    if (autoSolve) autoSolveLoop();

    await new Promise((resolve) => { resolveTrial = resolve; });
    resolveTrial = null;
    document.removeEventListener("keydown", onKeyLocal);

    const elapsedSec = (performance.now() - startTime) / 1000;
    lastCompletionTimeSec = trialSkipped ? null : elapsedSec;
    lastMisclicks = misclicks;
    logEvent("block_end", {
      block_index: blockIndex, condition_label: "schulte_trial", grid_size: gridSize,
      skipped: trialSkipped, completion_time_sec: lastCompletionTimeSec, misclicks,
    });
  }

  function showEndScreen() {
    showScreen("end-screen");
    el("end-stats").textContent = lastCompletionTimeSec != null
      ? `Completed in ${lastCompletionTimeSec.toFixed(1)}s (${lastMisclicks} incorrect click${lastMisclicks === 1 ? "" : "s"})`
      : "Trial skipped";
    document.addEventListener("keydown", function onSpace(e) {
      if (e.key === " " || e.key === "Enter") {
        document.removeEventListener("keydown", onSpace);
        window.pywebview.api.close_window();
      }
    });
  }

  window.addEventListener("pywebviewready", main);
})();
