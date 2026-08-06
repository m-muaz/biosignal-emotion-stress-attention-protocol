// Stroop color-word test front end. The stimulus list (word/ink-color
// pairs, congruent/incongruent) comes entirely from Python
// (app/tasks/attention_focus.py's build_sequence(), using ctx.rng -- so
// provenance stays with the one rng_seed recorded in session_manifest.json)
// -- this file only renders, collects clicks, and logs events back through
// the pywebview bridge (see app/webui/bridge.py).
//
// Self-paced: no response deadline. Reaction time itself is the dependent
// measure, so nothing here should be racing a clock -- same "no artificial
// time pressure" design choice as the Schulte table (per PI request
// 2026-08-04).
(() => {
  let bootstrap = null;
  let colors = [];
  let stimuli = [];
  let quitting = false;
  let trialSkipped = false;
  let stats = { correct: 0, total: 0, rts: [] };
  let resolveStimulus = null;
  let currentStimulus = null;
  let currentTrialIndex = 0;
  let currentBlockIndex = 0;
  let stimulusOnset = 0;
  let autoSolve = false;
  let lastStats = null;

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

  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
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
    colors = bootstrap.colors;
    stimuli = bootstrap.stimuli;
    buildColorButtons();

    if (bootstrap.demo) {
      // Non-interactive preview for familiarization: a handful of
      // auto-answered stimuli so the participant watches the
      // click-the-ink-color mechanic -- same role as every other task's
      // demo mode.
      el("hud").classList.add("hidden");
      logEvent("task_start", {});
      await showInstructions(
        "PREVIEW -- STROOP TEST\n\nWatch: the test answers itself automatically so you can see how it works.",
        "Press SPACE to watch.",
      );
      if (!quitting) await runTrial(0, /* isAutoSolve */ true);
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
      `ATTENTION TRIAL ${bootstrap.trial_index + 1} of ${bootstrap.total_trials} -- STROOP TEST\n\n` +
      "A color name will appear, printed in an ink color. Click the button matching the INK color " +
      "it's printed in -- not the word itself. You can also press the number key shown on each " +
      "button instead of clicking.\n\n" +
      "There's no time limit -- take the time you need.\n\n" +
      "B = skip this trial   |   Esc = quit",
      "Press SPACE to begin.",
    );

    if (!quitting) {
      await runBaseline(0);
      if (!quitting) await runTrial(1, /* isAutoSolve */ false);
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

  // Plain luminance heuristic (ITU-R BT.601 weights) for picking readable
  // label text over an arbitrary swatch color -- avoids needing a
  // hand-picked text color per entry in session_config.yaml's color list.
  function contrastTextColor(hex) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    const luminance = 0.299 * r + 0.587 * g + 0.114 * b;
    return luminance > 150 ? "#0e2331" : "#ffffff";
  }

  function buildColorButtons() {
    const container = el("color-buttons");
    container.innerHTML = "";
    // Same fixed order/position every trial (bootstrap.colors' own order)
    // -- see style.css's comment on why that matters for RT noise. Each
    // button is labeled with its own color's name -- see style.css's
    // comment on why that isn't a second Stroop conflict -- plus its
    // number-key shortcut (1..colors.length, matching on-screen position,
    // same convention as the questionnaire screens in shell.js) so a
    // response doesn't have to pay a mouse-reach cost on top of the
    // decision-time RT being measured (per feedback 2026-08-04). Mouse
    // clicking still works too -- this is an addition, not a replacement.
    colors.forEach((color, idx) => {
      const btn = document.createElement("button");
      btn.className = "color-btn";
      btn.style.background = color.hex;
      btn.style.color = contrastTextColor(color.hex);
      btn.dataset.name = color.name;
      btn.disabled = true;

      const key = document.createElement("span");
      key.className = "color-btn-key";
      key.textContent = String(idx + 1);
      btn.appendChild(key);

      const label = document.createElement("span");
      label.className = "color-btn-label";
      label.textContent = color.name.toUpperCase();
      btn.appendChild(label);

      btn.addEventListener("click", () => handleResponse(color.name));
      container.appendChild(btn);
    });
  }

  function setButtonsEnabled(enabled) {
    document.querySelectorAll(".color-btn").forEach((b) => { b.disabled = !enabled; });
  }

  async function runTrial(blockIndex, isAutoSolve) {
    showScreen("game-screen");
    autoSolve = isAutoSolve;
    currentBlockIndex = blockIndex;
    trialSkipped = false;
    stats = { correct: 0, total: 0, rts: [] };
    logEvent("block_start", { block_index: blockIndex, condition_label: "stroop_trial", n_stimuli: stimuli.length });

    document.addEventListener("keydown", onTrialKey);

    for (currentTrialIndex = 0; currentTrialIndex < stimuli.length; currentTrialIndex++) {
      if (quitting || trialSkipped) break;
      await runStimulus(blockIndex, currentTrialIndex);
    }

    document.removeEventListener("keydown", onTrialKey);
    setButtonsEnabled(false);

    const meanRtMs = stats.rts.length ? stats.rts.reduce((a, b) => a + b, 0) / stats.rts.length : null;
    const accuracy = stats.total > 0 ? stats.correct / stats.total : null;
    lastStats = { accuracy, meanRtMs, total: stats.total };
    logEvent("block_end", {
      block_index: blockIndex, condition_label: "stroop_trial",
      skipped: trialSkipped || quitting, accuracy, mean_rt_ms: meanRtMs, n_trials: stats.total,
    });
  }

  function onTrialKey(e) {
    let shouldResolve = false;
    if (e.key === "Escape") { quitting = true; trialSkipped = true; shouldResolve = true; }
    else if (e.key.toLowerCase() === "b") { trialSkipped = true; shouldResolve = true; }
    if (shouldResolve && resolveStimulus) {
      const resolve = resolveStimulus;
      resolveStimulus = null;
      resolve();
      return;
    }

    // Number keys 1..colors.length answer with the color in that on-screen
    // position -- only while a stimulus is actually up (currentStimulus set,
    // buttons enabled); a stray digit during the ISI blank or the auto-solve
    // demo is just ignored (handleResponse() no-ops without a stimulus, but
    // checking here too avoids resolving a promise that's already null).
    if (!currentStimulus) return;
    const idx = Number(e.key) - 1;
    if (!Number.isNaN(idx) && idx >= 0 && idx < colors.length) {
      handleResponse(colors[idx].name);
    }
  }

  async function runStimulus(blockIndex, trialIndex) {
    el("stimulus-word").textContent = "";
    setButtonsEnabled(false);
    currentStimulus = null;

    await sleep(bootstrap.isi_sec * 1000);
    if (quitting || trialSkipped) return;

    const stim = stimuli[trialIndex];
    currentStimulus = stim;
    const colorDef = colors.find((c) => c.name === stim.ink_color);
    const wordEl = el("stimulus-word");
    wordEl.textContent = stim.word.toUpperCase();
    wordEl.style.color = colorDef.hex;
    setButtonsEnabled(true);
    stimulusOnset = performance.now();

    logEvent("trial_start", {
      block_index: blockIndex, trial_index: trialIndex, condition_label: "stroop_trial",
      word: stim.word, ink_color: stim.ink_color, congruent: stim.congruent,
    });

    await new Promise((resolve) => {
      resolveStimulus = resolve;
      if (autoSolve) {
        setTimeout(() => {
          if (!quitting && !trialSkipped && currentStimulus) handleResponse(stim.ink_color);
        }, 500 + Math.random() * 400);
      }
    });
  }

  function handleResponse(clickedColorName) {
    if (!currentStimulus) return;
    const rtMs = performance.now() - stimulusOnset;
    const correct = clickedColorName === currentStimulus.ink_color;
    stats.total += 1;
    if (correct) { stats.correct += 1; stats.rts.push(rtMs); }

    logEvent("response", {
      block_index: currentBlockIndex, trial_index: currentTrialIndex, condition_label: "stroop_trial",
      word: currentStimulus.word, ink_color: currentStimulus.ink_color, congruent: currentStimulus.congruent,
      clicked_color: clickedColorName, correct, rt_ms: rtMs,
    });

    flashButton(clickedColorName, correct);
    setButtonsEnabled(false);
    currentStimulus = null;

    const resolve = resolveStimulus;
    resolveStimulus = null;
    setTimeout(() => { if (resolve) resolve(); }, 200);
  }

  function flashButton(name, correct) {
    const btn = document.querySelector(`.color-btn[data-name="${name}"]`);
    if (!btn) return;
    btn.classList.add(correct ? "correct-flash" : "wrong-flash");
    setTimeout(() => btn.classList.remove("correct-flash", "wrong-flash"), 200);
  }

  function showEndScreen() {
    showScreen("end-screen");
    el("end-stats").textContent = lastStats && lastStats.total > 0
      ? `${Math.round(lastStats.accuracy * 100)}% correct, avg RT ${Math.round(lastStats.meanRtMs || 0)}ms (${lastStats.total} trials)`
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
