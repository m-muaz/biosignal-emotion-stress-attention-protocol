// Raindrop math game front end. All arithmetic/randomness comes from Python
// (app/tasks/stress_mat.generate_question, pre-generated into the bootstrap
// payload by app/tasks/stress_raindrop.py) -- this file only renders,
// animates, collects input, and logs events back through the pywebview
// bridge (see app/webui/bridge.py).
//
// Commit-based input (per PI request 2026-08-10): typing builds up `buffer`
// (handleDigit) but never resolves anything on its own -- Enter/the
// on-screen Enter key (submitAnswer) is the only thing that checks it
// against the live drops and clears it, matching how every reference
// "Raindrops" clone works (type freely, Backspace to fix a typo, Enter to
// submit -- see e.g. github.com/nicksebasco/Raindrops's rcInter.js) rather
// than grading every keystroke live. Deliberately keeps Lumosity's
// commercial version's "wrong submission gets a visual cue" but drops its
// input-freeze-on-repeated-wrong-answers penalty -- the goal is to keep the
// participant attempting questions, not lock them out.
(() => {
  let bootstrap = null;
  let score = 0;
  let buffer = "";
  const liveDrops = new Map(); // id -> { el, answer, expr, spawnTime, blockIndex, tierId, trialIndex, rafId }
  let dropIdCounter = 0;
  let quitting = false;
  let blockStats = { correct: 0, total: 0 };
  // Set at the top of runArithmeticBlock, read by submitAnswer() to label a
  // wrong-submission log row with the block it happened in.
  let currentBlockIndex = null;
  let currentTierId = null;

  const el = (id) => document.getElementById(id);

  // See app/webui/shared/clock_sync.js -- shared with every other webui
  // task and the session shell, so the clock-correspondence fix (browser
  // performance.now() tagging on every event, plus sync_checkpoint round
  // trips at session start/end -- see app/eventlog/event_logger.py's
  // timestamp_monotonic and app/webui/bridge.py's sync_checkpoint()) landed
  // identically everywhere instead of highway/game.js being a one-off.
  const logEvent = ClockSync.wrapLogEvent((eventType, fields) => window.pywebview.api.log_event(eventType, fields));

  async function main() {
    bootstrap = await window.pywebview.api.get_bootstrap();

    if (bootstrap.demo) {
      // Non-interactive preview for familiarization: no keypad/typing, no
      // score/high-score interaction at all -- drops fall and auto-resolve
      // themselves so the participant just watches the mechanic. Runs for
      // bootstrap.arithmetic_duration_sec (set to demo_duration_sec by
      // stress_raindrop.py's run_demo()) -- Escape still quits early via
      // waitForBlock's own handling, same as every other block.
      el("high-score").classList.add("hidden");
      logEvent("task_start", {});
      const tierId = bootstrap.trial_order[0];
      await runArithmeticBlock(0, tierId, bootstrap.tiers[String(tierId)], /* autoSolve */ true);
      logEvent("task_end", {});
      if (quitting) window.pywebview.api.request_quit();
      else window.pywebview.api.close_window();
      return;
    }

    renderHighScore(await window.pywebview.api.get_high_score());

    document.addEventListener("keydown", onKeyDown);
    document.querySelectorAll(".key").forEach((btn) => {
      btn.addEventListener("click", () => {
        if (btn.dataset.key === "enter") submitAnswer();
        else handleDigit(btn.dataset.key);
      });
    });

    // Several round trips at session start AND end (not just one) -- see
    // ClockSync.runCheckpoints -- so offline analysis can fit drift, not
    // just a single offset. Skipped in demo mode above, same reasoning as
    // highway/game.js: a practice/familiarization run isn't part of the
    // scored dataset.
    await ClockSync.runCheckpoints(logEvent, "session_start", bootstrap.sync_checkpoint_count, bootstrap.sync_checkpoint_gap_sec);

    logEvent("task_start", {});

    // baseline_mode "per_trial" (default): a fresh baseline + a
    // participant-paced "ready for next round" pause before EVERY tier --
    // each stress ramp gets its own physiological reference point and the
    // participant gets a breather between rounds.
    // baseline_mode "once": the baseline fixation is shown ONLY before the
    // very first tier, then every tier runs back-to-back with no rest in
    // between -- for when the point is a single unbroken stress ramp
    // (built to induce panic, so repeatedly letting the participant
    // recover mid-ramp would work against that), per PI request 2026-08-04.
    let blockIndex = 0;
    let trialNum = 0;
    for (const tierId of bootstrap.trial_order) {
      if (quitting) break;
      const showBaseline = bootstrap.baseline_mode !== "once" || trialNum === 0;
      if (trialNum > 0 && showBaseline) {
        await waitForContinue();
        if (quitting) break;
      }
      const tierCfg = bootstrap.tiers[String(tierId)];
      if (showBaseline && !bootstrap.practice) {
        await runBaseline(blockIndex++, tierCfg);
        if (quitting) break;
      }
      await runArithmeticBlock(blockIndex++, tierId, tierCfg, /* autoSolve */ false);
      trialNum += 1;
    }

    logEvent("task_end", {});
    await ClockSync.runCheckpoints(logEvent, "session_end", bootstrap.sync_checkpoint_count, bootstrap.sync_checkpoint_gap_sec);
    const result = quitting ? { high_score: null } : await window.pywebview.api.submit_score(score);
    if (quitting) {
      window.pywebview.api.request_quit();
    } else {
      showEndScreen(result.high_score);
    }
  }


  function renderHighScore(hs) {
    // Score only, no participant_id -- per PI request 2026-08-10, the
    // leaderboard-style display shouldn't name whoever set it.
    el("high-score-value").textContent = hs ? `${hs.score}` : "--";
  }

  function showScreen(name) {
    document.querySelectorAll(".screen").forEach((s) => s.classList.add("hidden"));
    el(name).classList.remove("hidden");
  }

  // Shared by baseline and arithmetic blocks: resolves when `durationMs`
  // elapses, OR the participant presses B (skip block, voluntary
  // participation -- mirrors SKIP_BLOCK_KEY in app/ui/common_widgets.py) or
  // Escape (full quit, handled by the caller via the `quitting` flag).
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

  // Gate between trials so the next round starts on the participant's own
  // pace instead of auto-starting -- logged as its own block so a long gap
  // here reads as "waiting," not as an anomalously slow baseline/arithmetic
  // block in post-hoc analysis.
  function waitForContinue() {
    showScreen("pause-screen");
    el("pause-score").textContent = `Score so far: ${score}`;
    logEvent("block_start", { condition_label: "inter_trial_pause" });
    return new Promise((resolve) => {
      function onKey(e) {
        if (e.key === "Escape") { quitting = true; finish(); }
        else if (e.key === " " || e.key === "Enter") finish();
      }
      function finish() {
        document.removeEventListener("keydown", onKey);
        logEvent("block_end", { condition_label: "inter_trial_pause", skipped: quitting });
        resolve();
      }
      document.addEventListener("keydown", onKey);
    });
  }

  async function runBaseline(blockIndex, tierCfg) {
    // Fixation cross, with a deliberately subtle countdown bar underneath
    // (see .fixation-progress-track/-fill) -- there's still essentially
    // nothing to look at but the cross, but it gives some peripheral sense
    // of when the round starts rather than an unmarked, indefinite wait.
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

  async function runArithmeticBlock(blockIndex, tierId, tierCfg, autoSolve) {
    showScreen("game-screen");
    el("drop-field").innerHTML = "";
    liveDrops.forEach((drop) => cancelAnimationFrame(drop.rafId));
    liveDrops.clear();
    buffer = "";
    renderBuffer();
    currentBlockIndex = blockIndex;
    currentTierId = tierId;

    logEvent("block_start", { block_index: blockIndex, condition_label: `tier_${tierId}`, tier: tierId });
    blockStats = { correct: 0, total: 0 };

    const spawnMs = tierCfg.spawn_interval_sec * 1000;
    const fallMs = tierCfg.fall_duration_sec * 1000;
    let questionIndex = 0;
    let trialIndex = 0;
    let spawnTimer = null;

    function spawnNext() {
      if (questionIndex >= tierCfg.questions.length) return;
      const q = tierCfg.questions[questionIndex++];
      spawnDrop(q, blockIndex, tierId, trialIndex++, fallMs, autoSolve);
      spawnTimer = setTimeout(spawnNext, spawnMs);
    }
    spawnNext();

    const skipped = await waitForBlock(bootstrap.arithmetic_duration_sec * 1000);
    clearTimeout(spawnTimer);

    // Skipped/quit: clear remaining drops immediately. Otherwise, let
    // whatever's still falling finish naturally so a drop spawned right
    // before the deadline isn't just erased mid-air.
    if (skipped || quitting) {
      liveDrops.forEach((drop, id) => missDrop(id, /* silent */ true));
    } else {
      await waitUntil(() => liveDrops.size === 0 || quitting);
    }

    const accuracy = blockStats.total > 0 ? blockStats.correct / blockStats.total : null;
    logEvent("block_end", {
      block_index: blockIndex, condition_label: `tier_${tierId}`, tier: tierId,
      skipped: skipped || quitting, accuracy,
    });
  }

  function waitUntil(predicate) {
    return new Promise((resolve) => {
      (function poll() {
        if (predicate()) resolve();
        else setTimeout(poll, 150);
      })();
    });
  }

  function spawnDrop(question, blockIndex, tierId, trialIndex, fallMs, autoSolve) {
    const id = ++dropIdCounter;
    const wrap = document.createElement("div");
    wrap.className = "drop";
    wrap.style.left = (10 + Math.random() * 80) + "%";
    const label = document.createElement("div");
    label.className = "drop-label";
    label.textContent = question.expr;
    wrap.appendChild(label);
    el("drop-field").appendChild(wrap);

    const spawnTime = performance.now();
    const drop = {
      el: wrap, answer: question.answer, expr: question.expr,
      spawnTime, blockIndex, tierId, trialIndex, rafId: null,
    };
    liveDrops.set(id, drop);

    logEvent("trial_start", {
      block_index: blockIndex, trial_index: trialIndex, condition_label: `tier_${tierId}`,
      tier: tierId, expression: question.expr, correct_answer: question.answer,
    });

    if (autoSolve) {
      // Demo mode: highlight the drop partway down, then "solve" it shortly
      // after, so a participant just watching can see the same
      // highlight -> pop feedback the real game gives, without typing.
      setTimeout(() => { if (liveDrops.has(id)) drop.el.classList.add("highlighted"); }, fallMs * 0.35);
      setTimeout(() => {
        if (liveDrops.has(id)) popDrop(id, (performance.now() - spawnTime) / 1000);
      }, fallMs * 0.55);
    }

    // Per PI request 2026-08-10 ("numbers become blurry" at fast tiers):
    // vertical motion is driven by `transform` (translateY, in px), NOT by
    // animating `top` -- `top` is a layout property, so mutating it every
    // rAF frame forces a full layout recalc + repaint (re-rasterizing the
    // rotated expression text at a new sub-pixel offset) on every single
    // frame. At fast tiers with several drops overlapping, that's enough
    // per-frame work to drop frames -- and the resulting bigger, uneven
    // position jumps read as blur even before any actual GPU/font blur
    // does. `transform` is compositor-only: the browser can rasterize the
    // rotated text once and just re-translate that cached layer, which is
    // both cheaper and doesn't touch the text's rendering after the first
    // frame. `top: 0` (fixed, never touched again) is kept purely so this
    // element still participates in normal layout at all -- all the actual
    // movement is in `transform`. See .drop's `will-change: transform` in
    // style.css for the compositing-layer hint that makes this effective.
    wrap.style.top = "0";
    const fieldHeight = el("drop-field").clientHeight;

    function setFallPosition(t) {
      // Only ever sets the --fall-offset custom property, NOT the whole
      // transform -- .drop's own CSS rule (and the pop/sink keyframes)
      // read it via var(--fall-offset). Per PI request 2026-08-10: writing
      // the offset straight into a JS-authored transform string used to
      // make a correctly-answered drop visibly jump back up to 0 offset
      // the instant popDrop()'s "correct" class took over -- the pop
      // keyframe's `to` had its own hardcoded translate that didn't know
      // about the drop's current fall position. Going through the shared
      // custom property instead means the keyframe picks up wherever the
      // drop actually was, no jump.
      //
      // Same -10% -> 110% visual range as before, just expressed in px
      // (translate's own %-unit means "% of THIS element's size", not the
      // container's, so the fall range has to be pre-converted to px here
      // rather than passed through as a percentage). Rounded to a whole
      // pixel -- sub-pixel positions are exactly what forces the rotated
      // text to be re-anti-aliased at a new offset each frame.
      const offsetPx = Math.round(((-10 + t * 120) / 100) * fieldHeight);
      wrap.style.setProperty("--fall-offset", `${offsetPx}px`);
    }
    setFallPosition(0);

    const start = performance.now();
    function step(now) {
      if (!liveDrops.has(id)) return; // already resolved (correct/missed)
      const t = Math.min(1, (now - start) / fallMs);
      setFallPosition(t);
      if (t >= 1) {
        missDrop(id);
      } else {
        drop.rafId = requestAnimationFrame(step);
      }
    }
    drop.rafId = requestAnimationFrame(step);
  }

  function missDrop(id, silent) {
    const drop = liveDrops.get(id);
    if (!drop) return;
    liveDrops.delete(id);
    cancelAnimationFrame(drop.rafId);
    if (!silent) {
      drop.el.classList.add("missed");
      // Per PI request 2026-08-10: reaching the bottom uncracked is the
      // ONLY way to lose a point. A wrong Enter submission never costs
      // anything (see submitAnswer()) -- the participant can keep
      // attempting a drop for as long as it's still falling, right up
      // until it lands here. `silent` is true for runArithmeticBlock's
      // end-of-block cleanup of still-falling drops on a skip/quit, which
      // is NOT a genuine "reached the bottom" miss, so that path is
      // excluded from the penalty. Clamped at 0 rather than going
      // negative -- flag if unclamped (visibly negative) is actually
      // wanted for the stress manipulation.
      score = Math.max(0, score - 1);
      el("score-value").textContent = String(score);
      flashScoreLoss();
    }
    blockStats.total += 1;
    logEvent("response", {
      block_index: drop.blockIndex, trial_index: drop.trialIndex, condition_label: `tier_${drop.tierId}`,
      tier: drop.tierId, expression: drop.expr, correct_answer: drop.answer,
      participant_answer: null, rt: null, correct: false, missed: true,
    });
    setTimeout(() => drop.el.remove(), silent ? 0 : 300);
  }

  // Same "brief flash, no lasting lockout" treatment as flashWrong(), just
  // on the score HUD instead of the answer bar -- the only cue that a
  // point was actually lost.
  function flashScoreLoss() {
    const scoreEl = el("score");
    scoreEl.classList.remove("score-flash");
    void scoreEl.offsetWidth; // force reflow so re-adding the class restarts the animation
    scoreEl.classList.add("score-flash");
    setTimeout(() => scoreEl.classList.remove("score-flash"), 300);
  }

  function popDrop(id, rt) {
    const drop = liveDrops.get(id);
    if (!drop) return;
    liveDrops.delete(id);
    cancelAnimationFrame(drop.rafId);
    drop.el.classList.remove("highlighted");
    drop.el.classList.add("correct");
    score += 1;
    blockStats.correct += 1;
    blockStats.total += 1;
    el("score-value").textContent = String(score);
    logEvent("response", {
      block_index: drop.blockIndex, trial_index: drop.trialIndex, condition_label: `tier_${drop.tierId}`,
      tier: drop.tierId, expression: drop.expr, correct_answer: drop.answer,
      participant_answer: drop.answer, rt, correct: true, missed: false,
    });
    setTimeout(() => drop.el.remove(), 250);
  }

  function renderBuffer() {
    el("answer-display").textContent = buffer || " ";
  }

  // Purely advisory: highlights any live drop whose answer STARTS WITH the
  // current buffer (a full match highlights too, since a string "starts
  // with" itself -- that's the participant's "you've got it, hit Enter"
  // cue) and flags the answer bar red when nothing on screen matches at
  // all. Never touches `buffer` itself -- see the module docstring's
  // "commit-based input" note for why typing no longer self-resolves.
  function updateHighlights() {
    let anyMatch = false;
    for (const drop of liveDrops.values()) {
      const matches = buffer.length > 0 && String(drop.answer).startsWith(buffer);
      drop.el.classList.toggle("highlighted", matches);
      if (matches) anyMatch = true;
    }
    el("answer-bar").classList.toggle("mismatch", buffer.length > 0 && !anyMatch);
    return anyMatch;
  }

  // Free-text-style buffer editing ONLY -- no auto-submit, no auto-clear.
  // Per PI request 2026-08-10: typing no longer resolves a drop the instant
  // the buffer happens to match (that let the game half-solve itself by
  // accident, and was also the root of an earlier bug where a WRONG entry
  // just kept appending forever instead of clearing -- see submitAnswer(),
  // which is now the only thing that ever resolves or clears the buffer,
  // same as every reference "Raindrops" clone: type freely, Backspace to
  // fix a typo, Enter to commit). This mirrors a plain text input's
  // behavior on purpose -- the participant has to decide "this is my
  // answer" and commit to it, rather than the game grading every keystroke
  // live.
  function handleDigit(key) {
    if (el("game-screen").classList.contains("hidden")) return;
    if (key === "backspace") buffer = buffer.slice(0, -1);
    else if (key === "-") { if (buffer === "") buffer = "-"; }
    else buffer += key;

    renderBuffer();
    updateHighlights();
  }

  // The one and only commit point: Enter (or the on-screen Enter key).
  // Checks the buffer against every live drop; a match pops it exactly like
  // before. No match -- including an empty submit, silently ignored -- logs
  // a wrong_submission row and clears the buffer, same as a real submit
  // action always clearing regardless of correctness (see the GitHub
  // "Raindrops" clones this was modeled on). Deliberately NOT paired with
  // Lumosity's input-freeze penalty on repeated wrong answers, per PI
  // request 2026-08-10 -- the goal here is to keep the participant
  // attempting questions, not lock them out.
  function submitAnswer() {
    if (el("game-screen").classList.contains("hidden")) return;
    if (buffer === "") return;

    // Pop EVERY live drop with this exact answer, not just the first one
    // found -- each drop's answer is an independent RNG draw (see
    // stress_mat.generate_question), so two concurrently-falling drops can
    // legitimately land on the same number (e.g. "12 + 1" and "10 + 3"
    // both = 13) with no way for the participant to aim at one over the
    // other. Matches updateHighlights(), which already highlights every
    // matching drop, not just one, and mirrors Lumosity's own "drops that
    // share a solution pop together" rule. Collected into an array first
    // (rather than deleting from liveDrops mid-iteration) purely so this
    // reads unambiguously; Map iteration tolerates in-loop deletion fine.
    const matchIds = [];
    for (const [id, drop] of liveDrops) {
      if (String(drop.answer) === buffer) matchIds.push(id);
    }

    if (matchIds.length > 0) {
      for (const id of matchIds) {
        const drop = liveDrops.get(id);
        popDrop(id, (performance.now() - drop.spawnTime) / 1000);
      }
      buffer = "";
      renderBuffer();
      updateHighlights();
      return;
    }

    // Wrong submission -- not tied to any one drop (with several falling
    // at once there's no way to know which one the participant meant), so
    // this is its own event type rather than a "response" row, which
    // everywhere else always pairs 1:1 with a specific trial_start.
    logEvent("wrong_submission", {
      block_index: currentBlockIndex, condition_label: `tier_${currentTierId}`,
      tier: currentTierId, participant_answer: buffer,
    });
    flashWrong();
    buffer = "";
    renderBuffer();
    updateHighlights();
  }

  // Brief visual "that was wrong" cue on the answer bar itself -- since
  // there's no input freeze, this is the only feedback a wrong submission
  // gets, so it shouldn't be silent.
  function flashWrong() {
    const bar = el("answer-bar");
    bar.classList.remove("wrong-flash");
    void bar.offsetWidth; // force reflow so re-adding the class restarts the animation
    bar.classList.add("wrong-flash");
    setTimeout(() => bar.classList.remove("wrong-flash"), 300);
  }

  function onKeyDown(e) {
    if (el("game-screen").classList.contains("hidden")) return;
    if (e.key >= "0" && e.key <= "9") handleDigit(e.key);
    else if (e.key === "-") handleDigit("-");
    else if (e.key === "Backspace") handleDigit("backspace");
    else if (e.key === "Enter") submitAnswer();
  }

  function showEndScreen(highScore) {
    showScreen("end-screen");
    el("end-score").textContent = `Your score: ${score}`;
    el("end-high-score").textContent = highScore
      ? `All-time high score: ${highScore.score}`
      : "";
    document.addEventListener("keydown", function onSpace(e) {
      if (e.key === " " || e.key === "Enter") {
        document.removeEventListener("keydown", onSpace);
        window.pywebview.api.close_window();
      }
    });
  }

  window.addEventListener("pywebviewready", main);
})();
