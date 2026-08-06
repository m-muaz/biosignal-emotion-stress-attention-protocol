// Session shell: drives the whole session end-to-end (consent ->
// questionnaire -> familiarization -> emotion task -> break -> stress task ->
// conclusion). Game/video experiences run in separate subprocesses launched
// via the bridge (see app/tasks/session_shell.py); this file only renders
// text/questionnaire/break screens and calls out to those subprocesses at
// the right points.
(() => {
  let bootstrap = null;
  let sessionEnded = false; // true once conclusion/quit/crash has already logged+closed, so the global Escape handler doesn't double-fire

  const el = (id) => document.getElementById(id);
  const api = () => window.pywebview.api;

  // See app/webui/shared/clock_sync.js -- shared with every *_app.py
  // subprocess this shell launches. The shell's own window is its OWN
  // browser engine instance with its own independent performance.now()
  // epoch (distinct from every subprocess's), so it needs its own
  // session_start/session_end checkpoints too -- see main() below -- not
  // just whatever each subprocess does for itself.
  const logEvent = ClockSync.wrapLogEvent((eventType, fields, task) => api().log_event(eventType, fields, task || null));

  function showScreen(name) {
    document.querySelectorAll(".screen").forEach((s) => s.classList.add("hidden"));
    el(name).classList.remove("hidden");
  }

  // Global Escape handling: any screen not currently inside a subprocess
  // call counts as "the participant quit right here" -- mirrors
  // app/ui/common_widgets.py's request_quit()/check_quit() semantics for
  // the whole shell (subprocess-launched screens have their own Escape
  // handling, scoped to that subprocess's own window).
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && !sessionEnded) {
      sessionEnded = true;
      logEvent("session_aborted", {}, null);
      api().request_quit();
    }
  });

  function endSession(status) {
    if (sessionEnded) return;
    sessionEnded = true;
    if (status === "quit") logEvent("session_aborted", {}, null);
    // "crashed" already logged session_crashed from the Python side (see
    // ShellApi._run_guarded) -- nothing further to log here.
    api().request_quit();
  }

  // Runs a subprocess-launching bridge call; returns true if the session
  // should keep going, false if it just ended (quit/crashed) and the caller
  // should stop advancing through the rest of the flow.
  async function runGuarded(bridgeCall) {
    const result = await bridgeCall();
    if (result.status === "ok") return true;
    endSession(result.status);
    return false;
  }

  function showMessage(text, opts) {
    opts = opts || {};
    showScreen("message-screen");
    el("message-text").innerHTML = text;
    el("message-hint").textContent = opts.hint || "Press SPACE to continue.";
    return new Promise((resolve) => {
      function onKey(e) {
        if (e.key === " " || e.key === "Enter") {
          document.removeEventListener("keydown", onKey);
          resolve();
        }
      }
      document.addEventListener("keydown", onKey);
    });
  }

  function heading(text) {
    return `<span class="heading">${text}</span>`;
  }

  // ---- Questionnaire ----------------------------------------------------
  //
  // Every item screen highlights the current selection but does NOT
  // auto-advance -- the participant confirms with the persistent Next
  // button (or Enter/ArrowRight), and can revisit a prior item with
  // Previous (or ArrowLeft) to change an earlier answer. Each render*Item
  // function returns "next" or "prev"; runQuestionnaire walks an index
  // back and forth through the item list accordingly. `answers` persists
  // each item's current value(s) across visits so a revisited screen
  // reopens pre-filled with what was chosen before.

  function setNav(opts) {
    const prevBtn = el("q-prev-btn");
    const nextBtn = el("q-next-btn");
    // A focused nav button reacts to a bare Space/Enter itself (native
    // button activation) on top of our own keydown handler -- blur it so a
    // stray keypress after a mouse click can't double-fire the next item's
    // confirm as well.
    if (document.activeElement === prevBtn || document.activeElement === nextBtn) {
      document.activeElement.blur();
    }
    prevBtn.classList.toggle("visible", !!opts.showPrev);
    nextBtn.disabled = !opts.nextEnabled;
  }

  // Wires the persistent Previous/Next buttons for the currently-rendered
  // item; returns an unbind function the item must call before it resolves.
  function bindNav(onNext, onPrev) {
    const prevBtn = el("q-prev-btn");
    const nextBtn = el("q-next-btn");
    function nextHandler() { if (!nextBtn.disabled) onNext(); }
    function prevHandler() { if (onPrev) onPrev(); }
    nextBtn.addEventListener("click", nextHandler);
    prevBtn.addEventListener("click", prevHandler);
    return () => {
      nextBtn.removeEventListener("click", nextHandler);
      prevBtn.removeEventListener("click", prevHandler);
    };
  }

  async function runQuestionnaire(items) {
    showScreen("questionnaire-screen");
    const activeItems = items.filter((item) => item.enabled !== false);
    const answers = {}; // item.id -> {value, rt} or {values, rts} for group/multi_rating
    let i = 0;
    while (i < activeItems.length) {
      const item = activeItems[i];
      const isFirst = i === 0;
      let action;
      if (item.type === "choice") action = await renderChoiceItem(item, answers, isFirst);
      else if (item.type === "numeric") action = await renderNumericItem(item, answers, isFirst);
      else if (item.type === "rating") action = await renderRatingItem(item, answers, isFirst);
      else if (item.type === "multi_rating") action = await renderMultiRatingItem(item, answers, isFirst);
      else if (item.type === "group") action = await renderGroupItem(item, answers, isFirst);
      else action = "next";
      i += action === "prev" ? -1 : 1;
    }
  }

  function questionnaireContainer() {
    const c = el("questionnaire-content");
    c.innerHTML = "";
    return c;
  }

  function renderChoiceItem(item, answers, isFirst) {
    const container = questionnaireContainer();
    const prompt = document.createElement("div");
    prompt.className = "q-prompt";
    prompt.textContent = item.prompt;
    container.appendChild(prompt);

    const options = document.createElement("div");
    options.className = "q-options";
    container.appendChild(options);

    const saved = answers[item.id];
    let selectedIndex = saved ? item.options.indexOf(saved.value) : -1;
    let selectTimestamp = null; // set only when the participant actively (re)picks this visit
    const start = performance.now();
    const buttons = [];

    item.options.forEach((opt, idx) => {
      const btn = document.createElement("button");
      btn.className = "q-option-btn";
      btn.textContent = opt;
      if (idx === selectedIndex) btn.classList.add("selected");
      btn.addEventListener("click", () => select(idx));
      options.appendChild(btn);
      buttons.push(btn);
    });

    function select(idx) {
      selectedIndex = idx;
      selectTimestamp = performance.now();
      buttons.forEach((b, i) => b.classList.toggle("selected", i === idx));
      setNav({ showPrev: !isFirst, nextEnabled: true });
    }

    setNav({ showPrev: !isFirst, nextEnabled: selectedIndex >= 0 });

    return new Promise((resolve) => {
      function finish(action) {
        document.removeEventListener("keydown", onKey);
        unbindNav();
        resolve(action);
      }
      function confirmNext() {
        if (selectedIndex < 0) return;
        const value = item.options[selectedIndex];
        const rt = selectTimestamp !== null ? (selectTimestamp - start) / 1000 : saved.rt;
        answers[item.id] = { value, rt };
        logEvent("questionnaire_response", { item_id: item.id, value, rt }, "preparation");
        finish("next");
      }
      function goPrev() {
        if (!isFirst) finish("prev");
      }
      function onKey(e) {
        const idx = Number(e.key) - 1;
        if (idx >= 0 && idx < item.options.length) { select(idx); return; }
        if (e.key === "Enter" || e.key === "ArrowRight") confirmNext();
        else if (e.key === "ArrowLeft") goPrev();
      }
      const unbindNav = bindNav(confirmNext, isFirst ? null : goPrev);
      document.addEventListener("keydown", onKey);
    });
  }

  function renderNumericItem(item, answers, isFirst) {
    const container = questionnaireContainer();
    const prompt = document.createElement("div");
    prompt.className = "q-prompt";
    prompt.textContent = item.prompt;
    container.appendChild(prompt);

    const display = document.createElement("div");
    display.className = "q-numeric-display";
    container.appendChild(display);

    const keypad = document.createElement("div");
    keypad.className = "q-keypad";
    container.appendChild(keypad);

    const hint = document.createElement("div");
    hint.className = "q-continue-hint";
    hint.textContent = "Type a number, then press ENTER or click Next.";
    container.appendChild(hint);

    const saved = answers[item.id];
    let buffer = saved ? String(saved.value) : "";
    display.textContent = buffer || " ";
    let editedTimestamp = null;
    const start = performance.now();
    const keys = ["7", "8", "9", "4", "5", "6", "1", "2", "3", ".", "0", "⌫"];
    keys.forEach((label) => {
      const btn = document.createElement("button");
      btn.textContent = label;
      btn.addEventListener("click", () => handle(label === "⌫" ? "Backspace" : label));
      keypad.appendChild(btn);
    });

    function handle(key) {
      if (key === "Backspace") buffer = buffer.slice(0, -1);
      else if (key === ".") { if (!buffer.includes(".")) buffer += "."; }
      else buffer += key;
      editedTimestamp = performance.now();
      display.textContent = buffer || " ";
      setNav({ showPrev: !isFirst, nextEnabled: buffer !== "" });
    }

    setNav({ showPrev: !isFirst, nextEnabled: buffer !== "" });

    return new Promise((resolve) => {
      function finish(action) {
        document.removeEventListener("keydown", onKey);
        unbindNav();
        resolve(action);
      }
      function confirmNext() {
        if (buffer === "") return;
        const value = Number(buffer);
        const rt = editedTimestamp !== null ? (editedTimestamp - start) / 1000 : saved.rt;
        answers[item.id] = { value, rt };
        logEvent("questionnaire_response", { item_id: item.id, value, rt }, "preparation");
        finish("next");
      }
      function goPrev() {
        if (!isFirst) finish("prev");
      }
      function onKey(e) {
        if (e.key >= "0" && e.key <= "9") handle(e.key);
        else if (e.key === ".") handle(".");
        else if (e.key === "Backspace") handle("Backspace");
        else if (e.key === "Enter" || e.key === "ArrowRight") confirmNext();
        else if (e.key === "ArrowLeft") goPrev();
      }
      const unbindNav = bindNav(confirmNext, isFirst ? null : goPrev);
      document.addEventListener("keydown", onKey);
    });
  }

  function buildRatingRow(container, key, prompt, leftLabel, rightLabel, scaleMin, scaleMax, onSelect) {
    const row = document.createElement("div");
    row.className = "q-rating-row";
    const promptEl = document.createElement("div");
    promptEl.className = "q-prompt";
    promptEl.style.fontSize = "18px";
    promptEl.textContent = prompt;
    row.appendChild(promptEl);

    const scale = document.createElement("div");
    scale.className = "q-rating-scale";
    const buttons = [];
    for (let v = scaleMin; v <= scaleMax; v++) {
      const btn = document.createElement("button");
      btn.className = "q-rating-btn";
      btn.textContent = String(v);
      btn.addEventListener("click", () => onSelect(v));
      scale.appendChild(btn);
      buttons.push(btn);
    }
    row.appendChild(scale);

    // Endpoint labels ("Not at all" / "Very much so") must span exactly the
    // width of the buttons above them -- scales vary in length (STAI: 4,
    // PANAS: 5, emotion task: 7), so a fixed CSS width would only line up
    // for one of them.
    const rowWidthPx = (scaleMax - scaleMin + 1) * 42 + (scaleMax - scaleMin) * 10;
    scale.style.width = `${rowWidthPx}px`;
    scale.style.margin = "0 auto";

    if (leftLabel || rightLabel) {
      const endpoints = document.createElement("div");
      endpoints.className = "q-rating-endpoints";
      endpoints.style.width = `${rowWidthPx}px`;
      endpoints.innerHTML = `<span>${leftLabel || ""}</span><span>${rightLabel || ""}</span>`;
      row.appendChild(endpoints);
    }
    container.appendChild(row);
    return buttons;
  }

  function renderRatingItem(item, answers, isFirst) {
    // Plain "rating" items always use the emotion_task-wide scale (matches
    // the old rating_scale_0_7(win, ..., cfg["emotion_task"]["rating_scale_min"],
    // ...) call) -- only "multi_rating" items (STAI-6/PANAS) define their
    // own per-item scale_min/scale_max in session_config.yaml.
    const scaleMin = bootstrap.rating_scale_min;
    const scaleMax = bootstrap.rating_scale_max;
    const container = questionnaireContainer();
    const saved = answers[item.id];
    let selectedValue = saved ? saved.value : null;
    let selectTimestamp = null;
    const start = performance.now();

    function select(value) {
      selectedValue = value;
      selectTimestamp = performance.now();
      buttons.forEach((b, i) => b.classList.toggle("selected", i === value - scaleMin));
      setNav({ showPrev: !isFirst, nextEnabled: true });
    }

    const buttons = buildRatingRow(
      container, item.id, item.prompt, item.left_label, item.right_label, scaleMin, scaleMax, select,
    );
    if (selectedValue !== null) {
      buttons.forEach((b, i) => b.classList.toggle("selected", i === selectedValue - scaleMin));
    }
    setNav({ showPrev: !isFirst, nextEnabled: selectedValue !== null });

    return new Promise((resolve) => {
      function finish(action) {
        document.removeEventListener("keydown", onKey);
        unbindNav();
        resolve(action);
      }
      function confirmNext() {
        if (selectedValue === null) return;
        const rt = selectTimestamp !== null ? (selectTimestamp - start) / 1000 : saved.rt;
        answers[item.id] = { value: selectedValue, rt };
        logEvent("questionnaire_response", { item_id: item.id, value: selectedValue, rt }, "preparation");
        finish("next");
      }
      function goPrev() {
        if (!isFirst) finish("prev");
      }
      function onKey(e) {
        const v = Number(e.key);
        if (!Number.isNaN(v) && v >= scaleMin && v <= scaleMax) { select(v); return; }
        if (e.key === "Enter" || e.key === "ArrowRight") confirmNext();
        else if (e.key === "ArrowLeft") goPrev();
      }
      const unbindNav = bindNav(confirmNext, isFirst ? null : goPrev);
      document.addEventListener("keydown", onKey);
    });
  }

  async function renderMultiRatingItem(item, answers, isFirst) {
    if (item.prompt) await showMessage(item.prompt);
    showScreen("questionnaire-screen");
    const container = questionnaireContainer();
    const saved = answers[item.id] || { values: {}, rts: {} };
    const values = { ...saved.values };
    const rts = { ...saved.rts };
    const rowsByKey = {};
    const start = performance.now();

    item.items.forEach((sub) => {
      if (!(sub.id in values)) values[sub.id] = null;
      rowsByKey[sub.id] = buildRatingRow(
        container, sub.id, sub.prompt, item.left_label, item.right_label,
        item.scale_min, item.scale_max, (value) => setValue(sub.id, value),
      );
      if (values[sub.id] !== null) {
        rowsByKey[sub.id].forEach((btn, i) => btn.classList.toggle("selected", item.scale_min + i === values[sub.id]));
      }
    });

    function allAnswered() {
      return item.items.every((sub) => values[sub.id] !== null);
    }

    function setValue(key, value) {
      values[key] = value;
      rts[key] = (performance.now() - start) / 1000; // always the latest pick this visit
      rowsByKey[key].forEach((btn, i) => btn.classList.toggle("selected", item.scale_min + i === value));
      setNav({ showPrev: !isFirst, nextEnabled: allAnswered() });
    }

    const hint = document.createElement("div");
    hint.className = "q-continue-hint";
    hint.textContent = "Answer each row above, then press ENTER or click Next.";
    container.appendChild(hint);

    setNav({ showPrev: !isFirst, nextEnabled: allAnswered() });

    const action = await new Promise((resolve) => {
      function finish(a) {
        document.removeEventListener("keydown", onKey);
        unbindNav();
        resolve(a);
      }
      function confirmNext() {
        if (allAnswered()) finish("next");
      }
      function goPrev() {
        if (!isFirst) finish("prev");
      }
      function onKey(e) {
        if (e.key === "Enter" || e.key === "ArrowRight") { confirmNext(); return; }
        if (e.key === "ArrowLeft") { goPrev(); return; }
        // Number keys apply to the active row (first unanswered), same
        // semantics as app/ui/common_widgets.py's rating_scale_multi.
        const v = Number(e.key);
        if (Number.isNaN(v) || v < item.scale_min || v > item.scale_max) return;
        const active = item.items.find((sub) => values[sub.id] === null);
        if (active) setValue(active.id, v);
      }
      const unbindNav = bindNav(confirmNext, isFirst ? null : goPrev);
      document.addEventListener("keydown", onKey);
    });

    if (action === "next") {
      for (const sub of item.items) {
        logEvent("questionnaire_response", { item_id: sub.id, value: values[sub.id], rt: rts[sub.id] }, "preparation");
      }
    }
    answers[item.id] = { values, rts };
    return action;
  }

  async function renderGroupItem(item, answers, isFirst) {
    if (item.prompt) await showMessage(item.prompt);
    showScreen("questionnaire-screen");
    const container = questionnaireContainer();
    const saved = answers[item.id] || { values: {}, rts: {} };
    const values = { ...saved.values };
    const rts = { ...saved.rts };
    const fieldEls = {};
    const start = performance.now();
    const fields = item.fields.filter((f) => f.enabled !== false);

    function isVisible(field) {
      if (!field.show_if) return true;
      return values[field.show_if.field] === field.show_if.equals;
    }

    function refreshVisibility() {
      fields.forEach((field) => {
        fieldEls[field.id].wrap.classList.toggle("hidden", !isVisible(field));
      });
      setNav({ showPrev: !isFirst, nextEnabled: unfilledVisibleFields().length === 0 });
    }

    const choiceFieldButtons = {}; // field.id -> button[], for keyboard shortcuts below

    function selectChoice(field, index, buttons) {
      const opt = field.options[index];
      values[field.id] = opt;
      rts[field.id] = (performance.now() - start) / 1000; // always the latest pick this visit
      buttons.forEach((b, i) => b.classList.toggle("selected", i === index));
      refreshVisibility();
    }

    fields.forEach((field) => {
      const wrap = document.createElement("div");
      wrap.className = "q-group-field";
      const label = document.createElement("div");
      label.className = "q-field-label";
      label.textContent = field.prompt;
      wrap.appendChild(label);

      let inputEl;
      if (field.type === "choice") {
        const options = document.createElement("div");
        options.className = "q-options";
        const buttons = [];
        field.options.forEach((opt, i) => {
          const btn = document.createElement("button");
          btn.className = "q-option-btn";
          btn.textContent = opt;
          if (values[field.id] === opt) btn.classList.add("selected");
          btn.addEventListener("click", () => selectChoice(field, i, buttons));
          buttons.push(btn);
          options.appendChild(btn);
        });
        choiceFieldButtons[field.id] = buttons;
        wrap.appendChild(options);
      } else if (field.type === "dropdown") {
        inputEl = document.createElement("select");
        inputEl.className = "q-dropdown";
        const blank = document.createElement("option");
        blank.value = "";
        blank.textContent = "-- select --";
        inputEl.appendChild(blank);
        field.options.forEach((opt) => {
          const optionEl = document.createElement("option");
          optionEl.value = opt;
          optionEl.textContent = opt;
          inputEl.appendChild(optionEl);
        });
        if (values[field.id]) inputEl.value = values[field.id];
        inputEl.addEventListener("change", () => {
          values[field.id] = inputEl.value || null;
          rts[field.id] = (performance.now() - start) / 1000;
          refreshVisibility();
        });
        wrap.appendChild(inputEl);
      } else { // text
        inputEl = document.createElement("input");
        inputEl.type = "text";
        inputEl.className = "q-text-input";
        if (values[field.id]) inputEl.value = values[field.id];
        inputEl.addEventListener("input", () => {
          values[field.id] = inputEl.value || null;
          rts[field.id] = (performance.now() - start) / 1000;
          setNav({ showPrev: !isFirst, nextEnabled: unfilledVisibleFields().length === 0 });
        });
        wrap.appendChild(inputEl);
      }

      fieldEls[field.id] = { wrap, inputEl };
      container.appendChild(wrap);
    });

    const hint = document.createElement("div");
    hint.className = "q-continue-hint";
    hint.textContent = "Fill in the fields above, then press ENTER or click Next.";
    container.appendChild(hint);

    function unfilledVisibleFields() {
      return fields.filter(
        (f) => isVisible(f) && (values[f.id] === undefined || values[f.id] === null || values[f.id] === ""),
      );
    }

    function flashMissing(missing) {
      // Visible feedback for "why didn't ENTER/SPACE do anything" -- without
      // this, a forgotten dropdown (easy to miss -- no placeholder text
      // stands out) looks exactly like a stuck/broken screen.
      missing.forEach((f) => fieldEls[f.id].wrap.classList.add("q-field-missing"));
      hint.textContent = `Please answer: ${missing.map((f) => f.prompt).join(", ")}`;
      hint.classList.add("q-continue-hint-warning");
      setTimeout(() => {
        missing.forEach((f) => fieldEls[f.id].wrap.classList.remove("q-field-missing"));
        hint.textContent = "Fill in the fields above, then press ENTER or click Next.";
        hint.classList.remove("q-continue-hint-warning");
      }, 2000);
    }

    refreshVisibility();

    const action = await new Promise((resolve) => {
      function finish(a) {
        document.removeEventListener("keydown", onKey);
        unbindNav();
        resolve(a);
      }
      function confirmNext() {
        const missing = unfilledVisibleFields();
        if (missing.length > 0) { flashMissing(missing); return; }
        finish("next");
      }
      function goPrev() {
        if (!isFirst) finish("prev");
      }
      function onKey(e) {
        // Space/ArrowLeft only act as navigation when NOT typing/selecting in
        // a field -- a free-text answer (e.g. gender_self_describe) is
        // allowed to contain spaces, and a focused <select> uses its own
        // arrow-key behavior, so both must pass through untouched there.
        const activeEl = document.activeElement;
        const isTyping = activeEl && activeEl.classList.contains("q-text-input");
        const isSelecting = activeEl && activeEl.tagName === "SELECT";
        if (e.key === "Enter" || (e.key === " " && !isTyping)) { confirmNext(); return; }
        if (e.key === "ArrowLeft" && !isTyping && !isSelecting) { goPrev(); return; }
        // Number keys select an option on the first unanswered, visible
        // choice-type field -- dropdown/text fields have no natural
        // single-keypress equivalent, so they stay click/type-only.
        const i = Number(e.key) - 1;
        if (Number.isNaN(i) || i < 0 || isTyping || isSelecting) return;
        const activeField = fields.find(
          (f) => f.type === "choice" && isVisible(f) && (values[f.id] === undefined || values[f.id] === null),
        );
        if (activeField && i < activeField.options.length) {
          selectChoice(activeField, i, choiceFieldButtons[activeField.id]);
        }
      }
      const unbindNav = bindNav(confirmNext, isFirst ? null : goPrev);
      document.addEventListener("keydown", onKey);
    });

    if (action === "next") {
      for (const field of fields) {
        if (!isVisible(field)) continue;
        logEvent("questionnaire_response", { item_id: field.id, value: values[field.id], rt: rts[field.id] }, "preparation");
      }
    }
    answers[item.id] = { values, rts };
    return action;
  }

  // ---- Break screen -------------------------------------------------------

  function runBreak(durationSec) {
    showScreen("break-screen");
    const fill = el("break-progress");
    fill.style.transition = "none";
    fill.style.transform = "scaleX(1)";
    void fill.offsetWidth;
    fill.style.transition = `transform ${durationSec}s linear`;
    fill.style.transform = "scaleX(0)";
    return new Promise((resolve) => setTimeout(resolve, durationSec * 1000));
  }

  // ---- Main flow ----------------------------------------------------------

  async function main() {
    bootstrap = await api().get_bootstrap();

    // Several round trips at true session start AND end (not just one) --
    // see ClockSync.runCheckpoints -- so offline analysis can fit drift
    // between THIS window's clock and the host's, not just a single
    // offset. Routed to task="preparation" since that's the phase this
    // falls inside; see the matching session_end checkpoints below.
    await ClockSync.runCheckpoints((type, fields) => logEvent(type, fields, "preparation"), "session_start");

    logEvent("phase_start", {}, "preparation");
    await showMessage(heading("DATA COLLECTION SESSION") + "\n\nEmotion task, a stress task, then a short attention/focus task.");
    logEvent("consent_given", {}, "preparation");

    if (!bootstrap.skip_questionnaire) {
      await runQuestionnaire(bootstrap.questionnaire_items);
    }

    if (!bootstrap.skip_familiarization) {
      logEvent("phase_start", {}, "familiarization");
      await showMessage(
        heading("FAMILIARIZATION") +
        "\n\nLet's quickly walk through what each part of today's session looks like.\n" +
        "Nothing in this section is scored.",
      );
      await showMessage(
        "Comfort check: make sure your headphones and sensors feel secure, and you can see the screen clearly.",
        { hint: "Press SPACE when ready." },
      );

      await showMessage(
        "Preview: EMOTION task. You'll watch a short clip, then answer a couple of quick questions " +
        "about how it made you feel, like this.",
        { hint: "Press SPACE to play the practice clip." },
      );
      if (!(await runGuarded(() => api().run_practice_video()))) return;

      await showMessage(
        "Preview: STRESS task, part 1 -- MENTAL ARITHMETIC. Math problems fall like raindrops -- in " +
        "the real round, you'll type the answer before they reach the bottom. Watch how it works below.",
        { hint: "Press SPACE to watch a short demo." },
      );
      if (!(await runGuarded(() => api().run_practice_raindrop_demo()))) return;

      await showMessage(
        "Preview: STRESS task, part 2 -- HIGHWAY DRIVING. Hazards approach in different lanes -- " +
        "press LEFT/RIGHT (or A/D) to dodge into a clear lane before they reach you. Watch how it " +
        "works below.",
        { hint: "Press SPACE to watch a short demo." },
      );
      if (!(await runGuarded(() => api().run_practice_highway_demo()))) return;

      await showMessage(
        "Preview: ATTENTION task, part 1 -- SCHULTE TABLE. A grid of numbers will appear scattered " +
        "in random positions -- click them in order starting from 1. There's no time limit. Watch " +
        "how it works below.",
        { hint: "Press SPACE to watch a short demo." },
      );
      if (!(await runGuarded(() => api().run_practice_schulte_demo()))) return;

      await showMessage(
        "Preview: ATTENTION task, part 2 -- STROOP TEST. A color word will appear -- click the " +
        "colored button matching the INK color it's printed in, not the word itself. There's no " +
        "time limit. Watch how it works below.",
        { hint: "Press SPACE to watch a short demo." },
      );
      if (!(await runGuarded(() => api().run_practice_stroop_demo()))) return;

      await showMessage("That's everything. The real tasks begin now.");
      logEvent("phase_end", {}, "familiarization");
    }

    await showMessage(
      heading("TASK 2: EMOTION") +
      "\n\nYou will watch a series of short video clips, grouped into three blocks.\n\n" +
      "Each clip starts with a brief '+' fixation cross, then plays with sound. Afterwards you'll " +
      "answer a couple of quick questions about how it made you feel -- whether it was positive, " +
      "negative, or neutral overall, and how pleasant, arousing, and likeable it was.\n\n" +
      "For every question, you can either press a number key or click the on-screen option -- " +
      "whichever is more comfortable.\n\n" +
      "Some clips are lighthearted, others may be sad, tense, or unpleasant -- that's expected and " +
      "part of the study.\n\n" +
      "Your participation is voluntary:\n" +
      "- Press N before or during a clip to skip it\n" +
      "- Press B to skip the rest of the current block\n" +
      "- Press Esc to stop the session entirely",
    );
    if (!(await runGuarded(() => api().run_emotion_task()))) return;

    await runBreak(bootstrap.break_duration_sec);

    await showMessage(
      heading("TASK 3: MENTAL ARITHMETIC") +
      "\n\nMath problems will fall like raindrops. Type the answer before each one reaches the " +
      "bottom of the screen -- you can use the on-screen keypad or your keyboard.\n\n" +
      "Answer as many as you can. Your score only ever goes up, and the current all-time high " +
      "score is shown on screen.\n\n" +
      "Your participation is voluntary:\n" +
      "- Press B at any time to skip the current round\n" +
      "- Press Esc to stop the session entirely",
    );
    if (!(await runGuarded(() => api().run_stress_task()))) return;

    // Highway driving is the 2nd half of "stress" (per PI request
    // 2026-08-05) -- runs immediately after raindrop, no break in between;
    // the break below covers both halves at once.
    await showMessage(
      heading("TASK 3, PART 2: HIGHWAY DRIVING") +
      "\n\nThe car auto-drives down the highway. Hazards will approach ahead in different lanes -- " +
      "press LEFT/RIGHT (or A/D) to dodge into a clear lane before one reaches you.\n\n" +
      "A collision is just logged, not game over -- the drive keeps going, so don't worry about " +
      "\"losing\".\n\n" +
      "Your participation is voluntary:\n" +
      "- Press B at any time to skip the current tier\n" +
      "- Press Esc to stop the session entirely",
    );
    if (!(await runGuarded(() => api().run_highway_task()))) return;

    await runBreak(bootstrap.break_duration_after_task2_sec);

    await showMessage(
      heading("TASK 4: ATTENTION / FOCUS") +
      "\n\nA few short trials, alternating between two quick games:\n\n" +
      "SCHULTE TABLE -- a grid of numbers appears scattered in random positions. Click them in " +
      "order starting from 1. There's no time limit.\n\n" +
      "STROOP TEST -- a color word appears. Click the colored button matching the INK color it's " +
      "printed in, NOT the word itself. There's no time limit.\n\n" +
      "Each trial starts with a brief '+' fixation cross, same as before.\n\n" +
      "Your participation is voluntary:\n" +
      "- Press B at any time to skip the current trial\n" +
      "- Press Esc to stop the session entirely",
    );
    if (!(await runAttentionFocusTask())) return;

    await showMessage(
      heading("CONCLUSION") +
      "\n\nThat concludes the emotion, stress, and attention/focus portion of the session. Thank you!\n\n" +
      "Next: the SART attention/focus task runs separately as its own program.",
      { hint: "Press SPACE to end." },
    );
    await ClockSync.runCheckpoints((type, fields) => logEvent(type, fields, "debrief"), "session_end");
    logEvent("session_end", {}, "debrief");
    sessionEnded = true;
    api().close_window();
  }

  // ---- Attention/focus task (Schulte table + Stroop test) ------------------
  //
  // bootstrap.attention_focus_sequence is the pre-shuffled game order built
  // once at session start (app/tasks/attention_focus.py's build_sequence())
  // -- this just walks it trial by trial, showing a short reminder before
  // each repeat of a game already introduced above, and dispatching to
  // run_attention_focus_trial(i) (which resolves the actual per-trial
  // content server-side).

  const ATTENTION_FOCUS_REMINDERS = {
    schulte: "Next trial: SCHULTE TABLE -- click the numbers in order, starting from 1. No time limit.",
    stroop: "Next trial: STROOP TEST -- click the button matching the INK color, not the word. No time limit.",
  };

  async function runAttentionFocusTask() {
    const sequence = bootstrap.attention_focus_sequence || [];
    for (let i = 0; i < sequence.length; i++) {
      const game = sequence[i];
      await showMessage(
        `Trial ${i + 1} of ${sequence.length}\n\n${ATTENTION_FOCUS_REMINDERS[game]}`,
        { hint: "Press SPACE to begin." },
      );
      if (!(await runGuarded(() => api().run_attention_focus_trial(i)))) return false;
    }
    return true;
  }

  window.addEventListener("pywebviewready", main);
})();
