// Shared in-webview Likert self-report widget. Extracted from
// app/webui/video_player/player.js's runRatings() (kept there UNCHANGED --
// this is a copy, not a refactor, so the working emotion task isn't touched
// by this change) so any *_app.py webview task can render the same
// click-or-number-key rating rows and log them the same way, instead of
// each task growing its own bespoke self-report UI. See
// app/webui/highway/game.js for the first reuse of this.
//
// Usage (see highway/index.html for the matching #rating-screen/#rating-rows
// markup and highway/style.css for the matching .rating-row/.rating-btn/etc.
// CSS, both copied verbatim from video_player/index.html + style.css so this
// widget doesn't need its own styling):
//
//   const result = await RatingWidget.run({
//     container: document.getElementById("rating-rows"),
//     items: ["stress", "workload"],
//     prompts: { stress: ["How stressed...", "Not at all", "Extremely"], ... },
//     scaleMin: 1, scaleMax: 7,
//     isQuitKey: (e) => e.key === "Escape",   // optional, defaults to Escape
//   });
//   // result === { values: {stress: 5, ...}, rts: {stress: 1.2, ...}, quit: false }
//   // or { quit: true } immediately if isQuitKey fires before every item is answered.
(() => {
  function run({ container, items, prompts, scaleMin, scaleMax, isQuitKey }) {
    isQuitKey = isQuitKey || ((e) => e.key === "Escape");
    container.innerHTML = "";
    const values = {};
    const rts = {};
    const rowsByKey = {};

    items.forEach((key) => {
      const [prompt, leftLabel, rightLabel] = prompts[key];
      values[key] = null;
      const row = document.createElement("div");
      row.className = "rating-row";
      const promptEl = document.createElement("div");
      promptEl.className = "rating-prompt";
      promptEl.textContent = prompt;
      row.appendChild(promptEl);

      const scale = document.createElement("div");
      scale.className = "rating-scale";
      const buttons = [];
      for (let v = scaleMin; v <= scaleMax; v++) {
        const btn = document.createElement("button");
        btn.className = "rating-btn";
        btn.textContent = String(v);
        scale.appendChild(btn);
        buttons.push(btn);
      }
      row.appendChild(scale);

      const endpoints = document.createElement("div");
      endpoints.className = "rating-endpoints";
      endpoints.innerHTML = `<span>${leftLabel}</span><span>${rightLabel}</span>`;
      row.appendChild(endpoints);

      container.appendChild(row);
      rowsByKey[key] = buttons;
    });

    // Onset is captured the instant the rows are built (rendering happens
    // synchronously above), so per-item RT is "time since this whole rating
    // screen appeared", same convention runRatings() in player.js uses --
    // the caller should log a *_prompt_onset event right before calling
    // run() if it also wants that instant recorded on its own row.
    const onsetMs = performance.now();

    return new Promise((resolve) => {
      function setValue(key, v) {
        values[key] = v;
        if (rts[key] === undefined) rts[key] = (performance.now() - onsetMs) / 1000;
        rowsByKey[key].forEach((btn, i) => btn.classList.toggle("selected", scaleMin + i === v));
      }
      items.forEach((key) => {
        rowsByKey[key].forEach((btn, i) => btn.addEventListener("click", () => setValue(key, scaleMin + i)));
      });

      function onKey(e) {
        if (isQuitKey(e)) { cleanup(); resolve({ quit: true }); return; }
        const active = items.find((k) => values[k] === null);
        if (active && e.key >= String(scaleMin) && e.key <= String(scaleMax)) {
          setValue(active, Number(e.key));
          return;
        }
        if ((e.key === "Enter" || e.key === " ") && !items.some((k) => values[k] === null)) {
          cleanup();
          resolve({ values, rts, quit: false });
        }
      }
      function cleanup() { document.removeEventListener("keydown", onKey); }
      document.addEventListener("keydown", onKey);
    });
  }

  window.RatingWidget = { run };
})();
