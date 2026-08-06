// Emotion video task front end. Clip list, block layout, and rating/prompt
// copy all come from Python (app.tasks.emotion_faced's select_task_clips /
// RATING_PROMPTS / EMOTION_OPTIONS, reused unchanged -- see
// app/tasks/emotion_web_player.py) -- this file only renders, plays,
// collects responses, and logs through the pywebview bridge.
(() => {
  let bootstrap = null;
  let quitting = false;

  const el = (id) => document.getElementById(id);
  // See app/webui/shared/clock_sync.js -- shared with every other webui
  // task and the session shell (browser performance.now() tagging on every
  // event, plus sync_checkpoint round trips at session start/end -- see
  // app/eventlog/event_logger.py's timestamp_monotonic and
  // app/webui/bridge.py's sync_checkpoint()).
  const logEvent = ClockSync.wrapLogEvent((eventType, fields) => window.pywebview.api.log_event(eventType, fields || {}));

  // SkipClip/SkipBlock/Quit mirror app/ui/common_widgets.py's
  // SkipTrial/SkipBlock/UserQuit -- now usable DURING clip playback too,
  // since we own this whole window (VLC couldn't give us that).
  class SkipClip extends Error {}
  class SkipBlock extends Error {}
  class Quit extends Error {}

  // Every awaited step below (fixation/clip/choice/ratings/rest) registers
  // exactly ONE keydown listener and removes it on every exit path (timeout,
  // local completion, or a global skip key) -- avoids leaking a listener per
  // step, which a naive two-listeners-per-step design would do whenever the
  // step is abandoned via Escape/N/B rather than completing normally.
  function checkGlobalSkip(e) {
    if (e.key === "Escape") { quitting = true; return new Quit(); }
    if (e.key.toLowerCase() === "n") return new SkipClip();
    if (e.key.toLowerCase() === "b") return new SkipBlock();
    return null;
  }

  // handler(e) returns undefined/null to keep waiting, or {value} to resolve.
  // Global skip keys are checked before handler() on every keydown.
  function stepWithKeys(handler) {
    return new Promise((resolve, reject) => {
      function onKey(e) {
        const skip = checkGlobalSkip(e);
        if (skip) { cleanup(); reject(skip); return; }
        const result = handler(e);
        if (result) { cleanup(); resolve(result.value); }
      }
      function cleanup() { document.removeEventListener("keydown", onKey); }
      document.addEventListener("keydown", onKey);
    });
  }

  function delayWithSkipKeys(ms) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => { cleanup(); resolve(); }, ms);
      function onKey(e) {
        const skip = checkGlobalSkip(e);
        if (skip) { cleanup(); reject(skip); }
      }
      function cleanup() { clearTimeout(timer); document.removeEventListener("keydown", onKey); }
      document.addEventListener("keydown", onKey);
    });
  }

  function playVideoWithSkipKeys(video, safetyTimeoutMs, clip) {
    return new Promise((resolve, reject) => {
      let settled = false;
      const timer = setTimeout(finish, safetyTimeoutMs);
      function finish() { if (settled) return; settled = true; cleanup(); resolve(); }
      function onKey(e) {
        if (settled) return;
        const skip = checkGlobalSkip(e);
        if (skip) { settled = true; cleanup(); reject(skip); }
      }
      // Logs the video's actual DECODED resolution (video.videoWidth/Height --
      // read straight from the file by the browser's decoder, so this is
      // ground truth, not something our own code could get wrong), so a
      // session's events.jsonl can be checked after the fact for "was this
      // really played at its native resolution". below_720p flags clips
      // known to be under the resolution floor (see
      // scripts/emotion_clip_picks_2026-08-04.txt's "<-- below 720p" notes)
      // so that's visible in the log, not just guessed at.
      //
      // object_fit logs the ACTUAL computed CSS object-fit value in effect
      // on #clip-video (should be "contain": uniform scale + letterbox, no
      // distortion) -- this is the real "is it stretched" check. An earlier
      // version of this instead compared video.getBoundingClientRect()'s box
      // aspect ratio against the native aspect ratio, which is WRONG: with
      // object-fit: contain, the element's own box is always ~full-screen
      // (whatever size its CSS says, e.g. 100% width/height) and is *meant*
      // to have a different aspect ratio than the letterboxed content inside
      // it -- that comparison flagged perfectly fine playback (e.g. a
      // non-16:9 source like VID_603's 1280x676) as "stretched" just because
      // the screen itself is 16:9. Checking the computed style directly is
      // what actually catches a real regression (e.g. object-fit
      // accidentally changed to `fill`, which WOULD stretch it).
      function onLoadedMetadata() {
        logEvent("clip_video_metadata", {
          clip_id: clip.clip_id,
          native_width: video.videoWidth,
          native_height: video.videoHeight,
          object_fit: getComputedStyle(video).objectFit,
          below_720p: video.videoHeight < 720,
        });
      }
      function cleanup() {
        clearTimeout(timer);
        document.removeEventListener("keydown", onKey);
        video.onended = null;
        video.onerror = null;
        video.onloadedmetadata = null;
        // Stop playback immediately on EVERY exit path (skip, quit, natural
        // end, or safety timeout) -- previously this only happened in
        // runClip() after a successful (resolved) await, so skipping a clip
        // (N/B/Esc) rejected this promise and left the <video> element
        // playing -- hidden but still with audio -- straight through the
        // fixation/rest/next-clip screens until the next real clip's
        // video.src assignment finally cut it off.
        video.pause();
        video.src = "";
      }
      document.addEventListener("keydown", onKey);
      video.onended = finish;
      video.onerror = finish;
      video.onloadedmetadata = onLoadedMetadata;
      video.play().catch(finish);
    });
  }

  function showScreen(name) {
    document.querySelectorAll(".screen").forEach((s) => s.classList.add("hidden"));
    if (name) el(name).classList.remove("hidden");
  }

  async function runFixation(durationSec, progressLabel) {
    showScreen("fixation-screen");
    el("fixation-progress-label").textContent = progressLabel || "";
    await delayWithSkipKeys(durationSec * 1000);
  }

  // Block-level physiological baseline periods -- per PI request 2026-08-05,
  // replacing the old per-clip pre-stimulus fixation (5s x every clip, too
  // short/numerous to be useful for windowed DL analysis) with 3 longer,
  // chunkier touchpoints per block instead: once before the block's first
  // clip, once at the block's midpoint, once after its last clip. Reuses
  // runFixation's existing fixation-cross screen (a baseline period LOOKS
  // exactly like an extended fixation, just longer and logged distinctly so
  // it's unambiguous in events.jsonl which seconds are "baseline" vs. the
  // brief inter-clip rest -- see runBlock() below for where these fire.
  async function runBaselinePeriod(durationSec, blockIndex, position) {
    logEvent("baseline_start", { block_index: blockIndex, position, duration_sec: durationSec });
    await runFixation(durationSec, `Baseline recording (${position})`);
    logEvent("baseline_end", { block_index: blockIndex, position });
  }

  async function runClip(clip) {
    showScreen("video-screen");
    logEvent("clip_playback_start", { clip_id: clip.clip_id, file_path: clip.file_path });
    const start = performance.now();

    const video = el("clip-video");
    const placeholder = el("placeholder");

    if (clip.placeholder) {
      video.classList.add("hidden");
      placeholder.classList.remove("hidden");
      placeholder.style.background = clip.placeholder_color || "#333";
      el("placeholder-label").textContent = `[placeholder clip -- file not found]\n\n${clip.clip_id} (${clip.fine_grained_label})`;
      await delayWithSkipKeys(clip.duration_sec * 1000);
    } else {
      video.classList.remove("hidden");
      placeholder.classList.add("hidden");
      video.src = clip.file_url;
      video.muted = false;
      // Safety timeout mirrors the old subprocess.run timeout -- real
      // playback runs to the file's actual end via the `ended` event.
      // playVideoWithSkipKeys's own cleanup() always pauses + clears
      // video.src on every exit path (including a skip/quit rejection), so
      // no separate stop-playback call is needed here.
      await playVideoWithSkipKeys(video, (clip.duration_sec + 45) * 1000, clip);
    }
    logEvent("clip_playback_end", { clip_id: clip.clip_id, wall_duration_sec: (performance.now() - start) / 1000 });
  }

  async function runDiscreteChoice(prompt, options) {
    showScreen("choice-screen");
    el("choice-prompt").textContent = prompt;
    const container = el("choice-options");
    container.innerHTML = "";
    let selected = null;
    const buttons = options.map((opt, i) => {
      const btn = document.createElement("button");
      btn.className = "option-btn";
      btn.textContent = opt;
      container.appendChild(btn);
      return btn;
    });
    function select(i) {
      selected = i;
      buttons.forEach((b, j) => b.classList.toggle("selected", j === i));
    }
    buttons.forEach((btn, i) => btn.addEventListener("click", () => select(i)));
    const start = performance.now();

    return stepWithKeys((e) => {
      if (e.key >= "1" && e.key <= String(options.length)) { select(Number(e.key) - 1); return null; }
      if ((e.key === "Enter" || e.key === " ") && selected !== null) {
        return { value: { value: options[selected], rt: (performance.now() - start) / 1000 } };
      }
      return null;
    });
  }

  async function runRatings(items, prompts, scaleMin, scaleMax) {
    showScreen("rating-screen");
    const container = el("rating-rows");
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

    const start = performance.now();
    function setValue(key, v) {
      values[key] = v;
      if (rts[key] === undefined) rts[key] = (performance.now() - start) / 1000;
      rowsByKey[key].forEach((btn, i) => btn.classList.toggle("selected", scaleMin + i === v));
    }
    items.forEach((key) => {
      rowsByKey[key].forEach((btn, i) => btn.addEventListener("click", () => setValue(key, scaleMin + i)));
    });

    return stepWithKeys((e) => {
      const active = items.find((k) => values[k] === null);
      if (active && e.key >= String(scaleMin) && e.key <= String(scaleMax)) {
        setValue(active, Number(e.key));
        return null;
      }
      if ((e.key === "Enter" || e.key === " ") && !items.some((k) => values[k] === null)) {
        return { value: { values, rts } };
      }
      return null;
    });
  }

  async function runRest(durationSec) {
    showScreen("rest-screen");
    const fill = el("rest-progress");
    fill.style.transition = "none";
    fill.style.transform = "scaleX(1)";
    void fill.offsetWidth;
    fill.style.transition = `transform ${durationSec}s linear`;
    fill.style.transform = "scaleX(0)";
    await delayWithSkipKeys(durationSec * 1000);
  }

  async function runOneClip(clip, blockIndex, trialIndex, conditionLabel, progressLabel) {
    logEvent("trial_start", {
      block_index: blockIndex, trial_index: trialIndex, condition_label: conditionLabel,
      clip_id: clip.clip_id, fine_grained_label: clip.fine_grained_label,
    });
    try {
      // No more per-clip fixation here -- see runBaselinePeriod() above for
      // why (per PI request 2026-08-05); progressLabel is now unused by this
      // step but kept as a param for callers/future use.
      await runClip(clip);
      const choice = await runDiscreteChoice(bootstrap.emotion_prompt, bootstrap.emotion_options);
      const ratings = await runRatings(
        bootstrap.rating_items, bootstrap.rating_prompts, bootstrap.rating_scale_min, bootstrap.rating_scale_max,
      );
      const payload = {
        block_index: blockIndex, trial_index: trialIndex, condition_label: conditionLabel,
        clip_id: clip.clip_id, emotion_pick: choice.value, emotion_pick_rt: choice.rt, skipped: false,
      };
      for (const key of bootstrap.rating_items) {
        payload[key] = ratings.values[key];
        payload[`${key}_rt`] = ratings.rts[key] ?? null;
      }
      logEvent("rating_response", payload);
    } catch (err) {
      if (err instanceof Quit || err instanceof SkipClip || err instanceof SkipBlock) {
        logEvent("rating_response", {
          block_index: blockIndex, trial_index: trialIndex, condition_label: conditionLabel,
          clip_id: clip.clip_id, emotion_pick: null, emotion_pick_rt: null, skipped: true,
        });
      }
      throw err;
    }
  }

  async function runBlock(block, blockIndex) {
    const conditionLabels = [...new Set(block.map((c) => c.valence_group))].sort();
    logEvent("block_start", { block_index: blockIndex, condition_labels: conditionLabels, num_trials: block.length });
    let skippedBlock = false;

    // 3 baseline touchpoints per block (start/mid/end) instead of the old
    // per-clip fixation -- see runBaselinePeriod() above. The midpoint one
    // REPLACES the ordinary inter-clip rest at that position (not stacked on
    // top of it) so a block isn't padded with both back to back.
    const midGapIndex = Math.floor(block.length / 2) - 1; // rest/baseline slot right after this clip index

    async function runSkippable(promiseFactory) {
      try { await promiseFactory(); }
      catch (err) {
        if (err instanceof SkipBlock) { skippedBlock = true; }
        else if (!(err instanceof SkipClip)) throw err; // Quit
      }
    }

    await runSkippable(() => runBaselinePeriod(bootstrap.block_baseline_start_sec, blockIndex, "start"));

    for (let i = 0; i < block.length && !skippedBlock; i++) {
      try {
        await runOneClip(block[i], blockIndex, i, block[i].valence_group, `Clip ${i + 1} of ${block.length}`);
      } catch (err) {
        if (err instanceof SkipBlock) { skippedBlock = true; break; }
        if (!(err instanceof SkipClip)) throw err; // Quit
      }
      if (skippedBlock || i >= block.length - 1) continue;
      if (i === midGapIndex) await runSkippable(() => runBaselinePeriod(bootstrap.block_baseline_mid_sec, blockIndex, "mid"));
      else await runSkippable(() => runRest(bootstrap.rest_duration_sec));
    }

    if (!skippedBlock) {
      await runSkippable(() => runBaselinePeriod(bootstrap.block_baseline_end_sec, blockIndex, "end"));
    }

    logEvent("block_end", { block_index: blockIndex, condition_labels: conditionLabels, skipped: skippedBlock });
  }

  async function runBlockIntro(blockIndex, total) {
    showScreen("block-intro-screen");
    el("block-intro-heading").textContent = `VIDEO BLOCK ${String.fromCharCode(65 + blockIndex)} (Block ${blockIndex + 1} of ${total})`;
    await stepWithKeys((e) => (e.key === " " || e.key === "Enter" ? { value: true } : null));
  }

  async function main() {
    bootstrap = await window.pywebview.api.get_bootstrap();

    try {
      if (bootstrap.practice) {
        const clip = bootstrap.blocks[0][0];
        try {
          await runOneClip(clip, 0, 0, clip.valence_group, null);
        } catch (err) {
          // runOneClip logs a "skipped" rating_response for Quit/SkipClip/
          // SkipBlock, then unconditionally re-throws -- the real (non-
          // practice) flow's runBlock() catches SkipClip/SkipBlock per-clip
          // (see above) so only Quit ever reaches this far there. This
          // practice branch called runOneClip() directly with no such
          // wrapper, so pressing N (or B) during the familiarization
          // preview threw SkipClip/SkipBlock straight past the outer
          // catch below (which only swallows Quit) -- an uncaught
          // exception in this event-listener callback, silently dropped as
          // an unhandled promise rejection, so neither close_window() nor
          // request_quit() ever ran and the window just sat there. Fixed
          // by treating a skip here the same as the clip finishing
          // normally -- there's nothing left to skip TO in a one-clip
          // preview anyway.
          if (!(err instanceof SkipClip) && !(err instanceof SkipBlock)) throw err;
        }
      } else {
        // Skipped in the practice branch above, same reasoning as every
        // other task's demo/practice mode -- not part of the scored dataset.
        await ClockSync.runCheckpoints(logEvent, "session_start");
        logEvent("task_start", {});
        for (let b = 0; b < bootstrap.blocks.length; b++) {
          await runBlockIntro(b, bootstrap.blocks.length);
          await runBlock(bootstrap.blocks[b], b);
        }
        logEvent("task_end", {});
        await ClockSync.runCheckpoints(logEvent, "session_end");
      }
    } catch (err) {
      if (!(err instanceof Quit)) throw err;
    }

    if (quitting) window.pywebview.api.request_quit();
    else window.pywebview.api.close_window();
  }

  window.addEventListener("pywebviewready", main);
})();
