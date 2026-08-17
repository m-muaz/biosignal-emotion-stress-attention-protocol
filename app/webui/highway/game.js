// Highway dodge attention/stress game front end. Added per PI discussion
// 2026-08-04 as a second, complementary attention task alongside SART (see
// README "Attention task (SART)"): unlike SART's single go/no-go button
// press, this needs continuous monitoring of self-position AND several
// lanes of oncoming hazards at once, plus a *directional* (not just
// present/absent) response -- see app/tasks/attention_highway.py for the
// full rationale.
//
// No permadeath: a collision is just logged and the drive continues, same
// "fixed-duration, mistakes don't end the round" design as
// app/webui/raindrop/game.js. Difficulty ramps across discrete TIERS
// (bootstrap.tiers, built by attention_highway.py from
// attention_highway_task.trial_order/.tiers) -- each tier's own pacing is
// constant, only the jump BETWEEN tiers is a deliberate step up, so
// analysis can still isolate vigilance decrement within a tier from the
// difficulty increase between them. pickSpawnLane() below also guarantees
// at least one lane is always free of any live hazard, in every tier, AND
// implements the anti-camping nudge: the longer the participant sits in
// one lane, the more likely the next hazard spawns directly into it (can't
// literally force a keypress, but makes staying put increasingly risky
// instead of a free pass -- see camp_grace_sec/camp_ramp_sec/
// camp_max_probability in attention_highway_task's config).
//
// Repurposing as a stress task (per PI discussion 2026-08-05): tier speed
// (spawn_interval_range_sec/fall_duration_sec, i.e. nominal_difficulty) is
// the experimenter-controlled INDUCTION -- faster tiers demand more
// continuous monitoring/cognitive load, which is the mechanism for stress
// induction. Collisions/near-misses/reaction time are the resulting
// BEHAVIORAL outcome of that induction, not a stress label themselves --
// this file only ever logs raw behavioral telemetry (nominal_difficulty
// stays a separate field from collision/avoidance outcomes everywhere
// below); turning "faster tier -> more collisions -> did stress rise" into
// an actual answer is deliberately left to offline analysis (behavioral
// strain computed from these raw events, cross-checked against self-report
// and physiology) rather than baked in here as a hardcoded label. Also NOT
// closed-loop: collision rate never feeds back into difficulty -- keeping
// the induction fixed/scripted is what makes the dose-response question
// ("did the fixed speed step change the outcome") answerable at all,
// instead of confounding the manipulation with the thing it's meant to
// explain.
(() => {
  let bootstrap = null;
  let lanes = 3;
  let playerLane = 1;
  let lastLaneChangeTime = 0; // performance.now() timestamp -- drives the anti-camping nudge
  let avoided = 0;
  let collisions = 0;
  // Where (as a % of #road's height) a falling obstacle's CENTER needs to
  // land to actually visually reach #player-car -- recomputed from the real
  // rendered layout (see updateObstacleEndTopPct()) rather than a guessed
  // constant, since collision is judged the instant an obstacle's fall timer
  // elapses (obstacle.lane === playerLane in resolveObstacle()), not by any
  // on-screen overlap check. A guessed endpoint that lands lower than the
  // car's actual position meant a hazard could be logged as a collision (or
  // avoided) well after it had visually passed the car on screen (bug
  // reported by a participant 2026-08-13). 98 is only the pre-layout
  // fallback, overwritten before any obstacle ever spawns.
  let obstacleEndTopPct = 98;
  const liveObstacles = new Map(); // id -> { el, lane, spawnTime, blockIndex, conditionLabel, trialIndex, onResolved, rafId, respondedAt }
  let obstacleIdCounter = 0;
  let quitting = false;
  let abortLogged = false; // guards task_abort against being logged more than once

  // Rolling behavioral-outcome history for the state_tick stream below --
  // NOT used for anything gameplay-facing (never read by pickSpawnLane/
  // difficulty logic), purely so state_tick can report a recent
  // avoidance/collision rate and recent RT without recomputing from the
  // full event history offline. Entries older than
  // bootstrap.rolling_window_sec are dropped lazily on each read.
  const recentOutcomes = []; // { t: performance.now(), collided: bool }
  const recentReactionTimes = []; // { t: performance.now(), rtMs: number }
  let lastCollisionTime = null;
  let lastNearMissTime = null;

  const el = (id) => document.getElementById(id);

  // Every event gets the browser-side clock reading attached automatically
  // (not just the ones that already compute a *_ms field by hand), and
  // runSyncCheckpoints() below does the JS<->Python round-trip probes at
  // session start/end -- see app/webui/shared/clock_sync.js (shared with
  // every other webui task and the session shell, so this isn't a
  // highway-only copy) and app/eventlog/event_logger.py's
  // timestamp_monotonic / app/webui/bridge.py's sync_checkpoint() for the
  // Python half.
  const logEvent = ClockSync.wrapLogEvent((eventType, fields) => window.pywebview.api.log_event(eventType, fields));

  function runSyncCheckpoints(labelPrefix) {
    return ClockSync.runCheckpoints(logEvent, labelPrefix, bootstrap.sync_checkpoint_count, bootstrap.sync_checkpoint_gap_sec);
  }

  function logTaskAbort(reason) {
    if (abortLogged) return;
    abortLogged = true;
    quitting = true;
    logEvent("task_abort", { reason });
  }

  function laneCenterPct(lane) {
    return ((lane + 0.5) / lanes) * 100;
  }

  // Measures the real, currently-rendered gap between #road's top and
  // #player-car's vertical center, as a % of #road's height -- this is what
  // spawnObstacle()'s fall animation now targets, instead of a hardcoded
  // guess, so "the fall timer elapsed" and "the hazard visually reached the
  // car" are the same moment regardless of window size/DPI. Only meaningful
  // once #game-screen is actually visible (getBoundingClientRect returns a
  // real height then); called at the start of every driving block, where
  // that's guaranteed true.
  function updateObstacleEndTopPct() {
    const road = el("road").getBoundingClientRect();
    const car = el("player-car").getBoundingClientRect();
    if (!road.height) return; // not laid out yet -- keep the previous/fallback value
    obstacleEndTopPct = ((car.top + car.height / 2 - road.top) / road.height) * 100;
  }

  async function main() {
    bootstrap = await window.pywebview.api.get_bootstrap();
    lanes = bootstrap.lanes;
    buildLaneDividers();
    setPlayerLane(Math.floor(lanes / 2), /* animate */ false);
    installFocusListeners();
    installFrameDropWatchdog();

    if (bootstrap.demo) {
      // Non-interactive preview for familiarization: hazards approach and
      // the car auto-dodges into whichever lane is clear, so the
      // participant just watches the mechanic -- same role
      // stress_raindrop.py's demo mode gives the raindrop game. No sync
      // checkpoints/self-report here -- this run isn't part of the scored
      // dataset, so there's nothing downstream that needs its clock
      // reconciled or its subjective state rated.
      el("hud").classList.add("hidden");
      logEvent("task_start", {});
      avoided = 0;
      collisions = 0;
      lastLaneChangeTime = performance.now();
      await runDrivingBlock(0, bootstrap.tiers[0], /* autoDodge */ true);
      logEvent("task_end", {});
      if (quitting) window.pywebview.api.request_quit();
      else window.pywebview.api.close_window();
      return;
    }

    await runSyncCheckpoints("session_start");

    document.addEventListener("keydown", onKeyDown);
    logEvent("task_start", {});
    avoided = 0;
    collisions = 0;
    updateHud();

    await runBaseline(0);
    lastLaneChangeTime = performance.now(); // camping clock starts once hazards actually begin, not during the hazard-free baseline
    let blockIndex = 1;
    for (const tier of bootstrap.tiers) {
      if (quitting) break;
      await runDrivingBlock(blockIndex++, tier, /* autoDodge */ false);
      if (!quitting && bootstrap.self_report) await runSelfReport(blockIndex - 1, tier);
    }

    logEvent("task_end", {});
    await runSyncCheckpoints("session_end");
    if (quitting) window.pywebview.api.request_quit();
    else showEndScreen();
  }

  // Window focus loss/regain -- e.g. the participant alt-tabs away, or the
  // OS pops a notification that steals focus. Not itself an annotation
  // verdict (offline labeling decides whether the affected window is
  // valid/uncertain/invalid) -- this just records that it happened and
  // when, since without it there'd be no way to tell "attention lapsed"
  // apart from "the OS took the window away" after the fact.
  function installFocusListeners() {
    window.addEventListener("blur", () => logEvent("focus_lost", {}));
    window.addEventListener("focus", () => logEvent("focus_regained", {}));
  }

  // A single, separate requestAnimationFrame loop whose only job is
  // measuring the gap between frames -- deliberately independent of each
  // obstacle's own rAF-driven fall animation (spawnObstacle()'s step()
  // below), so a dropped frame is detected once per actual frame, not once
  // per live obstacle. A big gap means the render thread stalled (GC pause,
  // OS scheduling, etc.) for that long -- worth flagging since it can
  // distort any reaction-time reading taken around the same instant.
  // Debounced by frame_drop_min_gap_sec so one bad patch of frames can't
  // spam the log with dozens of near-identical warnings.
  function installFrameDropWatchdog() {
    const thresholdMs = bootstrap.frame_drop_threshold_ms ?? 50;
    const minGapMs = (bootstrap.frame_drop_min_gap_sec ?? 0.5) * 1000;
    let last = performance.now();
    let lastLoggedAt = -Infinity;
    function tick(now) {
      const delta = now - last;
      last = now;
      if (delta > thresholdMs && now - lastLoggedAt > minGapMs) {
        lastLoggedAt = now;
        logEvent("frame_drop_warning", { delta_ms: delta, threshold_ms: thresholdMs });
      }
      requestAnimationFrame(tick);
    }
    requestAnimationFrame(tick);
  }

  function buildLaneDividers() {
    const road = el("road");
    road.querySelectorAll(".lane-divider").forEach((d) => d.remove());
    for (let i = 1; i < lanes; i++) {
      const divider = document.createElement("div");
      divider.className = "lane-divider";
      divider.style.left = `${(i / lanes) * 100}%`;
      road.appendChild(divider);
    }
  }

  // Shared by baseline and each tier's driving block: resolves when
  // `durationMs` elapses, OR the participant presses B (skip block --
  // mirrors SKIP_BLOCK_KEY in app/ui/common_widgets.py) or Escape (full
  // quit, handled by the caller via the `quitting` flag).
  function waitForBlock(durationMs) {
    return new Promise((resolve) => {
      let skipped = false;
      const timer = setTimeout(finish, durationMs);
      function onKey(e) {
        if (e.key === "Escape") { logTaskAbort("participant_escape"); finish(); }
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

  // Block-level self-report -- runs between driving blocks (never mid-drive
  // -- see attention_highway.py's bootstrap.self_report for the
  // configured items/prompts/scale). Logs prompt onset separately from the
  // response so offline analysis gets both an exact onset timestamp and a
  // response RT per item, same convention app/tasks/emotion_faced.py's
  // rating_response already uses -- this is the SAME event type/shape,
  // just produced by the in-webview widget (app/webui/shared/
  // rating_widget.js) instead of a psychopy screen. The temporal interval
  // being rated is "the block that just ended" -- rather than duplicate its
  // start/end time here, this just tags the response with the block_index/
  // condition_label/tier it followed, joinable against that block's own
  // block_start/block_end rows.
  async function runSelfReport(blockIndex, tier) {
    const cfg = bootstrap.self_report;
    showScreen("rating-screen");
    logEvent("self_report_prompt_onset", {
      block_index: blockIndex, condition_label: tier.label, tier: tier.id,
      items: cfg.items,
    });
    const result = await RatingWidget.run({
      container: el("rating-rows"),
      items: cfg.items,
      prompts: cfg.prompts,
      scaleMin: cfg.scale_min,
      scaleMax: cfg.scale_max,
      isQuitKey: (e) => e.key === "Escape",
    });
    if (result.quit) {
      logTaskAbort("participant_escape_during_self_report");
      logEvent("self_report_response", {
        block_index: blockIndex, condition_label: tier.label, tier: tier.id, skipped: true,
      });
      return;
    }
    const payload = {
      block_index: blockIndex, condition_label: tier.label, tier: tier.id,
      scale_min: cfg.scale_min, scale_max: cfg.scale_max, skipped: false,
    };
    for (const key of cfg.items) {
      payload[key] = result.values[key];
      payload[`${key}_rt`] = result.rts[key] ?? null;
    }
    logEvent("self_report_response", payload);
  }

  async function runDrivingBlock(blockIndex, tier, autoDodge) {
    showScreen("game-screen");
    el("obstacle-field").innerHTML = "";
    liveObstacles.forEach((o) => cancelAnimationFrame(o.rafId));
    liveObstacles.clear();
    updateObstacleEndTopPct(); // road/car are now actually visible and laid out

    const nominalDifficulty = tier.nominal_difficulty ?? null;
    logEvent("block_start", {
      block_index: blockIndex, condition_label: tier.label, tier: tier.id,
      nominal_difficulty: nominalDifficulty,
    });
    let tierAvoided = 0;
    let tierCollisions = 0;
    function onResolved(collided) {
      if (collided) { collisions += 1; tierCollisions += 1; }
      else { avoided += 1; tierAvoided += 1; }
      updateHud();
    }

    const [minGapSec, maxGapSec] = tier.spawn_interval_range_sec;
    const fallMs = tier.fall_duration_sec * 1000;
    let trialIndex = 0;
    let spawnTimer = null;

    // Second difficulty axis (per playtesting feedback 2026-08-05): a wave
    // spawns SEVERAL hazards together, in different lanes, instead of
    // always one at a time -- raises how many lanes demand attention AT
    // ONCE, independent of how fast any single hazard approaches. Still
    // safe: pickSpawnLane()'s guaranteed-clear-lane invariant holds across
    // the whole wave, since each call inside the loop below updates
    // liveObstacles before the next call in the SAME wave runs, so a wave
    // can never fill more than lanes-1 lanes.
    //
    // concurrent_obstacles in the config is a CAP, not a fixed count: every
    // wave always spawning exactly that many read as suspiciously uniform/
    // predictable (participant feedback 2026-08-13), so each wave now rolls
    // its own size, from 1 up to the cap.
    const concurrentObstaclesMax = tier.concurrent_obstacles ?? 1;
    let waveIndex = 0;

    function scheduleNext() {
      const gapSec = minGapSec + Math.random() * (maxGapSec - minGapSec);
      spawnTimer = setTimeout(() => {
        const wave = waveIndex++;
        const waveSize = 1 + Math.floor(Math.random() * concurrentObstaclesMax);
        const lanesUsedThisWave = new Set();
        const picks = [];
        for (let i = 0; i < waveSize; i++) {
          const picked = pickSpawnLane(lanesUsedThisWave);
          // null means pickSpawnLane deliberately skipped this slot (see its
          // comment) rather than force another duplicate -- no obstacle, no
          // trial, for this particular slot.
          if (!picked) continue;
          lanesUsedThisWave.add(picked.lane);
          picks.push(picked);
        }
        // spawn_wave_size passed to spawnObstacle() below is picks.length --
        // the ACTUAL number of hazards this wave produced (after both the
        // random waveSize roll and any pickSpawnLane() skips), not the
        // tier's static cap, so the logged wave size always matches what a
        // participant actually saw on screen.
        picks.forEach((picked) => {
          spawnObstacle(blockIndex, tier.label, trialIndex++, fallMs, autoDodge, onResolved, nominalDifficulty, wave, picks.length, picked);
        });
        scheduleNext();
      }, gapSec * 1000);
    }
    scheduleNext();

    const blockStartMs = performance.now();
    const blockDurationMs = tier.duration_sec * 1000;
    const stateTickTimer = startStateTick(tier, nominalDifficulty, blockStartMs, blockDurationMs);

    const skipped = await waitForBlock(blockDurationMs);
    clearTimeout(spawnTimer);

    // Skipped/quit: resolve whatever's still on screen immediately
    // (silently, no visual/score update). Otherwise let it play out
    // naturally so a hazard spawned right before the deadline isn't just
    // erased mid-air.
    if (skipped || quitting) {
      liveObstacles.forEach((o, id) => resolveObstacle(id, /* silent */ true));
    } else {
      await waitUntil(() => liveObstacles.size === 0 || quitting);
    }
    clearInterval(stateTickTimer);

    logEvent("block_end", {
      block_index: blockIndex, condition_label: tier.label, tier: tier.id,
      nominal_difficulty: nominalDifficulty,
      skipped: skipped || quitting, avoided: tierAvoided, collisions: tierCollisions,
    });
  }

  // Fixed-rate gameplay state stream (default 10 Hz, bootstrap.state_tick_hz
  // -- 0 disables it). Independent of any single obstacle/keypress event:
  // this is what lets offline analysis align a continuous EEG/ECG/EDA
  // recording against "what was the game state at time T" without having
  // to reconstruct it by replaying the discrete event stream. Returns the
  // interval handle so the caller can clearInterval() it at block end.
  function startStateTick(tier, nominalDifficulty, blockStartMs, blockDurationMs) {
    const hz = bootstrap.state_tick_hz ?? 10;
    if (!hz) return null;
    const windowMs = (bootstrap.rolling_window_sec ?? 15) * 1000;

    return setInterval(() => {
      const now = performance.now();

      // Prune rolling history lazily (cheaper than a separate timer, and
      // there are at most a few hazards/keypresses per second to prune).
      while (recentOutcomes.length && now - recentOutcomes[0].t > windowMs) recentOutcomes.shift();
      while (recentReactionTimes.length && now - recentReactionTimes[0].t > windowMs) recentReactionTimes.shift();

      let nearestTtcMs = null;
      let ownLaneTtcMs = null;
      const blockedLanes = new Set();
      for (const obstacle of liveObstacles.values()) {
        const ttcMs = obstacle.fallMs - (now - obstacle.spawnTime);
        blockedLanes.add(obstacle.lane);
        if (nearestTtcMs === null || ttcMs < nearestTtcMs) nearestTtcMs = ttcMs;
        if (obstacle.lane === playerLane && (ownLaneTtcMs === null || ttcMs < ownLaneTtcMs)) ownLaneTtcMs = ttcMs;
      }

      const rollingCollisionRate = recentOutcomes.length
        ? recentOutcomes.filter((o) => o.collided).length / recentOutcomes.length
        : null;
      const rollingAvgRtMs = recentReactionTimes.length
        ? recentReactionTimes.reduce((sum, r) => sum + r.rtMs, 0) / recentReactionTimes.length
        : null;

      logEvent("state_tick", {
        condition_label: tier.label, tier: tier.id, nominal_difficulty: nominalDifficulty,
        player_lane: playerLane, lanes,
        active_obstacle_count: liveObstacles.size, blocked_lane_count: blockedLanes.size,
        nearest_time_to_collision_ms: nearestTtcMs, own_lane_time_to_collision_ms: ownLaneTtcMs,
        fall_duration_sec: tier.fall_duration_sec, spawn_interval_avg_sec: (tier.spawn_interval_range_sec[0] + tier.spawn_interval_range_sec[1]) / 2,
        score_avoided: avoided, score_collisions: collisions,
        rolling_collision_rate: rollingCollisionRate, rolling_avg_reaction_time_ms: rollingAvgRtMs,
        rolling_window_sec: bootstrap.rolling_window_sec ?? 15,
        time_remaining_sec: Math.max(0, (blockStartMs + blockDurationMs - now) / 1000),
        time_since_last_collision_sec: lastCollisionTime === null ? null : (now - lastCollisionTime) / 1000,
        time_since_last_near_miss_sec: lastNearMissTime === null ? null : (now - lastNearMissTime) / 1000,
        // No "target status" field -- this game has no signal-detection
        // target concept (unlike e.g. SART's go/no-go stimulus), so there's
        // nothing meaningful to report there; omitted rather than filled
        // with a misleading placeholder.
      });
    }, 1000 / hz);
  }

  function waitUntil(predicate) {
    return new Promise((resolve) => {
      (function poll() {
        if (predicate()) resolve();
        else setTimeout(poll, 150);
      })();
    });
  }

  // Anti-camping nudge probability: 0 until camp_grace_sec of no lane
  // change has passed, then ramps linearly to camp_max_probability over the
  // following camp_ramp_sec. Absent from bootstrap (e.g. an older
  // bootstrap payload) just means no nudging.
  function campNudgeProbability() {
    if (!bootstrap.camp_grace_sec && !bootstrap.camp_ramp_sec) return 0;
    const timeInLaneSec = (performance.now() - lastLaneChangeTime) / 1000;
    const graceSec = bootstrap.camp_grace_sec || 0;
    if (timeInLaneSec <= graceSec) return 0;
    const rampSec = bootstrap.camp_ramp_sec || 0;
    const rampFrac = rampSec > 0 ? Math.min(1, (timeInLaneSec - graceSec) / rampSec) : 1;
    return rampFrac * (bootstrap.camp_max_probability ?? 1);
  }

  // Guarantees a safe lane is always available: if every lane but one
  // already has a live (unresolved) hazard in it, the new hazard is forced
  // into one of the ALREADY-threatened lanes instead of the one remaining
  // clear lane -- so the count of simultaneously-threatened lanes can never
  // reach `lanes`, and the participant can never be boxed in with no move
  // that avoids every live hazard. The anti-camping nudge is applied FIRST,
  // but only takes effect if the participant's lane isn't already
  // threatened AND at least one OTHER lane would still be left clear -- so
  // nudging can never itself create an unavoidable collision, and never
  // stacks a redundant second hazard onto a lane it already succeeded in
  // threatening.
  // `lanesUsedThisWave` is the set of lanes already assigned to an EARLIER
  // obstacle in the SAME wave (see scheduleNext()) -- distinct from
  // `occupied`, which also includes still-live hazards left over from
  // previous, overlapping waves (spawn_interval_range_sec can be shorter
  // than fall_duration_sec, so waves stack up). Two obstacles landing in the
  // same lane from DIFFERENT waves are visually distinguishable (different
  // depths on screen, staggered arrival) and are a deliberate difficulty
  // knob; two obstacles landing in the same lane from the SAME wave share
  // an identical spawn/fall time and render as one indistinguishable
  // hazard, so those get avoided wherever possible below.
  //
  // Returns null when this wave slot is deliberately skipped rather than
  // spawning anything -- see the "permanently safe lane" fix below.
  function pickSpawnLane(lanesUsedThisWave) {
    const occupied = new Set([...liveObstacles.values()].map((o) => o.lane));
    // Don't nudge into a lane that's already threatened (by an earlier pick
    // in this same wave, or a still-live hazard from a prior wave) --
    // camp_grace_sec/camp_ramp_sec/camp_max_probability are untouched
    // (participant-tuned, per feedback 2026-08-13), but a hazard already
    // sitting in the player's lane means the nudge has already done its
    // job; rolling it again just piles a second, perfectly-overlapping
    // obstacle onto the same lane and double-scores it (same duplicate-
    // scoring bug as the branches below, just via the nudge path -- this
    // was showing up on nearly every wave once concurrent_obstacles>1,
    // since all of a wave's picks land in the same instant with no chance
    // for the participant to move in between).
    const wantsNudge = !occupied.has(playerLane) && Math.random() < campNudgeProbability();
    if (wantsNudge && occupied.size < lanes - 1) return { lane: playerLane, nudged: true };
    if (occupied.size >= lanes - 1) {
      // Every tier spawns hazards faster than they fall (spawn_interval <
      // fall_duration -- the "concurrent hazards" difficulty axis), so
      // occupied.size reaches lanes-1 almost immediately and then NEVER
      // drops again for the rest of the block -- every following wave just
      // keeps refilling the same lanes-1 lanes below via the `pool` pick.
      // Left unchecked, that permanently locks in whichever lane happened
      // to be spared during the very first wave as "the safe lane" -- a
      // participant who finds it and camps there can sit out an entire
      // block untouched, since wantsNudge above can never fire once
      // occupied.size stops dipping below lanes-1 (bug reported by a
      // participant 2026-08-13). If the participant HAS been camping long
      // enough that the nudge wants to fire but can't (every other lane is
      // currently live), skip refilling this slot instead of forcing
      // another duplicate -- letting one of those lanes drain naturally
      // brings occupied.size back below lanes-1 within a wave or two, at
      // which point wantsNudge (still being rolled every pick while camp
      // probability stays elevated) finally lands the hazard in the lane
      // the participant is actually sitting in, instead of the "safe" lane
      // being permanent for the rest of the block.
      if (wantsNudge) return null;
      // Prefer doubling up on a lane from an OLDER, already-live wave over
      // one just claimed earlier in THIS wave -- the former stays visually
      // distinguishable (staggered depth), the latter would be a second,
      // perfectly-overlapping obstacle indistinguishable from the first.
      const candidates = [...occupied].filter((l) => !lanesUsedThisWave.has(l));
      const pool = candidates.length ? candidates : [...occupied];
      return { lane: pool[Math.floor(Math.random() * pool.length)], nudged: false };
    }
    // Pick only among lanes with no live hazard yet -- picking from ALL
    // lanes here (including already-occupied ones) would let two obstacles
    // land in the same lane purely by chance whenever a clear lane was
    // available, silently doubling up what the participant sees as one
    // hazard into two independently-scored avoided/collision trials (bug
    // reported by a participant 2026-08-13). Only the forced branch above
    // (every other lane already taken) is allowed to double up, since that
    // one is unavoidable given the guaranteed-clear-lane invariant.
    const free = [];
    for (let l = 0; l < lanes; l++) if (!occupied.has(l)) free.push(l);
    return { lane: free[Math.floor(Math.random() * free.length)], nudged: false };
  }

  function spawnObstacle(blockIndex, conditionLabel, trialIndex, fallMs, autoDodge, onResolved, nominalDifficulty, spawnWaveIndex, spawnWaveSize, picked) {
    const id = ++obstacleIdCounter;
    const { lane, nudged } = picked;
    const wrap = document.createElement("div");
    wrap.className = "obstacle";
    wrap.textContent = "🚧";
    wrap.style.left = `${laneCenterPct(lane)}%`;
    el("obstacle-field").appendChild(wrap);

    const spawnTime = performance.now();
    const obstacle = {
      el: wrap, lane, spawnTime, fallMs, blockIndex, conditionLabel, trialIndex, onResolved, rafId: null,
      respondedAt: undefined, // set by handleDodge() the moment the participant leaves this obstacle's lane
    };
    liveObstacles.set(id, obstacle);

    // obstacle_spawn/obstacle_resolved together stand in for an explicit
    // "response-window open/close" pair (the window during which a dodge
    // counts as addressing THIS hazard is exactly [spawn, resolve]) --
    // logging separate response_window_open/close events would just
    // duplicate that same interval under a different name.
    logEvent("obstacle_spawn", {
      block_index: blockIndex, trial_index: trialIndex, condition_label: conditionLabel,
      lane, lanes, fall_duration_sec: fallMs / 1000, camp_nudged: nudged,
      nominal_difficulty: nominalDifficulty,
      // Raw wave structure preserved rather than left to be inferred from
      // timestamps -- spawn_wave_size > 1 means this obstacle spawned
      // simultaneously with (spawn_wave_size - 1) others sharing the same
      // spawn_wave_index, see scheduleNext()'s concurrentObstacles.
      spawn_wave_index: spawnWaveIndex, spawn_wave_size: spawnWaveSize,
    });

    if (autoDodge) {
      // Demo mode only: move into whichever lane is clear of every
      // currently-live hazard, partway through this one's approach.
      setTimeout(() => {
        if (!liveObstacles.has(id)) return;
        for (let candidate = 0; candidate < lanes; candidate++) {
          const threatened = [...liveObstacles.values()].some((o) => o.lane === candidate);
          if (!threatened) {
            // Only a REAL move resets the anti-camping clock (setPlayerLane
            // does that whenever animate=true, regardless of whether `lane`
            // actually changed -- fine for the real participant's dodge key,
            // which only ever calls it with a genuinely different lane, but
            // this loop picks the lowest-index clear lane every time even
            // when that's the lane the car is already sitting in). Without
            // this guard the demo car "re-confirms" its own lane on almost
            // every wave, continually resetting the clock so it never
            // actually looks like it's camping -- and the anti-camping
            // nudge (camp_grace_sec/camp_ramp_sec) then never gets a chance
            // to fire, so the demo never shows a hazard spawning directly
            // into the lane the car has genuinely been sitting in (bug
            // reported by the user 2026-08-13, seen as "obstacle patterns
            // look deterministic/predictable" in demo mode).
            if (candidate !== playerLane) setPlayerLane(candidate, true);
            break;
          }
        }
      }, fallMs * 0.4);
    }

    const start = performance.now();
    const startTopPct = -10;
    function step(now) {
      if (!liveObstacles.has(id)) return; // already resolved
      const t = Math.min(1, (now - start) / fallMs);
      wrap.style.top = `${startTopPct + t * (obstacleEndTopPct - startTopPct)}%`;
      if (t >= 1) resolveObstacle(id);
      else obstacle.rafId = requestAnimationFrame(step);
    }
    obstacle.rafId = requestAnimationFrame(step);
  }

  function resolveObstacle(id, silent) {
    const obstacle = liveObstacles.get(id);
    if (!obstacle) return;
    liveObstacles.delete(id);
    cancelAnimationFrame(obstacle.rafId);

    const now = performance.now();
    const collided = obstacle.lane === playerLane;
    const responded = obstacle.respondedAt !== undefined;
    // near_miss/no_response only mean anything when the participant was
    // actually IN this obstacle's lane at some point during its approach
    // (i.e. a response was possible) -- an obstacle that was never in the
    // player's lane at all resolves as a routine avoidance, not a "miss".
    let nearMiss = false;
    let reactionMarginSec = null;
    if (responded) {
      reactionMarginSec = (obstacle.fallMs - (obstacle.respondedAt - obstacle.spawnTime)) / 1000;
      if (!collided && reactionMarginSec <= (bootstrap.near_miss_window_sec ?? 0.3)) nearMiss = true;
    }
    const noResponse = collided && !responded;

    if (collided) lastCollisionTime = now;
    if (nearMiss) lastNearMissTime = now;
    recentOutcomes.push({ t: now, collided });
    if (!silent) {
      obstacle.el.classList.add(collided ? "collided" : "avoided");
      if (collided) flashCollision();
    }
    obstacle.onResolved(collided);

    logEvent("obstacle_resolved", {
      block_index: obstacle.blockIndex, trial_index: obstacle.trialIndex, condition_label: obstacle.conditionLabel,
      lane: obstacle.lane, player_lane: playerLane, outcome: collided ? "collision" : "avoided", skipped: !!silent,
      near_miss: nearMiss, no_response: noResponse, response_attempted: responded, reaction_margin_sec: reactionMarginSec,
    });
    setTimeout(() => obstacle.el.remove(), silent ? 0 : 300);
  }

  function flashCollision() {
    const car = el("player-car");
    car.classList.remove("flash");
    void car.offsetWidth; // restart the animation if it's already mid-flash
    car.classList.add("flash");
  }

  function updateHud() {
    el("score-value").textContent = String(avoided);
    el("collision-value").textContent = String(collisions);
  }

  // Nearest still-falling hazard currently occupying `lane` (soonest to
  // arrive) -- the reaction-time reference for a dodge OUT of that lane.
  // null means the lane being left had no live threat, i.e. this move is
  // unforced (logged as a false-alarm-style extra response, not scored
  // against the participant in any way -- the game itself doesn't judge it).
  function threatInLane(lane) {
    let best = null;
    for (const obstacle of liveObstacles.values()) {
      if (obstacle.lane !== lane) continue;
      if (!best || obstacle.spawnTime < best.spawnTime) best = obstacle;
    }
    return best;
  }

  function setPlayerLane(lane, animate) {
    // animate is true exactly for real moves (player dodge or demo
    // auto-dodge), never for the one-time silent initial placement -- reset
    // the anti-camping clock on those, whether or not `lane` happens to
    // equal the previous one.
    if (animate) lastLaneChangeTime = performance.now();
    playerLane = lane;
    const car = el("player-car");
    if (!animate) car.style.transition = "none";
    car.style.left = `${laneCenterPct(lane)}%`;
    if (!animate) { void car.offsetWidth; car.style.transition = ""; }
  }

  function handleDodge(direction, key) {
    if (el("game-screen").classList.contains("hidden")) return;
    const fromLane = playerLane;
    const desiredLane = fromLane + direction;
    const blocked = desiredLane < 0 || desiredLane >= lanes;
    const toLane = blocked ? fromLane : desiredLane;

    const threat = threatInLane(fromLane);
    const reactionTimeMs = threat ? performance.now() - threat.spawnTime : null;
    if (threat) {
      // Marks THIS specific obstacle as having drawn a response -- read
      // back in resolveObstacle() to classify near_miss/no_response.
      // Deliberately records every attempt (not just the first), since a
      // blocked attempt at a track edge followed by a successful one a
      // moment later is still "responded", not "no_response".
      threat.respondedAt = performance.now();
      recentReactionTimes.push({ t: performance.now(), rtMs: reactionTimeMs });
    }
    if (!blocked) setPlayerLane(toLane, true);

    logEvent("lane_change", {
      key, from_lane: fromLane, to_lane: toLane, blocked,
      forced: !!threat, reaction_time_ms: reactionTimeMs,
      threatening_trial_index: threat ? threat.trialIndex : null,
    });
  }

  function onKeyDown(e) {
    if (e.key === "ArrowLeft" || e.key.toLowerCase() === "a") handleDodge(-1, e.key);
    else if (e.key === "ArrowRight" || e.key.toLowerCase() === "d") handleDodge(1, e.key);
  }

  function showScreen(name) {
    document.querySelectorAll(".screen").forEach((s) => s.classList.add("hidden"));
    el(name).classList.remove("hidden");
  }

  function showEndScreen() {
    showScreen("end-screen");
    el("end-score").textContent = `Avoided ${avoided} / ${avoided + collisions} hazards`;
    document.addEventListener("keydown", function onSpace(e) {
      if (e.key === " " || e.key === "Enter") {
        document.removeEventListener("keydown", onSpace);
        window.pywebview.api.close_window();
      }
    });
  }

  window.addEventListener("pywebviewready", main);
})();
