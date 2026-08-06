// Shared browser<->host clock-correspondence helpers. See
// app/eventlog/event_logger.py's timestamp_monotonic and
// app/webui/bridge.py's sync_checkpoint() for the Python half of this --
// this is the one JS-side implementation every webui task AND the session
// shell (app/webui/shell/shell.js) share, so the fix landed identically
// everywhere instead of each *_app.py frontend growing its own
// slightly-different copy that drifts out of sync with the others over
// time. First built for app/webui/highway/game.js, then generalized here.
(() => {
  function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  // Wraps a task's own logEvent so every event ALSO carries the browser's
  // performance.now() reading automatically -- without this, only events
  // that happen to compute their own *_ms field by hand would have any
  // browser-clock reference at all. Forwards any extra arguments
  // unchanged, e.g. shell.js's logEvent(eventType, fields, task) third
  // parameter -- this wrapper doesn't need to know that shape, it just
  // passes it through.
  function wrapLogEvent(rawLogEvent) {
    return (eventType, fields, ...rest) => {
      rawLogEvent(eventType, Object.assign({ client_perf_now_ms: performance.now() }, fields || {}), ...rest);
    };
  }

  // One JS<->Python round trip, logged with both the browser-side
  // before/after readings and the host-side readings sync_checkpoint()
  // (bridge.py) returned. `logEvent` here only needs to accept
  // (eventType, fields) -- a caller that must route this to a specific
  // task label (shell.js, which spans several) should pass a small
  // wrapper closure instead of its raw logEvent.
  async function doCheckpoint(logEvent, label) {
    const beforeMs = performance.now();
    const result = await window.pywebview.api.sync_checkpoint(beforeMs);
    const afterMs = performance.now();
    logEvent("sync_checkpoint", {
      label,
      client_perf_now_before_ms: beforeMs,
      client_perf_now_after_ms: afterMs,
      round_trip_ms: afterMs - beforeMs,
      host_utc_at_receipt: result.host_utc,
      host_perf_counter_at_receipt: result.host_perf_counter,
    });
  }

  // Several checkpoints (default 3), a short gap apart (default 0.2s), so
  // offline analysis can fit an offset AND a drift rate instead of trusting
  // one single instant. Call once at process/session start AND again at
  // end -- every *_app.py subprocess (and the session shell itself) is its
  // OWN browser engine instance with its own independent performance.now()
  // epoch, so each one needs its own pair of checkpoints; one subprocess's
  // checkpoints don't cover any other's clock.
  async function runCheckpoints(logEvent, labelPrefix, count, gapSec) {
    const n = count ?? 3;
    const gapMs = (gapSec ?? 0.2) * 1000;
    for (let i = 0; i < n; i++) {
      await doCheckpoint(logEvent, `${labelPrefix}_${i}`);
      if (i < n - 1) await sleep(gapMs);
    }
  }

  window.ClockSync = { wrapLogEvent, runCheckpoints };
})();
