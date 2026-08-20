# Dataset Sync & Dataloader Design

Turns one raw participant folder under
`C:\emotion-stress-attention-task-ear-wristband-device-data\data-collection-sessions`
into a canonical, host-UTC-synced set of per-stream Parquet files, plus an
event-windowed PyTorch-compatible dataloader on top. Code lives in
`dataset/`. This doc is the reference for *why* the code does what it does;
keep it in sync with `dataset/overrides.yaml` when a new participant needs
a documented manual fix.

<!-- ## 0. Source scripts (ground truth for everything below)

- Ear-EEG / in-ear-EEG sync + firmware: `ADS1299_BLE_main` (out-ear),
  `ADS1299_BLE-main-gaoteng` (in-ear) -- host side:
  `ADS1299_BLE-main-gaoteng/tools/ble_tool/ble_data_provider.py` (used for
  **both** devices; command `python .\tools\ble_tool\ble_data_provider.py`).
- Wristband firmware + sync: `0705文件/Final_Code.ino`,
  `0705文件/ble_sync_wristbands.py` (command:
  `python .\ble_sync_wristbands.py --cycles 1 --scan-seconds 15 --expected-count 2`).
- Polar H10 capture: `polar-h10-data-collection/polar_h10_capture.py` +
  `pmd_protocol.py` + `clock_sync.py` (command:
  `python .\polar_h10_capture.py --participant xxx`).
- **Note:** `app/sync/*.py` in this repo (biosignal-emotion-stress-attention-protocol)
  is NOT what was actually used to sync real sessions -- confirmed by
  `events.jsonl`'s `device_sync_skipped` event
  (`{"reason": "devices pre-synced/verified via external scripts before this
  app started"}`). The three scripts above are the real ground truth. -->

## 1. Confirmed sampling rates & formats

| Source | Sensor | Rate | Format |
|---|---|---|---|
| `out-ear/A1299_NN.BIN` | ADS1299 EEG, 8ch | 250 Hz | 36B header + 45B/record, magic `0x41443132` |
| `in-ear/A1299_00.bin` | ADS1299 EEG, 8ch | 250 Hz | identical struct layout |
| `in-ear/PPG_00.bin` | MAX30101 red/IR/green | 100 Hz (configured) | "SBN1"-framed: 32B header + 25B/record |
| `in-ear/IMU_00.bin` | BMI270 accel+gyro | 100 Hz | SBN1, 37B/record |
| `in-ear/MLX90632_00.bin` | IR temperature | 1 Hz | SBN1, 23B/record |
| `in-ear/BME680_00.bin` | env (P/H/gas) | 1 Hz | SBN1, 24B/record |
| `wristband/ppg-S*.csv` | MAX30105 red/IR/green | 200 Hz | CSV |
| `wristband/imu-S*.csv` | BMI270 accel+gyro | 200 Hz | CSV |
| `wristband/gsr-S*.csv` | analog EDA | 200 Hz | CSV |
| `wristband/mag-S*.csv` | BMM350 | 100 Hz | CSV |
| `wristband/mlx-S*.csv`, `bme-S*.csv` | IR temp, env | 1 Hz each | CSV |
| `polar-h10/*_polar_ecg.csv` | H10 ECG | 130 Hz (fixed by Polar, not configurable) | CSV |
| `polar-h10/*_polar_acc.csv` | H10 accel | 200 Hz (chosen setting) | CSV |
| `data_collection_logs/events.jsonl` | task/event log | event-driven | JSONL, already host UTC |

<!-- Polar H10 is **not** 250 Hz (confirmed via Polar's own SDK docs, GitHub SDK
issues, and this rig's own `pmd_protocol.py`: `ECG_SAMPLE_RATE_HZ = 130`,
`ACC_SAMPLE_RATE_HZ = 200`). -->

## 2. Sync anchor formulas (per `dataset/sync.py`)

### Ear-EEG / in-ear-EEG (`dataset/raw/ads1299_bin.py`, `sbn1_bin.py`)

One fixed offset per boot, from the `CMD_SET_TIME` handshake
(`ble_data_provider.py`), patched into every open file's header at the
instant it's received:

```
offset  = header.unix_time_at_sync - header.uptime_at_sync
wall_s  = record.timestamp + offset          # ADS1299 file
wall_s  = record.uptime_s + header.time_offset_s   # in-ear PPG/IMU/MLX/BME (offset precomputed)
```

Worked example (`part-P004/out-ear/META.TXT`):
`unix_time_at_sync=1786649693.069262`, `uptime_at_sync=49.611025` →
`offset=1786649643.458237`; a record at `timestamp=52.5` → `wall_s=1786649695.958237`.
Verified against real in-ear data (P007): the ADS file's derived offset
(`1786744409.389398 − 151.920443 = 1786744257.468955`) matches the sibling
PPG/IMU/MLX/BME files' `time_offset_s` **exactly**.

An aborted/restarted boot never receives `CMD_SET_TIME`, so its header's
`unix_time_at_sync` stays `0.0` -- unrecoverable to wall-clock time,
excluded (see §3).

### Wristband (`dataset/raw/wristband_csv.py`)

```
wall_s = meta.computer_epoch_ms/1000 + (device_us - meta.sync_device_us)/1e6
```

Anchor from `meta-S*.csv` (written by `ble_sync_wristbands.py`'s `SYNC_MS`
exchange). Same math applies backward-anchored for the P009 override (§3).

### Polar H10 (`dataset/raw/polar_csv.py`, `dataset/clock_fit.py`)

Not a single fixed offset -- an affine fit (offset + drift rate), refit from
**every** checkpoint in `*_polar_checkpoints.csv` at once:

```
wall_ns = wall_ns0 + rate * (device_ns - device_ns0)     # least-squares over all checkpoints
rate_ppm = (rate - 1) * 1e6                              # device-vs-host drift
```

This is a full-session-hindsight version of what
`polar_h10_capture.py`/`clock_sync.py` does live (causally, improving as
more checkpoints arrive) and what `polar_h10_postprocess.py` does offline --
`dataset/clock_fit.py` is a vectorized adaptation of the same
`LinearClockFit` math, integrated into this pipeline instead of run as a
separate script. Verified on real data (P007): refit vs. the live
`wall_clock_time` column already in the CSV differs by only ~1.5ms mean
(range -54ms..+68ms), confirming the hindsight refit is the expected small
correction, not a bug.

### Events (`dataset/raw/events_jsonl.py`)

`timestamp_host_utc` is already Unix epoch seconds (PC wall clock) -- no
transform. This is the master timeline every device stream above gets
pulled onto.

## 3. Restart / opt-out / low-confidence resolution (`dataset/session_resolver.py`)

**Collection order**: sync all devices → start each device recording →
start the app's own event log **last**. So every device's true sync
instant should sit a bit before the main session's first `events.jsonl`
timestamp ("log-start reference").

- **Ear-EEG**: prefer the file whose header has a nonzero
  `unix_time_at_sync`. If more than one candidate qualifies (shouldn't
  normally happen), prefer whichever's sync instant is closest to (and
  before) the log-start reference. Every excluded candidate is recorded
  with its reason (`ExcludedFile`), not silently dropped.
- **Wristband**: prefer the highest session index, unless
  `dataset/overrides.yaml` pins one for this participant. If the chosen
  session has no forward sync anchor, apply a documented override if one
  exists (see P009 below), else the stream is marked unavailable rather
  than guessed at.
- **Polar H10**: normally one file triplet (`_ecg`/`_acc`/`_checkpoints`);
  same closest-to-log-start rule if more than one is found, using each
  triplet's first checkpoint as its candidate sync instant. P001 has its
  Polar files nested one level deeper (`polar-h10/20260811/...`) --
  resolver checks device dir and one level of subfolders.
- **In-ear onboard sensors** reuse whichever boot index the in-ear ADS1299
  file resolved to -- firmware opens all of a boot's files together
  (confirmed via `meta.txt`'s create-event ordering), so there's no
  independent restart risk to resolve for PPG/IMU/MLX90632/BME680
  separately.
- **Opt-outs / device failures**: detected by absence of `A1299_*.bin` plus
  presence of a marker text file (`*opt*out*`, `*stopped*`, `*declined*`,
  case-insensitive) whose content (if any) is recorded as the reason.
- **Naming-convention drift** (`data_collection_logs` vs
  `data-collection-logs`, `OPTED_OUT` vs `OPTED_out`, upper/lowercase
  `.bin`/`.BIN`) is handled by case-insensitive globbing everywhere, not by
  per-participant overrides -- it's a uniform rule, not a judgment call.
- **Session-subdir naming, 3rd variant (found 2026-08-17, P010-P012)**: most
  participants' session dirs are prefixed with their own folder name
  (`part-P007_<ts>` / `part-P007_sart_<ts>`), but P010-P012 instead prefix
  with the participant's own first name (`zijian_<ts>` /
  `zijian_sart_<ts>`) -- exactly the "3rd naming variant" this doc
  anticipated (§6, old note). `session_resolver.py`'s
  `_find_session_subdirs` now matches structurally instead of hardcoding a
  prefix: any dir ending in `_sart_<digits>` is the SART sub-session; any
  OTHER dir ending in `_<digits>` is a main-session candidate -- the
  original `part-<ID>_...` convention is just a special case of "any
  prefix", so this covers both without a per-participant override. Fixed
  as part of building the npy export (§8) against all 12 real participants;
  before the fix, P010-P012 silently got zero events/windows (`build_dataset`
  "succeeded" with 9 streams written but no `events.parquet` at all --
  `global_warnings: ["no main session directory found"]` was the only
  sign).

### Documented per-participant overrides (`dataset/overrides.yaml`)

- **P006**: wristband session 1 was a short/aborted attempt; forced to
  session 2 per the participant folder's own `notes.txt.txt` ("Use Session
  2 data; that is synced with computer timestamp"). Session 2 has a valid
  forward sync anchor -- normal math applies, just pointed at session 2.
- **P009**: wristband session 2's `meta-S000002.csv` has a header row but
  **zero** data rows -- `SYNC_MS` never reached the device. Per
  `notes_for_wristband.txt.txt`, the wristband was turned off "a few
  moments" after the last task finished, so each modality file's own
  **last** `device_us` row is anchored backward to a reference wall-clock
  instant instead of a forward anchor:

  ```
  offset_s = reference_wall_s - (last_device_us / 1e6)     # per file, not shared
  wall_s   = offset_s + device_us / 1e6
  ```

  Reference: the SART sub-session's `session_end` event --
  `1786759795.203656`, the last logged event overall (~2.5s after
  `task_end`, SART being the last task this participant ran). Single-point,
  no drift correction, unknown slack between `session_end` and the actual
  power-off -- marked `confidence="reduced"` in `sync_report.json`, never
  treated as equivalent to a normally-synced session.

Every override entry in `overrides.yaml` must carry a `reason` -- it's audit
trail, not just config.

### Cross-check: sync-to-log-start gap

Every resolved device anchor gets `gap_to_log_start_s` = `log_start_reference
- anchor_wall_s` computed and sanity-checked (flagged if outside
`[0, 900]` seconds -- devices are synced then the log starts, so this should
normally be positive and at most ~tens of minutes for multi-device prep).
Observed real gaps range ~30s (Polar, synced right before its own capture
script's warm-up) up to ~650s (wristband, when multiple devices are synced
in sequence before the app starts) -- all currently within the threshold;
tune `LOG_START_GAP_WARN_S` in `session_resolver.py` if real data ever shows
a wider legitimate range.

## 4. Canonical output (`dataset/build_dataset.py`)

```
<out_dir>/<participant_id>/
  <device>.<role>.parquet   # wall_utc_s (float64) + native value columns, native rate
  events.parquet            # events.jsonl rows, event_payload_json JSON-encoded as a string
                             # (pyarrow can't infer one Arrow type for a dict column whose
                             # shape varies by event_type -- windows.py re-parses it)
  sync_report.json          # full resolution + sync audit trail: chosen files, excluded
                             # candidates + reasons, anchors, gaps, warnings, per-stream QC
                             # (CRC-8 ok-fraction, Polar drift ppm + checkpoint residuals)
```

Run: `python -m dataset.build_dataset --root <sessions_root> --out-dir <out_dir>`
(or `--participant-dir` for one participant). `--validate-crc` additionally
checks every ADS1299 record's CRC-8 (slower; validated 100% ok on all
sampled real files so far).

## 5. Dataloader (`dataset/windows.py`, `dataset/torch_dataset.py`)

Design decisions (2026-08-17):
- **Rate strategy**: streams stay at native rate in the canonical Parquet;
  resampling to a common target rate happens on the fly, per window, in
  `EventWindowDataset.__getitem__` (linear interpolation onto a fixed grid).
  Nothing is resampled at dataset-build time.
- **Windowing**: event-defined, not fixed sliding windows -- one window per
  task-defined trial/block/baseline/self-report/questionnaire, using each
  task's own event boundaries and labels.
- **Baselines are separate window entries** (`window_type="baseline"`), not
  folded into the following trial window.
- **Every gap is auto-detected**, not just specific named ones: the
  complement of all labeled windows within each recording's span (plus the
  inter-process gap before SART) is emitted as `task="transition"` windows,
  so nothing -- inter-task breaks, subprocess-launch dwell time, anything
  unnamed -- is silently invisible between labeled windows.

All seven tasks/sub-tasks are implemented in `WINDOW_EXTRACTORS`
(`dataset/windows.py`): `emotion`, `stress` (raindrop), `attention_highway`,
`attention_focus_schulte`, `attention_focus_stroop`, `sart`, `preparation`
(questionnaire). `extract_all_windows(events)` runs every one of them plus
gap detection and returns one flat list; `EventWindowDataset` filters that
by `tasks=`/`window_types=` as needed.

Window kinds (`Window.window_type`): `"trial"` (one scored unit -- an
arithmetic problem, an obstacle, a Stroop stimulus, a SART number, an
emotion clip), `"block"` (a whole tier/game-repetition span, which CONTAINS
its trial windows -- e.g. a highway tier block temporally overlaps its ~170
obstacle-trial windows), `"baseline"`, `"self_report"`, `"questionnaire"`,
`"transition"` (auto-detected gap).

**Which one is "a trial"?** For
stress/attention_highway/attention_focus_schulte/attention_focus_stroop/sart,
`"trial"` is usually too fine-grained on its own -- one arithmetic problem
or one obstacle doesn't carry much signal by itself. `"block"` (one
uninterrupted run: one tier, one Schulte/Stroop repetition, SART's single
continuous run) is the atomic unit for those tasks. Emotion is the
exception -- `"trial"` (one clip) already IS the right unit, since a clip
has no finer sub-event to fall back to. `windows.py`'s
`RECOMMENDED_TRIAL_WINDOW_TYPE` dict is the single source of truth per task:
```python
RECOMMENDED_TRIAL_WINDOW_TYPE = {
    "emotion": "trial",
    "stress": "block", "attention_highway": "block",
    "attention_focus_schulte": "block", "attention_focus_stroop": "block",
    "sart": "block",
}
```
The finer `"trial"` windows remain available under every task for
sub-trial-level analysis (e.g. per-obstacle reaction time) -- they're just
not the default "what is a trial" answer.

### Generic pairing mechanism

Every extractor is built on two small helpers rather than one-off code per
task:
- `_pair_events(df, start_type, end_type, key_specs)`: pairs a start/end
  event_type by matching key tuples (e.g. `["block_index", "trial_index"]`)
  when multiple instances can be in flight at once (e.g. a slow raindrop
  response overlapping the next problem's spawn), or plain sequential FIFO
  (`key_specs=None`) when the event stream has already been filtered so
  they can't overlap (e.g. one `condition_label` at a time).
- `_key_value(row, spec)`: reads either a direct DataFrame column or, via
  `"payload:<field>"`, a field buried in `event_payload_json` (needed e.g.
  for emotion's baseline pairing, which must key on `position` since
  `block_index` alone repeats 3x per block for the start/mid/end baselines).

### Correction (2026-08-17): `block_index`/`trial_index` population

An earlier version of this doc/the emotion extractor claimed
`block_index`/`trial_index` are *never* populated by the web-UI tasks'
JS logger (based on a stale demo session, not real collection data). Direct
verification against real participant data corrected this:
- **Emotion, stress (raindrop), attention_highway, attention_focus_stroop**:
  `block_index`/`trial_index`/`condition_label` ARE populated as real
  top-level columns wherever the JS `logEvent()` call includes them
  (`app/webui/bridge.py`'s `WebTaskApi.log_event()` pops these three keys
  out of the JS payload into named `EventLogger.log()` kwargs). They're
  `NaN` only on event types whose call site genuinely omits them (e.g.
  `task_start`/`sync_checkpoint`/`lane_change` for highway) -- a real,
  deliberate per-event-type sparsity, not a systemic gap.
- **SART**: same -- populated directly (native Python `event_logger.log()`
  call, not JS).
- **Attention_focus (Schulte/Stroop) session-level repetition number**:
  genuinely NOT populated anywhere (no field records "which of the 3
  repetitions is this") -- both sub-games run as one subprocess per
  repetition, so `windows.py` derives `session_trial_index` by counting
  `task_start` occurrences per task label, stored in `Window.meta`, not
  `Window.trial_index` (which for Stroop instead holds the real, directly-
  available per-stimulus index 0-19 within one repetition's game block).

### Per-task window specs (verified against real P007 data unless noted)

**Emotion** (`extract_emotion_windows`) -- unchanged from the original design
except pulling `block_index`/`trial_index`/`condition_label` directly off
the paired `trial_start` row instead of deriving them:
```
trial:     [clip_playback_start, clip_playback_end], keyed by payload clip_id
           metadata: trial_start (block/trial/condition/fine_grained_label),
                     rating_response (valence/arousal/liking/emotion_pick + RTs)
baseline:  [baseline_start, baseline_end], keyed by (block_index, payload:position)
           -- 3 per block (start/mid/end touchpoints)
```

**Stress / raindrop** (`extract_stress_windows`, `task=="stress"`):
```
baseline:    [block_start, block_end] where condition_label=="baseline", keyed by block_index
             (one per tier under baseline_mode=per_trial -- 3 total)
transition:  [block_start, block_end] where condition_label=="inter_trial_pause"
             (no block_index at all; sequential-FIFO pairing after filtering)
block:       [block_start, block_end] where condition_label starts with "tier_", keyed by block_index
trial:       [trial_start, response] (one arithmetic problem), keyed by (block_index, trial_index)
```
Known simplification: `wrong_submission` events (keystroke-buffer resets,
no `trial_index`) aren't turned into their own windows.

**Attention_highway** (`extract_attention_highway_windows`,
`task=="attention_highway"` -- **confirmed NOT `"stress"`** despite being
documented as "the second half of the stress task"; a pipeline filtering on
`task=="stress"` alone silently misses this entire sub-task):
```
baseline:      [block_start, block_end] where condition_label=="baseline" (single occurrence)
block:         [block_start, block_end] per tier, keyed by block_index
trial:         [obstacle_spawn, obstacle_resolved], keyed by (block_index, trial_index)
self_report:   [self_report_prompt_onset, self_report_response], one per tier (block_index unique)
```
Known simplification: continuous telemetry (`lane_change`, `state_tick`
@10Hz, `frame_drop_warning`, `focus_lost`/`focus_regained`) has no natural
discrete boundary and isn't turned into windows.

**SART** (`extract_sart_windows`, own separate session/process):
```
baseline:  [baseline_start, baseline_end], position=="pre_task" (single occurrence, ~30s)
block:     [block_start, block_end], keyed by block_index (0=practice, 1=real --
           both real sessions inspected had practice disabled via --skip-practice,
           so only block_index=1 appears in practice; the 0 path is identical code)
trial:     [trial_start, response], keyed by (block_index, trial_index)
```

**Attention_focus: Schulte & Stroop** (`extract_attention_focus_windows`,
`task in {"attention_focus_schulte","attention_focus_stroop"}` -- each
subprocess-per-repetition, 3 repetitions/game):
```
(per repetition, sliced between that repetition's task_start/task_end)
baseline:  [block_start, block_end] where condition_label=="baseline"
block:     [block_start, block_end] where condition_label in {"schulte_trial","stroop_trial"}
trial:     (Stroop only) [trial_start, response], keyed by (block_index, trial_index) -- 20/block
```
Known simplification: Schulte's `cell_click` events (25/trial, no
block_index/trial_index at all) aren't turned into per-cell windows --
available from raw events for anyone wanting finer-than-block granularity.
Real sessions can shuffle the 6 repetitions' order arbitrarily (not
strictly alternating Schulte/Stroop) -- confirmed a valid, not-buggy outcome
of `random.shuffle`.

**Preparation / questionnaire** (`extract_preparation_windows`,
`task=="preparation"`): no question-onset event exists, only the response
(`questionnaire_response`, with a self-timed `rt` field) -- window
`t_end`=response timestamp, `t_start`=`t_end - rt`, giving "time spent
answering" a real synced span from a single event.

### Gap / transition windows (`extract_gap_windows`)

Complement of every already-extracted window within a recording's span
(one span per `session_part` value -- "main"/"sart", written by
`dataset/sync.py`'s `sync_events`), `min_gap_s`-filtered to drop sub-second
scheduling jitter, each tagged with its nearest preceding/following
task+event_type for interpretability. Confirmed real, useful examples from
P007: a 70.18s gap between emotion's last baseline and stress's first block
(the documented `after_task1_min` break), an 82.45s gap between highway's
last self-report and Stroop's first block (the `after_task2_min` break plus
subprocess-launch dwell), and the pre-SART gap (main session's last
`session_end` to SART's first `device_sync_skipped`, confirmed a genuinely
separate-process gap with no bracketing event at all -- 16.46s for P007,
consistent with an operator manually launching
`python -m app.run_task --task sart` a few seconds after closing the main
session window).

### Known data-quality caveat: attention_highway obstacle-scoring bug

`notes/notes.txt` (logged 2026-08-13, fixed same day) documents that
`concurrent_obstacles` was set equal to `lanes` for every tier, which by
pigeonhole forced one lane per wave to receive two simultaneous,
perfectly-overlapping hazards that were then scored as two independent
`obstacle_spawn`/`obstacle_resolved` trials -- inflating trial counts (and
`avoided`/`collisions` tallies) by roughly 30-100% for **the first 4
participants run before the fix**. This shows up directly in window counts:
P001/P004's `attention_highway` extraction yields ~1100+ trial windows vs.
~525 for P007 (post-fix), for the same config and duration -- confirmed via
`extract_all_windows`, not a pipeline bug. Cross-check `notes/notes.txt` for
the exact affected participant list and correction guidance before training
on `attention_highway` trial-level data from early sessions; the
`obstacle_spawn`/`obstacle_resolved` pairing logic here faithfully
reproduces whatever's actually in the log, buggy or not.

One dataset sample (`EventWindowDataset.__getitem__`) =

```python
{
  "participant_id": "P007", "task": "emotion", "window_type": "trial",
  "t_start": ..., "t_end": ...,
  "block_index": 0, "trial_index": 0, "condition_label": "negative",
  "stimulus": {"clip_id": "clip_87", "file_path": "assets/clips/filmstim_negative/NEG_03.mp4", ...},
  "labels": {"valence": 2, "arousal": 1, "liking": 6, "emotion_pick": "Negative", ...},
  "meta": {},
  "biosignals": {
      "ear_eeg_out.ads1299": {"t_rel_s": [...], "values": [[8ch...], ...], "columns": [...]},
      "wristband.ppg": {...}, "polar_h10.ecg": {...}, ...   # only streams with >=1 sample in-window
  },
}
```

`EventWindowDataset(processed_dir, participant_ids, tasks=None, window_types=None, ...)`
-- `tasks=None`/`window_types=None` means every task/kind, including
`"transition"` gaps; pass e.g. `tasks="emotion"` or
`window_types=["baseline"]` to narrow it. `torch` is not required to import
`dataset/torch_dataset.py` or build/inspect windows -- only
`__getitem__(..., to_tensor=True)` touches it, imported lazily at that call
site.

Validated end-to-end (2026-08-17) across all 9 real participants: 10,103
total windows via `EventWindowDataset` with no crashes; every per-task
window count cross-checked exactly against the real event-type counts each
exploration pass reported (e.g. emotion 10 trials/9 baselines, stress 74
trials, highway 518 trials [P007]/~1100 [pre-fix participants, see caveat
above], SART 135 trials, Stroop 60 trials/block, Schulte 3 baselines+3
blocks, preparation 13 questionnaire items).

## 6. Known data-quality caveats (not pipeline bugs, but worth knowing)

- Some ear-EEG channels frequently rail at ±8388608 (the ADS1299's 24-bit
  saturation value) -- electrode contact/lead-off, not a parsing artifact.
- P007's in-ear PPG (`PPG_00.bin`) only has 2878 records over a ~2963s
  session (nominal 100 Hz would be ~296,300) -- the sensor was producing
  data far below its configured rate for this participant; same pattern
  likely worth checking across other in-ear sessions before training on it.
- Polar checkpoint-fit residuals run ~20ms mean / ~70-95ms max on the
  sessions checked so far -- within the expected range per
  `clock_sync.py`'s own docs, but `sync_report.json`'s `stream_qc` carries
  these numbers per participant so a future outlier session is easy to spot.
- P001 predates the `data_collection_logs`/opt-out-marker naming convention
  used from P004 onward; the resolver handles both. The anticipated "3rd
  naming variant" did in fact show up at P010-P012 -- see §3's
  session-subdir-naming note above for the fix.

## 7. Environment

Tested against conda env `data_collection` (Python 3.10.20) -- `numpy`,
`pandas`, `pyarrow`, `scipy`, `pyyaml` all present. `torch` is not installed
anywhere yet; only needed for `to_tensor=True` / actually wrapping
`EventWindowDataset` in a `torch.utils.data.DataLoader`. All modules use
`from __future__ import annotations` so 3.10-style `X | None` type hints
don't break on 3.9 too. Validated end-to-end
(`python -m dataset.build_dataset --root ... --out-dir ...`) against all 12
real participants (P001-P012) with zero crashes.

## 8. Fixed-shape `.npy` export (`dataset/export_npy.py`)

**Superseded 2026-08-20:** the segmenting design below (fixed segment count
`T` per file key, computed from each key's *median* real duration, `X.shape
== (B, C, T, samples_per_segment)`) has been replaced with fixed-length
**epochs** instead: every window is cut into `floor(duration_s / epoch_s)`
non-overlapping `epoch_s`-length epochs (default 1.0s, per-key
`--epoch-seconds` override), any leftover shorter than one epoch dropped
(never padded), and every epoch becomes its own row -- `X.shape == (N, C,
samples_per_epoch)` with `N` = total epochs (not windows/clips). This
means rows from clips of different real durations concatenate directly, no
padding/truncation needed. The label maps, circularity discussion, and
per-task window specs below are all still accurate; only the
segment-count/`T`/`DEFAULT_SEGMENTS` mechanics described in this section
and §9 are stale -- see `dataset/export_npy.py`'s module docstring and
README's "Fixed-shape `.npy` export" section for the current mechanics.

Built 2026-08-17 for a downstream collaborator whose own DL/ML pipeline
(benchmarked against datasets like Mumtaz2016, FACED, MentalArithmetic,
PhysioNet-MI) expects each task as its own ready-to-load
`{X.npy: (B, C, T, 200), y.npy: (B,)}` pair, rather than the variable-
length per-window dicts `EventWindowDataset` returns. Sits on top of
`dataset/windows.py`/`dataset/torch_dataset.py` -- no changes to those.

**File layout** (`<export_dir>/`):
```
<participant_id>/<key>_X.npy       # (B, 8, T, 200) float32
<participant_id>/<key>_y.npy       # (B,) int64
<participant_id>/<key>_meta.csv    # B rows, same order as X/y -- every raw
                                   # participant-response field (see below)
pooled/<key>_X.npy                 # every participant concatenated
pooled/<key>_y.npy
pooled/<key>_participant_ids.npy   # (B,) which participant each pooled row came from
pooled/<key>_meta.csv
export_manifest.json               # params used, per-participant/pooled counts,
                                    # dropped-window reasons, label maps -- audit trail
```
Twelve file keys (`{emotion,math,highway,stroop,schulte,sart}_{trial,baseline}`
-- "math" = the `stress`/raindrop mental-arithmetic task, named for the
collaborator's own terminology). Every key's rows are independently
re-derivable: `<key>_meta.csv`'s `t_start`/`t_end` + the participant's own
`ear_eeg_out.ads1299.parquet` are enough to reproduce `X` exactly (see
verification below) -- **read `t_start`/`t_end` back with
`pd.read_csv(..., float_precision="round_trip")`**, not the pandas default,
which silently loses the last bit of a ~1.79e9 UTC timestamp's precision
(confirmed: `xstrtod`, pandas' default fast C-engine float parser, not
`strtod`) -- close enough to look fine but not bit-exact.

**EEG source**: `ear_eeg_out.ads1299` (the mandatory device, 8 channels,
native 250Hz) -- present for all 12 participants, unlike the optional
in-ear device. `--eeg-stream ear_eeg_in.ads1299` can be substituted.

**Segmenting** (the `T` dimension): each window's own `[t_start, t_end]`
span is divided into a fixed `T` equal-width segments, each linearly
interpolated onto its own 200-point grid (same on-the-fly-resample
philosophy as `torch_dataset.py`'s `_slice_and_resample`, just reshaped
into `(T, 200)` instead of one flat time axis) -- so real seconds/segment
varies by window, but every window in one file still stacks into one
rectangular array. `T` is a free per-file-key parameter
(`--segments key=N,...`); `DEFAULT_SEGMENTS` was chosen 2026-08-17 as
`round(median real duration in seconds)` per key, computed from all 12
real participants (so one segment ~= 1 real second, i.e. ~200Hz-equivalent
resampling by default, not blindly over/under-sampled):

| key | median real duration | default T |
|---|---|---|
| emotion_trial | 67.5s | 68 |
| emotion_baseline | 15.0s | 15 |
| math_trial (block) | 60.0s | 60 |
| math_baseline | 10.0s | 10 |
| highway_trial (block) | 61.1s | 61 |
| highway_baseline | 5.0s | 5 |
| stroop_trial | 1.07s | 1 |
| stroop_baseline | 10.0s | 10 |
| schulte_trial (block) | 25.9s | 26 |
| schulte_baseline | 10.0s | 10 |
| sart_trial | 1.16s | 1 |
| sart_baseline | 30.0s | 30 |

`stroop_trial`/`sart_trial` use `window_type="trial"` (not the
`RECOMMENDED_TRIAL_WINDOW_TYPE` `"block"`) -- their natural classification
label (congruency, go/no-go) only exists at the per-stimulus level; SART's
`"block"` is also just one instance per participant (the whole run), so it
carries no class variation to label at all.

**Labels** -- each task's own natural experimental condition, encoded as
small integers with **0 reserved for baseline/rest in every file** (so
concatenating a task's `_baseline` `y` with its `_trial` `y` reproduces the
rest(0)/condition(1..) convention several of the reference benchmark
datasets use):
```
emotion_trial   1/2/3 = negative/neutral/positive   (valence group, FACED-style)
math_trial      1/2/3 = tier_1/tier_2/tier_3         (raindrop mental-arithmetic difficulty)
highway_trial   1/2/3 = tier_1/tier_2/tier_3         (highway difficulty)
stroop_trial    1/2   = incongruent/congruent        (classic Stroop label)
schulte_trial   1/2   = slower/faster than the cohort-wide median completion
                        time -- Schulte has no natural difficulty condition,
                        so this is a performance-based split (Mumtaz2016
                        MentalArithmetic "good/bad counter"-style), using ONE
                        cohort-wide median threshold, not a per-participant one
sart_trial      1/2   = go/no_go (digit is/isn't a designated no-go number)
*_baseline      0     = baseline/rest (every task)
```
The raw fields behind every label (valence/arousal/rt/accuracy/tier/
completion_time/etc.) are never discarded -- `<key>_meta.csv` carries the
original `stimulus`/`labels`/`meta` dicts as JSON-string columns alongside
`label`, so a caller who wants the richer original response data has it.

**Verification** (`dataset/sanity_check_npy.py`, does NOT trust
`export_manifest.json`'s own bookkeeping):
1. Shape/dtype checks (`X.shape==(B,8,T,200)` float32, `y.shape==(B,)` int64).
2. Independent re-extraction: reruns `extract_all_windows` fresh and
   confirms the window count matches what was recorded at export time.
3. **Recompute-and-compare**: rebuilds a sample of rows straight from the
   canonical Parquet again and asserts bit-exact match against the stored
   `.npy` row -- the strongest alignment check, since it repeats the real
   computation instead of inspecting metadata.
4. NaN reporting, pooled-vs-per-participant total cross-check.

Validated end-to-end 2026-08-17 across all 12 real participants
(P001-P012): 0 dropped windows (every matched window had both a resolvable
label and >=1 in-window EEG sample), 474 recomputed sample rows all
bit-exact, 0 warnings. P001 has 0 SART-baseline windows (predates that
event pair being logged) -- an empty-but-correctly-shaped file, not an
error.

## 9. Attentional-lapse labels + multi-stream export (added 2026-08-18)

Two extensions on top of §8, prompted by a direct question about whether
the Stroop/Schulte/SART labels risk circular reasoning or reverse causation
when used to claim something about "attention" specifically (as opposed to
just task condition).

### The circularity/reverse-causation problem

If a label is derived from the same response whose neural correlate you're
trying to decode, and the EEG epoch used to predict it *includes* that
response (button press, its motor preparation, the error-related brain
activity that follows a mistake), a classifier can hit high accuracy by
picking up motor/error-related signal rather than anything about attention
-- it looks like "EEG predicts attention" but is really "EEG contains a
copy of the label." Related: reverse causation, where the outcome (an
error) *causes* a neural signature (the error-related negativity, which
fires AFTER the mistake) rather than an attention lapse causing the error.

Where our existing §8 labels stand on this:
- `math_trial`/`highway_trial`/`stroop_trial` tier/congruency labels: SAFE
  -- assigned by experimental design before the trial even renders,
  independent of the participant's response entirely.
- `sart_trial` go/no_go: SAFE in the same way (labels *stimulus identity*,
  not attention), but it answers "what kind of stimulus was this," NOT "was
  the participant attending" -- don't conflate the two claims.
- `schulte_trial`'s median-split: the SHARPEST circularity risk of any
  label here -- `completion_time_sec` (what the label is based on) IS
  `t_end - t_start` for that window, i.e. the label is a direct function of
  the window's own duration. Resampling to a fixed T=26 segments means the
  model never sees the literal duration number as a feature, but subtle
  "how much real time got compressed into this grid" correlates could
  still leak in. Not changed (wasn't asked to be); `misclicks` in the same
  `labels_json` is a cleaner duration-independent alternative if this is
  ever revisited.

### New: `sart_lapse_trial` / `stroop_lapse_trial`

Genuine behavior-derived "attention lapse" labels, built the way the SART
literature actually defines one (Robertson et al., 1997, *"...the sound of
one hand clapping"*: an attentional lapse = a commission error, responding
on a trial you should have withheld on) -- with the epoch truncated to end
BEFORE the response so the label can't just be reading its own motor/error
signature.

**A mechanical detail that changes how the truncation has to work, found by
checking real event timestamps against the `rt`/`rt_ms` fields (P007):**
SART's `response` event fires at a ~1.157-1.16s FIXED trial-slot boundary
regardless of when the actual keypress happened (checked across `rt` from
0.41s to 1.06s -- the gap from `trial_start` to the `response` event barely
moves). The real keypress instant is `trial_start + rt`, NOT the event's
own timestamp. Stroop is the opposite -- its `response` event timestamp
matches `rt_ms` to within measurement noise, i.e. it IS the real click
instant. Both need truncation before their response, but from different
reference points.

```
sart_lapse_trial (dataset/export_npy.py's _build_sart_lapse_windows):
  scope: no-go trials only (stimulus.is_omit==True) -- Robertson et al.'s
         commission-error definition is specifically about no-go trials
  t_end: responded  -> trial_start + rt - 0.15s        (truncate before the real keypress)
         !responded -> original t_end (no motor event occurred at all)
  label: 1 = lapse (commission error), 0 = attentive (correct withhold)
  dropped if rt - 0.15s < 0.2s (responded too fast to leave a usable window)

stroop_lapse_trial (_build_stroop_lapse_windows):
  scope: every trial (any incongruent OR congruent trial can be an error)
  t_end: trial_start + rt_ms/1000 - 0.15s               (truncate before the click)
  label: 1 = lapse (incorrect response), 0 = attentive (correct response)
  dropped if rt_ms/1000 - 0.15s < 0.2s
```
0.15s buffer: excludes premotor readiness-potential + the keypress itself
(conservative; movement-preparation onset is commonly reported starting
~200ms+ before an overt response in the motor-EEG literature). No baseline
sibling for these two -- they're inherently binary attentive/lapse, not a
task-vs-rest contrast; use the existing `sart_baseline`/`stroop_baseline`
(same task) as the rest reference if one is needed.

Real cohort numbers (all 12 participants, primary EEG stream): `sart_lapse_trial`
406 no-go trials, 77 commission errors (~19%) -- lines up with typical
SART lapse rates in the literature. `stroop_lapse_trial` 708 trials, only
12 errors (~1.7%) -- this task was easy enough for these participants that
error-based Stroop lapses will be a very imbalanced label; congruency
(`stroop_trial`) remains the better-populated Stroop label if class balance
matters more than lapse-specificity.

### Multi-stream export (wristband added alongside EEG)

`export_all` now takes a `streams=` list (default: the primary EEG stream
+ every wristband modality: `ppg`, `imu`, `gsr`, `mag`, `mlx`, `bme`) and a
per-stream `samples_per_segment` dict, instead of hardcoding EEG's 200
samples/segment for everything. Wristband modalities have very different
native rates (§1: ppg/imu/gsr @200Hz, mag @100Hz, mlx/bme @1Hz) -- forcing
all of them through 200 samples/segment would either fabricate data
(upsampling 1Hz temperature/env sensors) or pointlessly oversample, so each
stream's default `samples_per_segment` is `round(its own native Hz)`:
sensible resampling out of the box, still fully overridable
(`--samples-per-segment wristband.gsr=50`, etc.) if a specific target rate
is wanted instead. `T` (segment count) stays a property of the file key
(task duration) and is shared across every stream exported for that key --
only the per-segment sample count differs by stream.

File naming: the primary EEG stream keeps its original unsuffixed
`<key>_X.npy` (backward compatible with §8); every other stream gets a
short suffix, `<key>_<stream>_X.npy` (e.g. `emotion_trial_ppg_X.npy`,
`schulte_trial_gsr_y.npy`). Each stream is fully self-contained (its own
X/y/meta.csv, meta.csv regenerated per stream) since a window can have
usable EEG but a wristband gap or vice versa -- row order/count can
legitimately differ stream-to-stream for the same file key, so don't
assume row `i` in `emotion_trial_ppg_X.npy` is the same window as row `i`
in `emotion_trial_X.npy` without checking their meta.csv's `t_start`.

Validated end-to-end 2026-08-18 across all 12 participants x 14 file keys
(the original 12 from §8 plus the 2 new lapse keys) x 7 streams: 3913
independently recomputed sample rows, all bit-exact, 0 warnings, 0 errors.
1Hz streams (mlx/bme) and the lapse windows' shorter/truncated spans show
lower per-stream row counts than the EEG version of the same file (e.g.
`sart_lapse_trial_bme` 314 rows vs `sart_lapse_trial` 406) -- expected, not
a bug: a 1Hz sensor can land zero samples inside a short/truncated window
purely from sampling-grid misalignment, tracked per-stream in
`export_manifest.json`'s `dropped.no_signal_samples_in_window`, never
silently reconciled against the EEG count.
