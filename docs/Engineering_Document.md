# Engineering Document — Multimodal Emotion/Stress/Attention Data Collection Protocol & Software

Status: Draft v1 — for review before implementation begins.
Owners: [Muhammad Muaz], with input from Lili (PI) and Yinan (postdoc).

---

## 1. Purpose & Scope

Build a single-computer application that runs a ~60-minute, three-task data collection session with a human participant, while a set of wearable biosensors record locally to their own SD storage. The application's job is to:

1. Present all stimuli/tasks (attention, emotion, stress) with precise, logged timing.
2. Perform BLE time-synchronization with each wearable at session start so that on-device recordings can later be mapped to wall-clock time.
3. Log every experimental event (trial onsets/offsets, condition labels, stimulus identity, participant responses, self-report ratings) to a structured, timestamped event log.
4. Optionally stream/record from a 32-channel Emotiv Flex EEG cap in parallel, for the subset of participants who wear it.
5. Produce a self-contained, versioned "session package" (event log + config + metadata) that a downstream offline pipeline uses to slice and label the SD-card recordings and the Emotiv stream for DL model training.

This document does not cover the offline alignment/labeling pipeline in detail (post-session), only enough to make sure the collection app emits what that pipeline will need.

---

## 2. Devices & Data Paths

| Device | Signals | Storage during session | Onboard clock? | Integration path |
|---|---|---|---|---|
| Custom out-of-ear EEG (8-ch) | EEG | On-device SD card | No (relative counter only) | BLE: one-time clock sync at session start; event log correlated post-hoc |
| Custom in-ear EEG | EEG | On-device SD card | No | Same as above |
| Wristband | PPG, accelerometer, temperature, GSR | On-device SD card | No | Same as above |
| Emotiv Flex (32-ch EEG cap) | Full-scalp EEG | TBD — see §7 open items | Has own clock/streaming stack | LSL (Lab Streaming Layer) preferred if supported, else EmotivPRO/Cortex API recording — **needs verification (open item)** |
| Host PC | — | Event log, config, session metadata | System clock = master clock for the whole session | Runs the collection app |

Key constraint driving the whole sync design: **the SD-card devices have no onboard real-time clock.** They only know elapsed time since power-on (a relative sample counter). The host PC's clock is therefore the single source of truth for wall-clock time, and every device's data must be mapped back to it via an offset computed at sync time.

---

## 3. Session Protocol

Timeline is based on Lili's 60-minute structure (task *content* adapted from Yinan's docs and the paper survey where indicated), extended by 3 minutes to add an explicit familiarization walkthrough (§3.2) — flagging this deviation from the original 60-min figure for PI sign-off; it can be compressed back into the existing Preparation block instead if the extra 3 minutes isn't acceptable.

| Phase | Duration | Notes |
|---|---|---|
| Preparation (consent, questionnaire, sensor fitting, **BLE clock sync**) | 10 min | Sync happens here, once, right before recording starts on each device |
| Familiarization walkthrough | 3 min | See §3.2 |
| Task 1: Attention (spatial n-back) | 15 min | See §4.1 |
| Break | 3 min | |
| Task 2: Emotion (FilmStim clips) | 20 min | See §4.2 |
| Break | 2 min | |
| Task 3: Stress (tiered mental arithmetic) | 8 min | See §4.3 |
| Debrief | 2 min | |
| **Total** | **63 min** | |

Task order (Attention → Emotion → Stress) is fixed, matching both source docs.

### 3.1 Balanced trial design (per participant)

| Task | Classes | Trials/class | Total trials |
|---|---|---|---|
| Attention | 0-back / 1-back / 2-back | 3 | 9 |
| Emotion | Positive / Negative / Neutral | 3 | 9 |
| Stress | Tier-1 / Tier-2 / Tier-3 (each paired with its own baseline) | 1 baseline + 1 arithmetic per tier | 3 baseline + 3 arithmetic = 6 |
| **Total** | — | — | **24** |

### 3.2 Familiarization Walkthrough

Runs once, after sensors are fitted and BLE-synced, before any real (recorded/scored) trial of any task. Purpose: make sure the participant understands the environment and response mechanics before it matters, generalizing the "demo trial" idea from the MAT paper to the whole session rather than just the stress task.

Content (untimed data, not part of the 24-trial dataset):
1. Comfort/contact check — confirm electrodes/wristband feel secure, headphones fit, screen is clearly visible.
2. One non-scored n-back trial (a few flashes, one button press) so the response mechanic is understood.
3. One non-scored sample arithmetic question shown with the real UI (countdown timer, "hurry up" flash, dummy leaderboard visible) so the participant knows what those elements mean before Task 3 starts for real.
4. A preview of the video + rating screen layout using a placeholder/neutral clip, just long enough to walk through what the rating scale looks like (see §3.3/§4.2) — not a full clip viewing.

### 3.3 Preparation Phase Content: Consent & Questionnaire

**Consent screen must explicitly state** (per your direction that participants should know exactly what they're in for):
- Study purpose: collecting biosignals to study emotion, stress, and attention/focus states for machine learning research.
- What will happen, in plain language: (1) an attention/memory grid game, (2) watching short video clips and rating your emotional reaction to each, (3) a timed mental arithmetic task with a countdown timer and leaderboard designed to be mildly stressful.
- Sensors worn: two ear-EEG devices (in-ear and around-ear), a wristband (heart rate/PPG, motion, skin temperature, skin conductance), and — for some participants — a 32-channel EEG cap (Emotiv Flex). All non-invasive.
- Total duration (~63 min per §3).
- Data collected: physiological signals, task responses/accuracy, self-report ratings. No audio/video recording of the participant unless a separate consent line is added for that.
- Foreseeable discomfort: mild sensation from electrode gel/wristband strap; some negative-valence clips may evoke sadness/disgust/fear; the arithmetic task is deliberately time-pressured and may feel stressful.
- Voluntary participation, right to withdraw at any time without penalty, right to skip any question.
- Data storage/anonymization: participant assigned an ID, no directly identifying information stored alongside biosignal data.
- Contact info for PI/IRB questions; signature/checkbox affirmative consent.

**Pre-experiment questionnaire — core battery (target ~2-3 min, to fit inside the 10-min Preparation block):**
- Demographics: age, sex/gender, handedness, highest education level, native language.
- Sensory: normal/corrected-to-normal vision, color vision deficiency, normal hearing (relevant given audio cues and instrumental music in Task 2).
- Health/exclusion screening: diagnosed neurological or psychiatric condition, current CNS-affecting medication, any skin condition/allergy relevant to electrode or wristband contact sites.
- State covariates known to confound EEG/GSR/HR baselines: hours of sleep last night, caffeine in the last 3 hours, nicotine/alcohol in the last 12 hours, time since last meal, current self-rated stress (0–7, same scale as the in-task ratings for consistency) and mood/energy (0–7).

**Optional extended battery** (config-toggle, adds ~3-5 min — only include if Lili/Yinan want trait-level covariates; matches precedent from the cited datasets):
- STAI-Trait short form (trait anxiety) — used in WESAD.
- BFI-10 (Big Five, 10-item short form) — used in MEMA.
- PANAS (state affect), administered pre-session and again at debrief for a pre/post comparison — used in WESAD.

This is a first-pass draft, not finalized — needs a review pass from Lili/Yinan, and confirmation of any IRB-mandated consent language before use.

---

## 4. Task Designs

### 4.1 Task 1 — Attention: Spatial N-Back

**Rationale:** avoids reusing video stimuli (which would overload/confound with Task 2's video-based emotion induction), and fuses sustained attention with a light working-memory component without becoming memory-dominant.

**Design:**
- Stimulus: one cell in a 3×3 grid highlights per trial.
- Participant presses a button if the current position matches the position from *n* trials back.
- Conditions: **0-back** (react to a fixed target position — vigilance/near-zero memory load), **1-back**, **2-back**. Capped at 2-back deliberately — do not extend to 3-back, which shifts the task from attention-dominant to memory-dominant.
- SOA (stimulus onset asynchrony): ~2.5 s (deliberately unhurried; this is not a speed task).
- Block length: 60 s of continuous trials (~24 trials/block at 2.5 s SOA).

**Per-trial/block structure** (mirrors the cue→task→rest pattern from Yinan's doc):

```
Cue (5s, auditory) → N-back block (60s) → Rest (15s)
```

9 blocks (3 per condition, condition order randomized per participant) × 80s = 720s = 12 min, leaving ~3 min of the 15-min budget for instructions + one practice block (untimed, not recorded as data) before the formal session.

**Target accuracy bands (pilot-tunable, not hardcoded):** 0-back ≥95%, 1-back ~85%, 2-back ~70–75%. If pilot 2-back accuracy falls well below this band, reduce difficulty (slower SOA and/or fewer distractor positions) rather than accept memory-overload confounds.

**Logged per trial:** condition (n-back level), stimulus position, is-target (bool), participant response, RT, correct/incorrect, trial onset/offset timestamps (host clock).

### 4.2 Task 2 — Emotion: FilmStim Clips

**Design (per your direction):**
- Source: FilmStim dataset clips (already downloaded on a separate machine — see open items). Fine-grained categories available (e.g., amusement, tenderness, joy, inspiration under positive; anger, disgust, fear, sadness under negative; neutral).
- **Blocking:** clips grouped by valence block (positive block / negative block / neutral block), following FACED's finding that same-valence clips should be shown consecutively. Order of the three blocks is randomized per participant. Order of clips *within* each block is shuffled per participant.
- 3 clips per valence category (9 total), 1–2 min each.
- **Labeling:** each clip carries both a fine-grained emotion label (for future finer-grained analysis) and a coarse 3-class group label (positive/negative/neutral) used as the primary DL target — both stored in the stimulus manifest, not hardcoded in task logic.
- **Post-clip self-report:** a rating screen after each clip, in two parts:
  1. Discrete emotion pick — single choice from that block's fine-grained category list (e.g., for the positive block: amusement / tenderness / inspiration / joy / other-none).
  2. Continuous ratings on a **0–7 integer scale** (per your direction): **valence**, **arousal**, **liking**. Presented as a row of 8 discrete numbered buttons rather than a drag slider — faster and less ambiguous to log under a time budget. **Dominance** (the third axis of the classic PAD/VAD affect model, also collected in DEAP-style datasets alongside valence/arousal/liking) is included as a config-toggleable 4th item — off by default to keep the rating screen quick, easy to turn on if Lili/Yinan want it.
  - This is a UI module driven by config, so the scale, item set, and item order can all change without touching the task engine.

**Per-trial structure (timing tuned to fit the 20-min budget with the 2-part rating screen):**
```
Fixation cross (5s) → Video clip (~90s avg, 1-2min range) → Self-report rating (~20s) → Rest (20s)
```
9 trials × ~135s ≈ 20.25 min — matches the Task 2 budget with the fuller rating screen included; clip-length/rest defaults are config values, not hardcoded, so this can be retuned once real clip durations are known.

**Stimulus manifest:** a config file (JSON/CSV) mapping `clip_id → file_path, fine_grained_label, valence_group, duration`. The app ships with placeholder clips (solid color / stock filler video) referenced by the same manifest schema; swapping in the real FilmStim files later is a config change, not a code change.

**Logged per trial:** clip_id, valence_group, fine_grained_label, block index, clip onset/offset timestamps, discrete emotion pick, valence/arousal/liking (and dominance if enabled) scores (0–7), rating onset/offset timestamps.

### 4.3 Task 3 — Stress: Tiered Mental Arithmetic (MAT-style)

**Design:** Lili's repeated baseline/arithmetic trial structure, with MAT-paper-style UI stress amplifiers, and difficulty organized into 3 hidden tiers instead of 3 repeats of the same difficulty.

```
Baseline 1min (eyes closed/quiet) → Arithmetic Tier-1 1min
Baseline 1min (eyes closed/quiet) → Arithmetic Tier-2 1min
Baseline 1min (eyes closed/quiet) → Arithmetic Tier-3 1min
```
6 blocks, 8 min total, matching Lili's budget exactly.

**Tier definitions** (pilot-tunable via config — operation count and time window are the primary levers, not raw arithmetic complexity, since the MAT paper found time pressure/novelty drove HR more than difficulty):

| Tier | Operations | Numbers | Time/question | Target accuracy |
|---|---|---|---|---|
| 1 (easy) | +, − | 2 | ~5–6 s | ~85–90% |
| 2 (medium) | +, −, × | 3 | ~4–5 s | ~65–75% |
| 3 (hard) | +, −, ×, ÷ | 3–4 | ~3–4 s | ~45–55% |

**UI stress amplifiers (all present from Tier 1 onward, per MAT paper design):**
- Visible countdown timer per question.
- "Hurry up!" blinking cue triggered at 50% of remaining time.
- Dummy leaderboard (static fake top-5 scores) visible during arithmetic blocks.
- Response via on-screen button/numeric input per question (not spoken aloud) — avoids speech-artifact contamination of EEG while still giving objective per-question accuracy/RT.

**Tier is never shown to the participant** — logged only on the backend. This preserves the option, at analysis time, to either (a) collapse all three arithmetic tiers into one "stress" class vs. baseline "no-stress" (binary), or (b) treat tiers as ordinal stress-intensity levels, without needing to re-collect data.

**Logged per block:** tier (baseline blocks logged as tier=null/"baseline"), per-question stimulus (operands, operators), participant answer, correct/incorrect, RT, block onset/offset timestamps.

---

## 5. Synchronization Design

### 5.1 Principle

Host PC's system clock is the master clock for the entire session. Every SD-card device is time-synced to it **once, at session start, before recording begins** (v1 scope — see §5.3 for the planned extension point). The application's own event log is written directly against the host clock, so no separate sync step is needed for the event log itself.

### 5.2 Sync procedure (v1: single sync at start)

1. Participant's devices are powered on and BLE-paired to the host PC.
2. For each device, the app sends the host's current UTC timestamp over BLE at connection time; the device stores this as its reference offset (`device_offset = host_time_at_sync - device_relative_clock_at_sync`).
3. The app records, per device, in the session metadata: device ID, sync timestamp (host clock), and any device-reported confirmation/ack.
4. Recording is started on each device (per device's own start mechanism) immediately after sync; the app logs the "recording start" event per device with a host timestamp.
5. Post-session, offline processing reconstructs each device's sample-level wall-clock time as `wall_time = sync_timestamp + (sample_relative_time)`, i.e., a single fixed offset applied across the whole session (no drift correction in v1).

### 5.2.1 Concrete device protocols (as implemented in the actual firmware repos)

Two genuinely different BLE sync protocols exist across the device families. The app's `sync` module needs one adapter per protocol, both implementing the same `DeviceSyncManager.sync(device) -> SyncRecord` interface from §5.3.

**Ear-EEG devices** (out-ear repo `ADS1299_BLE_muaz`; in-ear repo `ADS1299_BLE_gaoteng`), via `ble_data_provider.py`:
- GATT service `0000abf0-0000-1000-8000-00805f9b34fb`; characteristics WRITE `abf1`, NOTIFY `abf2`, COMMAND `abf3`, STATUS `abf4`.
- Sync command: write `struct.pack('<Bd', 0x04, unix_time_seconds)` (command byte `CMD_SET_TIME = 0x04` + a double-precision Unix timestamp) to the COMMAND characteristic.
- ACK: the device echoes the same double back over a STATUS-characteristic notification. The reference script treats a 3 s timeout as *non-fatal* and silently falls back to boot-relative SD timestamps — **our app must treat this as fatal instead** (hard stop + retry + operator alert), since a silent fallback here would produce a device with no usable wall-clock mapping for the whole session, discovered only during offline analysis.
- The same script also supports live BLE data streaming (ADS1299 8-ch @250Hz + optional PPG @100Hz + MLX90632 skin temperature) with CRC16-checked, chunked packets. Useful as an optional real-time signal-quality check during Preparation/Familiarization, but the SD-card recording (timestamped using the synced onboard clock) remains the authoritative data source for analysis, not the live BLE stream.
- **Firmware status:** this sync/PPG capability currently exists only on the out-ear repo's `feat/ads-ppg-timesync-ble` branch (not yet on `main`), and has **not been ported to the in-ear firmware at all** — see open items §7.

**Wristband**, via the provided `ble_sync_wristbands.py`:
- Nordic UART Service (`6E400001-...`); RX `6E400002`, TX `6E400003`.
- Sync command: plain-text `f"SYNC_MS {unix_ms}\n"` written to RX; device replies `SYNC_OK`/`SYNC_ERR` over a TX notification.
- Already batch-syncs multiple wristbands by BLE name prefix (`WristBand_`) and already supports a periodic re-sync loop (`--interval-minutes`, `--cycles 0` for indefinite). This means the §5.3 "dual/periodic sync" upgrade path is nearly free for wristbands specifically — the ear-EEG adapter is the one that would need equivalent looping logic added if/when we move past v1's single-sync design.

**Device identification risk (new, needs a decision before multi-device sessions):** both ear-EEG firmwares currently advertise the same default BLE name, `ESP_SPP_SERVER` (`DEVICE_NAME` in `ble_data_provider.py`). With an out-ear unit and an in-ear unit both active in the same room, plus multiple wristbands, name-based scanning cannot reliably disambiguate which physical device is which. Before implementation we need either (a) firmware-level distinct advertised names per unit (e.g., `EAR_OUT_01`, `EAR_IN_01`), or (b) a fixed MAC-address → device-role mapping configured per session in the app. Added to open items.

**Explicitly accepted limitation (per your direction):** this assumes on-device oscillator drift over a ~1 hr session is small enough not to matter for the current analysis needs. This is a known simplification, not an oversight.

### 5.3 Designed-in extension point (do not build now, but architect for it)

The sync module should expose a single interface, e.g. `DeviceSyncManager.sync(device) -> SyncRecord`, called once at session start. To upgrade later to dual-sync + linear drift correction (or periodic re-sync), only this module and the offline offset-reconstruction step should need to change — the event logger, task modules, and session schema should NOT need to know how many sync points exist. Concretely:
- `SyncRecord` should already be a list-per-device in the schema (not a single scalar), even though v1 only ever appends one entry. This avoids a schema migration later.
- The BLE connection/session-teardown code path should leave a natural hook where a second sync call could be inserted at end-of-session without restructuring the app.

### 5.4 Emotiv Flex integration (open item, see §7)

If the Emotiv Flex streams live via LSL, it can be tagged with host-clock-aligned timestamps directly (LSL supports clock offset correction between the LSL clock and local system clock), which is inherently more precise than the SD-card devices' single-point sync. If instead it only records to its own proprietary session format (EmotivPRO), it needs the same single-BLE/software-sync treatment as the other devices, using whatever marker/event API EmotivPRO exposes. **This needs a hardware/SDK capability check before implementation** (see open items).

### 5.5 Event log schema (draft)

One row per event, e.g. CSV or JSONL:

```
timestamp_host_utc, session_id, participant_id, task, block_index, trial_index,
condition_label, event_type, event_payload_json
```

`event_type` examples: `session_start`, `device_sync`, `device_recording_start`, `task_start`, `block_start`, `trial_start`, `stimulus_onset`, `response`, `rating_response`, `block_end`, `task_end`, `session_end`.

`event_payload_json` carries task-specific fields (e.g., n-back position/target/response, clip_id/valence_group, arithmetic tier/operands/answer) so the schema stays uniform across tasks while remaining fully descriptive.

---

## 6. Software Architecture

**Stack:** Python + PsychoPy.

**Proposed module breakdown:**

```
app/
  main.py                   # session runner / state machine (prep -> task1 -> break -> task2 -> break -> task3 -> debrief)
  config/
    session_config.yaml     # timings, tier params, n-back params, randomization seeds
    emotion_manifest.json   # clip_id -> file_path, fine_grained_label, valence_group, duration
  sync/
    device_sync.py          # DeviceSyncManager, SyncRecord schema, per-device-family adapter dispatch
    ear_eeg_sync.py          # GATT/struct protocol adapter — ports the sync logic from ble_data_provider.py
                              #   (out-ear repo: ADS1299_BLE_muaz, branch feat/ads-ppg-timesync-ble;
                              #    in-ear repo: ADS1299_BLE_gaoteng, once ported — see open item #2)
    wristband_sync.py        # wraps/reuses ble_sync_wristbands.py's SYNC_MS protocol
    emotiv_link.py            # Emotiv Flex integration (LSL or Cortex API) - TBD
  tasks/
    attention_nback.py       # spatial n-back task module
    emotion_faced.py         # video presentation + rating module
    stress_mat.py            # tiered arithmetic + MAT-style UI module
  logging/
    event_logger.py          # writes the unified event log described in 5.5
    session_manifest.py      # writes session-level metadata (participant id, device list, sync records, config snapshot)
  ui/
    common_widgets.py        # countdown timer, leaderboard widget, rating scale widget, fixation cross, cue screens
```

Each task module is responsible only for presenting stimuli and returning structured events to the shared `event_logger`; randomization, timing, and config are read from `session_config.yaml`/`emotion_manifest.json` rather than hardcoded, so tier difficulty, n-back parameters, and clip assignments can all be retuned after piloting without code changes.

---

## 7. Open Items (need resolution before/alongside implementation)

1. **Emotiv Flex integration method** — confirm whether LSL streaming is available for this device/software version, or whether we're limited to EmotivPRO/Cortex API recording. This determines the sync approach in §5.4. *(unresolved — pending your testing)*
2. **In-ear firmware has no BLE sync yet** — the time-sync protocol in §5.2.1 exists only in the out-ear repo (`ADS1299_BLE_muaz`, branch `feat/ads-ppg-timesync-ble`); it needs to be ported to the in-ear repo (`ADS1299_BLE_gaoteng`) before in-ear recordings can be wall-clock aligned. Until then, in-ear sessions would only have boot-relative timestamps. *(new, blocks in-ear data collection specifically)*
3. **Out-ear firmware branch not on `main`** — the sync/PPG-capable code lives on `feat/ads-ppg-timesync-ble`, not the default branch. Need a plan for merging or flashing devices from that branch before real sessions (this is a firmware/hardware task, not just app-side integration). *(new)*
4. ~~**BLE device name collision**~~ — RESOLVED: each advertised name is suffixed with the device's MAC address, so `ESP_SPP_SERVER`-prefixed names are unique per physical unit despite the shared prefix. No firmware change needed; the app's scanner should still key off full name (or MAC) rather than prefix alone when assigning device role.
5. **Wristband API** — RESOLVED. `ble_sync_wristbands.py` (provided) covers BLE sync via Nordic UART + `SYNC_MS` command; see §5.2.1. Firmware source for the wristband itself not yet reviewed — flag if data-streaming/SD-readout details are also needed, not just sync.
6. **FilmStim clip files** — currently on a separate machine; app will be built/tested against placeholder clips via the manifest schema until real files are transferred.
7. **Informed consent / pre-session questionnaire content** — draft spec written in §3.3; still needs a review pass from Lili/Yinan and confirmation of any IRB-mandated language.
8. **Pilot calibration pass** — the n-back accuracy bands (§4.1) and stress tier accuracy bands (§4.3) are design targets, not measured values; plan a small pilot (n=2–3) before full data collection to tune SOA/tier parameters against these targets.

---

## 8. Suggested Build Order

1. Session state machine + config loading + event logger (skeleton, no real tasks yet) — establishes the timing/logging backbone everything else plugs into.
2. BLE device sync module against one real device (whichever is easiest to get firmware access to first), with the extension-point interface from §5.3.
3. Task 1 (n-back) — simplest UI, validates the cue/block/rest timing pattern and response logging.
4. Task 3 (stress/MAT-style) — validates the countdown/leaderboard/hurry-up widgets and per-question logging.
5. Task 2 (emotion) — validates video playback + manifest-driven stimulus loading + rating widget (with placeholder clips).
6. Full session integration run (all three tasks + breaks + debrief) against placeholder/mock devices.
7. Swap in real FilmStim clips and real device firmware once available; pilot session(s); tune difficulty parameters per §7 item 7.
