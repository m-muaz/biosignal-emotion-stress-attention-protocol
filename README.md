# EXG Collection Protocol

PsychoPy application for the emotion/stress/attention biosignal collection session (out-ear EEG, in-ear EEG, wristband, optional Emotiv Flex cap).

Full protocol design and rationale: [`docs/Engineering_Document.md`](docs/Engineering_Document.md).

## Setup

```bash
conda create -n data_collection python=3.10 -y
conda activate data_collection
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## The protocol, in one line

`python -m app.main` runs: consent -> questionnaire -> familiarization ->
**Task 2: Emotion** -> break -> **Task 3: Stress** (raindrop mental
arithmetic -> highway driving) -> break -> **Task 4: Attention/focus**
(Schulte table + Stroop test, alternating) -> conclusion. **SART** is
deliberately **not** part of this flow -- per PI request, it's a separate
standalone task you run on its own (see "Attention task (SART)" below).

## Running a real experiment

With real hardware connected (BLE ear-EEG/wristband, `devices.mode: real`
in `session_config.yaml`, or override with `--devices-mode real`):

```bash
python -m app.main --participant-id P001 --devices-mode real
```

This is fullscreen, real timing, asks the full questionnaire, and syncs
every enabled device's clock at startup before anything else happens. If
you already sync/verify your hardware via your own external scripts
**before** starting this app (so this app's own BLE connection attempt
doesn't conflict with an already-open one), skip this app's sync step:

```bash
python -m app.main --participant-id P001 --devices-mode real --skip-device-sync
```

(or set `devices.skip_sync: true` in `session_config.yaml` once, instead of
passing the flag every time -- see that key's comment for what gets logged
either way, so it's traceable later whether a session synced for real or
skipped it intentionally.)

## Demo / testing / debugging

**Walk through the whole session quickly**, mocked devices (no hardware
needed), skipping the questionnaire so you get to the tasks faster:

```bash
python -m app.main --participant-id DEMO001 --demo-scale 0.15 --devices-mode mock --skip-questionnaire
```

- `--demo-scale 0.15` shortens every passive/macro duration (baselines,
  rest, fixation, video length, trial counts) to 15% of real -- a ~58 min
  session becomes ~9 min. It deliberately does **NOT** shorten
  interactive per-event timing (raindrop's fall speed, highway's hazard
  timing, SART's 1.15s SOA) -- scaling those would make the task either
  trivial or literally impossible to actually play. Use `1.0` (the
  default) for real timing, or omit the flag entirely.
- `--devices-mode mock` skips real BLE hardware entirely -- everything
  else in the session behaves identically, so this is the normal way to
  test/demo the software with no hardware on hand.
- `--skip-questionnaire` and `--skip-familiarization` each skip that one
  phase, for quicker iteration once you've already checked it once.
- Add `--skip-device-sync` too if you don't even want the (instant, in
  mock mode) sync step's log lines.

**Test ONE task module in isolation** -- same config/device-sync/
event-logging as a real session, just without consent/familiarization/
breaks/the other tasks, so what you see matches what a participant would
see for that one piece:

```bash
python -m app.run_task --task emotion --participant-id TEST001
python -m app.run_task --task stress --participant-id TEST001          # raindrop only
python -m app.run_task --task highway --participant-id TEST001         # highway only
python -m app.run_task --task attention_focus --participant-id TEST001 # Schulte + Stroop
python -m app.run_task --task sart --participant-id TEST001            # SART (separate task)
python -m app.run_task --task sart --participant-id TEST001 --skip-practice   # skip SART's 18-trial practice block

# fastest possible iteration loop -- mock devices, skip sync, sped up:
python -m app.run_task --task emotion --participant-id TEST001 --devices-mode mock --skip-device-sync --demo-scale 0.1
```

Every real run (`app.main`) and isolated task run (`app.run_task`) writes
a full event log either way -- see "Where session data is saved" below --
so a demo run is a completely normal way to sanity-check that a config
change actually did what you expected before running it on a real
participant.

**Quit/skip keys**, live in every task (real or demo, mock or real
hardware):

| Key | Effect |
|---|---|
| `Esc` | Quit the whole session/task right now (logs `session_aborted`) |
| `B` | Skip the rest of the current block/tier/round (where applicable) |
| `N` | Skip the current clip/trial only (emotion task) |

## Demoing tasks to the participant before a real session

Before the real recording session, walk the participant through each task
once so they know what to expect -- this is what lets the actual
data-collection run skip its own in-app familiarization/practice (see
`--skip-familiarization` above and `--skip-practice` in "Attention task
(SART)"). Run each one standalone via `app.run_task`, same as the "Test ONE
task module in isolation" commands above, but with two things kept
deliberate for a *participant-facing* demo rather than a developer test:

- `--devices-mode mock` -- a demo run captures no real biosignal data, so
  there's no reason to require hardware connected/synced for it (this also
  happens to be `devices.mode`'s config default already, but pass it
  explicitly so a demo never accidentally waits on a real BLE sync).
- No `--demo-scale` (i.e. leave it at its default `1.0`) -- the participant
  needs to feel the *real* pacing (raindrop's countdown, highway's hazard
  speed, SART's fixed 1.15s/trial) to know what they're actually signing up
  for; scaling it down here would demo a task that doesn't match what the
  real session then puts them through.

```bash
python -m app.run_task --task stress --participant-id P001 --devices-mode mock          # raindrop
python -m app.run_task --task highway --participant-id P001 --devices-mode mock         # highway
python -m app.run_task --task attention_focus --participant-id P001 --devices-mode mock # Schulte + Stroop
python -m app.run_task --task sart --participant-id P001 --devices-mode mock            # SART (practice block on by default)
```

Each of these writes its own throwaway `sessions/P001_<task>_<timestamp>/`
folder (see "Where session data is saved" below), so a demo run is never
mistaken for -- or mixed into -- that participant's real session data.

`--task emotion` is deliberately **not** listed here: it plays clips from
the real study's clip set (`emotion_task.manifest_path`/`fixed_clips_path`),
so running it standalone as a demo would show the participant the actual
stimuli before the real task -- exactly what `app.main`'s built-in
familiarization phase avoids, by auto-previewing only the dedicated,
non-scored `familiarization_clip_id` clip instead. That preview (plus the
raindrop/highway/Schulte/Stroop auto-play demos also baked into
familiarization) already runs automatically as part of a normal
`python -m app.main` invocation -- see the one-liner above -- whenever
`--skip-familiarization` is *not* passed; there's no separate standalone
command for it today.

Once the participant has seen SART's demo above, the real session should
pass `--skip-practice` so they aren't shown the same practice block twice --
`run_full_session.ps1` already does this (see "Attention task (SART)").

## Changing config parameters

**Everything** timing/content-related lives in one file:
`app/config/session_config.yaml`. Every task has its own top-level key
(`emotion_task`, `stress_task`, `attention_highway_task`,
`attention_focus_task`, `sart_task`, `questionnaire`, `devices`, ...) and
every individual setting has an inline comment explaining what it does and
why it's set the way it is -- read the comment above a value before
changing it, especially anything marked `2026-08-05` (the most recent
timing-budget pass; see `docs/session_overview_2026-08-05.md` for the full
rationale behind those specific numbers).

A few of the most commonly-tweaked ones, to get oriented:

| Want to change... | Edit this |
|---|---|
| How many emotion clips per valence group | `emotion_task.clips_per_group` |
| Emotion task's baseline-recording lengths | `emotion_task.block_baseline` (start/mid/end, seconds) |
| Stress task difficulty/pacing | `stress_task.tiers` / `attention_highway_task.tiers` |
| How many Schulte/Stroop trials | `attention_focus_task.trials_per_game` |
| SART repetitions | `sart_task.reps` |
| Break lengths | `breaks.after_task1_min` / `breaks.after_task2_min` |
| Whether devices sync for real | `devices.mode` (`mock`/`real`), `devices.skip_sync` |

To try a config change without touching the real file, pass a different
one entirely: `--config path/to/my_test_config.yaml` (works on both
`app.main` and `app.run_task`) -- handy for A/B-ing a change before
committing to it.

## Where session data is saved (and how to change it)

Every session (real or demo, full run or single isolated task) writes to
its own folder under `app/config/session_config.yaml`'s
`logging.session_root` (default: `./sessions`, relative to the repo root,
already gitignored):

```
sessions/
  P001_1785987546/                 # app.main: <participant_id>_<unix_timestamp>
    events.jsonl                   # every event from every task, one JSON row per line
    session_manifest.json          # participant/config-snapshot/device-sync-record/rng_seed for this session
  TEST001_emotion_1785988318/      # app.run_task: <participant_id>_<task>_<unix_timestamp>
    events.jsonl
    session_manifest.json
```

`events.jsonl` is the one file every task in this project logs to (emotion,
stress, highway, attention/focus, SART, the session shell itself) --
`event_type` + `task` + `event_payload_json` on every row tells you what
happened and where.

**To change where this all goes**, edit `logging.session_root` in
`session_config.yaml` (e.g. `D:/exg_data/sessions`) -- every future run
(real or test) will write there instead. There's no separate per-run
override flag today; use a different `--config` file (see above) if you
want some runs going to one place and others elsewhere without editing the
main config back and forth.

One thing that's **not** inside a session folder: `stress_task.high_score_path`
(default `sessions/raindrop_high_scores.json`) -- the raindrop game's
cross-participant high score persists across the whole study, not per
session, so it lives one level up from the per-session folders even though
it's under the same `sessions/` root by default.

## Stimulus clips

Real video clips (currently sourced from the OpenLAV dataset) are **not** committed to this repo. Place them locally under `assets/clips/` (gitignored -- in practice a directory junction to wherever the dataset lives, e.g. `assets/clips -> D:\Datasets\OpenLAV`) and point `app/config/emotion_manifest.json` at that path. See the `_comment` field in `emotion_manifest.json` for the current clip/valence mapping and curation rationale.

Use `scripts/download_openlav.py` to fetch the OpenLAV clips on a new machine (requires `pip install requests`):

```
python scripts/download_openlav.py --output D:\Datasets\OpenLAV
```

It scrapes the official PsychArchives item page for each clip's download link and metadata CSV, then sorts clips into per-emotion subfolders matching the layout `emotion_manifest.json` expects.

Clips are played in an external video player -- [VLC](https://www.videolan.org/vlc/) by default, auto-detected on PATH or its common Windows install location -- rather than PsychoPy's own `MovieStim`, which had recurring decode-stall and early-cutoff bugs. Install VLC, or point `emotion_task.external_player_path` in `session_config.yaml` at a different player's `.exe`. Each clip runs fullscreen via `vlc --play-and-exit` and closes itself when playback ends (or if the participant closes it manually); our app just waits for that process to exit before showing the rating screen.

Our own PsychoPy window is closed before VLC launches and reopened right after -- two apps each holding exclusive fullscreen on the same display at once is what caused VLC to sometimes render incorrectly when our own window was also fullscreen. We also pass `--no-one-instance` (so playback always runs in the process we're actually waiting on, rather than being handed off via IPC to an already-running VLC) and `--qt-continue=0` (so a "continue playback where you left off?" dialog can't silently block `--play-and-exit` forever) -- together these were the two causes of playback occasionally not displaying correctly or never returning control to the app.

## Attention/focus task (Schulte table + Stroop test)

Two selective-attention/interference-control games added per PI discussion
2026-08-04, taking over the highway task's old slot as the fourth step of
`python -m app.main`'s own flow (the highway task has since moved to Task 3
"STRESS" instead -- see "Highway dodge task" below). Modeled on
[freefocusgames.com's Schulte table](https://www.freefocusgames.com/games/schulte-table)
and [Stroop effect test](https://www.freefocusgames.com/games/stroop-effect-test).

- **Schulte table**: a grid of numbers 1..N (`attention_focus_task.schulte.grid_size`^2,
  default 5x5 = 1-25) scattered in random positions -- click them in
  ascending order. Explicitly **untimed**: no countdown, no response
  deadline -- completion time is the dependent measure, not a race against
  a visible clock.
- **Stroop test**: a color word rendered in an "ink" color that may or may
  not match the word (congruent/incongruent) -- click the on-screen color
  swatch matching the INK, not the word (or press its number key -- both
  work). Self-paced (no response deadline); reaction time and accuracy are
  the dependent measures. Each fixed-position swatch is labeled with its
  own color's name and number-key shortcut -- a button's label always
  matches its own fill color by construction, so this isn't a *second*
  word-reading conflict at the response stage, just a clearer target than a
  bare swatch (and the number-key option removes the mouse-reach latency a
  click response otherwise adds on top of the RT being measured). The
  fixed position/order across trials still matters for the same reason.
  6 colors by default (red/green/blue/yellow/purple/pink) -- see
  `session_config.yaml`'s `attention_focus_task.stroop.colors` comment for
  a caveat on purple/pink vs. the classic 4-color literature.

Neither game uses difficulty tiers (unlike `stress_task`/
`attention_highway_task`) -- per PI request 2026-08-04, both run at one
fixed, comparable difficulty, since the point is measuring attention/focus
under a known paradigm, not a difficulty ramp. Every relevant knob (grid
size, Stroop color set, trial counts, ISI, baseline duration) is
config-overridable in `session_config.yaml`'s `attention_focus_task`.

**Session structure**: `trials_per_game` trials of *each* game (default 3 +
3 = 6 total blocks), each trial = one baseline (fixation) + one game round.
The across-trial order (which game comes up in which slot) is shuffled once
per session via the session rng, so a given game isn't confounded with a
fixed position in the sequence -- same reasoning the emotion task
randomizes clip order while keeping clip selection fixed. See
`app/tasks/attention_focus.py`'s `build_sequence()` for the full rationale,
and `app/webui/schulte/`, `app/webui/stroop/` for the frontends (same
pywebview-subprocess-per-launch architecture as the raindrop/highway games,
see `app/webui/bridge.py`) -- except here each of the 6 trials is its own
subprocess launch (`app/webui/schulte_app.py`/`app/webui/stroop_app.py`),
since the two games are different frontends that can't share one running
process. Logs `cell_click` (Schulte, per click) and `trial_start`/`response`
(Stroop, per stimulus) to the same `events.jsonl` as every other task.

## Highway dodge task

**Wired into `python -m app.main`'s flow as of 2026-08-05**, as the *second
half* of Task 3 ("STRESS") -- runs immediately after the raindrop mental-
arithmetic round, no break in between (the break after Task 3 covers both
halves). Repurposed from its original attention/focus role (per PI
discussion 2026-08-04) into this stress role instead. Still runnable
standalone too, for isolated testing: `python -m app.run_task --task
highway`. Compared to SART (one discrete go/no-go response stream to a
single stimulus location), this instead requires continuously monitoring
several lanes of oncoming hazards *and* the participant's own position, and
choosing a direction to respond with rather than just present/absent -- a
different demand profile, not a duplicate measurement.

Mechanic: the car auto-drives down a 3-lane highway; hazards approach ahead
in a random lane over a few seconds, and the participant presses LEFT/RIGHT
(or A/D) to snap into an adjacent lane before a hazard reaches them. No
permadeath -- a collision is just logged and the drive continues, same
"fixed-duration, mistakes don't end the round" design as the raindrop game,
since continuous biosignal recording needs a block that runs its full
length. A hazard is never spawned into the one remaining clear lane while
every other lane already has a live threat (`pickSpawnLane()` in
`app/webui/highway/game.js`), so a safe lane is always available.

**Anti-camping nudge**: since there's no permadeath and a safe lane is
always guaranteed, sitting in one lane the whole time would otherwise be a
perfectly safe (if inattentive) strategy. Can't literally force a keypress,
but once the participant has stayed in the same lane for
`camp_grace_sec`, the probability that the *next* hazard spawns directly
into their current lane ramps from 0 to `camp_max_probability` over the
following `camp_ramp_sec` -- so staying put stops being free and starts
costing real collisions. Each `obstacle_spawn` event logs `camp_nudged` so
this is distinguishable from an ordinary random spawn during analysis.

**Speed/difficulty** is configured in `session_config.yaml`'s
`attention_highway_task.tiers` (same `tiers`/`trial_order` shape as
`stress_task`) -- each tier sets `spawn_interval_range_sec` (jittered gap
between hazards; lower = more frequent = harder) and `fall_duration_sec`
(time a hazard takes to reach the player, i.e. the reaction window; lower =
faster = harder). The default ramps from easy to hard across 3 tiers, like a
classic road-crossing arcade game, per PI request 2026-08-04 -- but as
discrete tiers rather than a smooth continuous ramp: a continuous ramp would
confound "it got harder" with "attention lapsed" at every instant, whereas
each tier's own `block_start`/`block_end` gives analysis a known,
constant-difficulty window to isolate vigilance decrement within, separate
from the deliberate jump between tiers. See `app/tasks/attention_highway.py`
for the full rationale.

Renders in a pywebview subprocess (`app/webui/highway_app.py`,
`app/webui/highway/`), same architecture as the raindrop game (see
`app/webui/bridge.py`). Logs `obstacle_spawn`/`obstacle_resolved` (per-hazard
lane/outcome) and `lane_change` (per-response direction, reaction time
measured from the threatening hazard's spawn, and whether the move was
forced by an actual threat or unprompted) to the same `events.jsonl` as
every other task.

## Attention task (SART)

The attention/focus task is the **Sustained Attention to Response Task (SART)**
(Robertson et al., 1997) -- participants press SPACE for every digit 1-9
*except* the omit number(s) (default: 3), and withhold on those. Chosen after
hands-on comparison of several existing attention/vigilance tasks -- PVT,
SART, gradCPT, the Lateralized Attention Task, and a PsychoPy whack-a-mole
go/no-go game -- found SART the most intuitive to play while still requiring
genuine sustained focus.

**Ported into this repo 2026-08-05** as `app/tasks/attention_sart.py`, from
the standalone tryout originally cloned to
`attention-task-tryouts/sart_cstothart/` (outside this repo). Per PI request,
it's kept as a **separate, standalone task** -- deliberately NOT wired into
`app/main.py`/`session_shell.py`'s session flow like the Schulte/Stroop
attention/focus task is (was briefly planned as a 3rd game sharing that
task's slot; reverted so it can be run as its own command instead):

```bash
python -m app.run_task --task sart --participant-id P001
```

**Running the main session and SART back-to-back for one participant:**
since participants are now shown a demo of every task (including SART)
before the session starts, there's no need for SART's own in-app practice
block anymore -- pass `--skip-practice` to skip straight to the real
135-trial block. To run both commands in sequence (main session, then a
short countdown, then SART) without typing two commands by hand, use
`run_full_session.ps1` (repo root) from an already-activated
`data_collection` prompt:

```powershell
(data_collection) PS ...\biosignal-emotion-stress-attention-protocol> .\run_full_session.ps1 -ParticipantId shruti
```

This runs `python -m app.main --participant-id shruti --devices-mode real
--skip-familiarization`, waits 5s (`-CountdownSeconds` to change), then runs
`python -m app.run_task --task sart --participant-id shruti --devices-mode
real --skip-practice`. It checks the main session's exit code first and
will **not** start SART if the operator aborted it (Escape) or it crashed --
see `app/main.py`/`app/run_task.py`'s exit codes (`0` = completed, `2` =
operator aborted, `1` = crashed). `-DevicesMode mock` and
`-SkipDeviceSync` are also available, applied to both commands.

Same fullscreen instructions -> 18-trial practice block (CORRECT/INCORRECT
feedback) -> a new ~90s pre-task physiological baseline (fixation cross, not
in the original script) -> the real block (`sart_task.reps` x 45 trials,
default 3 x 45 = 135 trials, no feedback) flow as the original, with the same
core 250ms-digit/900ms-mask timing (fixed 1.15s/trial, unchanged). Every knob
(`reps`, `omit_count`/`omit_num`, `practice`, `fixed_order`,
`baseline_duration_sec`) is config-overridable in `session_config.yaml`'s
`sart_task`.

Two things fixed during the port:
- **Escape now quits the task** (raises the same `UserQuit` every other task
  in this project uses) -- the original script had no way to quit mid-task
  at all; closing the window or Ctrl+C-ing the terminal was the only option.
- **Logs to `events.jsonl`** (via `ctx.event_logger`, same as every other
  task) instead of its own hand-rolled tab-delimited `.txt` file and
  `gui.Dlg` participant-info popup (participant ID already comes from
  `--participant-id`, matching how every other task in this project works).
  See "Logged data" below for the event schema.

Source of the original: [cstothart/sustained-attention-to-response-task](https://github.com/cstothart/sustained-attention-to-response-task)
(MIT licensed; Stothart, C. (2015). *Python SART* (Version 2) [software]) --
see `app/tasks/attention_sart.py`'s module docstring for the full attribution
and exactly what changed vs. the original. Needs no separate environment/
venv -- only depends on PsychoPy, already in this project's own
`requirements.txt`/`data_collection` env.

**Logged data** (`events.jsonl`, same file every other task writes to --
filter by `task: "sart"`): `task_start`/`task_end` bracket the whole task;
`baseline_start`/`baseline_end` (`position: "pre_task"`) bracket the new pre-
task baseline; `block_start`/`block_end` bracket the practice block
(`block_index: 0, practice: true`) and the real block (`block_index: 1,
practice: false`); each trial logs a `trial_start` (`number`, `font_size`,
`is_omit`, `omit_numbers`) immediately followed by a `response`
(`responded`, `rt` -- `null` if withheld, `accurate`). That's the same
commission/omission-error and RT-variability data the original's txt file
had, now timestamped on this project's host clock
(`timestamp_host_utc`/`timestamp_monotonic`) instead of `time.perf_counter()`
values relative to the script's own process start -- so it lines up with the
biosignal timestream the same way every other task's events already do,
without a separate alignment step.

## Data

Session output (`sessions/<participant>_<timestamp>/events.jsonl`, `session_manifest.json`) is written locally and gitignored — it is participant data, not code.
