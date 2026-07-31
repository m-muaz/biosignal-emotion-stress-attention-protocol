# EXG Collection Protocol

PsychoPy application for the emotion/stress/attention biosignal collection session (out-ear EEG, in-ear EEG, wristband, optional Emotiv Flex cap).

Full protocol design and rationale: [`docs/Engineering_Document.md`](docs/Engineering_Document.md).
Source brainstorming docs and paper surveys behind those decisions: [`docs/reference/`](docs/reference/).

## Setup

```bash
conda create -n exg_collection python=3.10 -y
conda activate exg_collection
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## Run

`python -m app.main` runs the emotion + stress portion of the session (consent ->
questionnaire -> familiarization -> emotion task -> break -> stress task ->
conclusion). The attention/focus task (OpenMATB) is **not** part of this flow --
it's a separate program, run as its own standalone step after this session
ends (see "Attention task (OpenMATB)" below).

```bash
python -m app.main --participant-id P001
# quick demo walkthrough, mocked devices, no hardware needed:
python -m app.main --participant-id DEMO001 --demo-scale 0.15 --devices-mode mock --skip-questionnaire
```

To test one task module in isolation -- same window, config, device sync, and event logging as a real session, just without consent/familiarization/breaks/the other tasks:

```bash
python -m app.run_task --task attention --participant-id TEST001
python -m app.run_task --task emotion --participant-id TEST001
python -m app.run_task --task stress --participant-id TEST001
# skip device sync entirely for the fastest UI/timing iteration loop:
python -m app.run_task --task stress --participant-id TEST001 --skip-device-sync
```

## Stimulus clips

Real video clips (currently sourced from the OpenLAV dataset) are **not** committed to this repo. Place them locally under `assets/clips/` (gitignored -- in practice a directory junction to wherever the dataset lives, e.g. `assets/clips -> D:\Datasets\OpenLAV`) and point `app/config/emotion_manifest.json` at that path. See the `_comment` field in `emotion_manifest.json` for the current clip/valence mapping and curation rationale.

Use `scripts/download_openlav.py` to fetch the OpenLAV clips on a new machine (requires `pip install requests`):

```
python scripts/download_openlav.py --output D:\Datasets\OpenLAV
```

It scrapes the official PsychArchives item page for each clip's download link and metadata CSV, then sorts clips into per-emotion subfolders matching the layout `emotion_manifest.json` expects.

Clips are played in an external video player -- [VLC](https://www.videolan.org/vlc/) by default, auto-detected on PATH or its common Windows install location -- rather than PsychoPy's own `MovieStim`, which had recurring decode-stall and early-cutoff bugs. Install VLC, or point `emotion_task.external_player_path` in `session_config.yaml` at a different player's `.exe`. Each clip runs fullscreen via `vlc --play-and-exit` and closes itself when playback ends (or if the participant closes it manually); our app just waits for that process to exit before showing the rating screen.

Our own PsychoPy window is closed before VLC launches and reopened right after -- two apps each holding exclusive fullscreen on the same display at once is what caused VLC to sometimes render incorrectly when our own window was also fullscreen (same reasoning as the OpenMATB subprocess below). We also pass `--no-one-instance` (so playback always runs in the process we're actually waiting on, rather than being handed off via IPC to an already-running VLC) and `--qt-continue=0` (so a "continue playback where you left off?" dialog can't silently block `--play-and-exit` forever) -- together these were the two causes of playback occasionally not displaying correctly or never returning control to the app.

## Attention task (OpenMATB)

The attention/focus task is [OpenMATB](https://github.com/juliencegarra/OpenMATB) -- an existing, validated sustained-attention/workload task battery -- run standalone as its own step, **after** `python -m app.main` (the emotion+stress session) finishes, rather than orchestrated by our own code. It is **not** committed to this repo (it's a separate open-source project, run via its own `main.py`); clone it locally and point `app/config/session_config.yaml`'s `attention_task.openmatb.install_path` at it (default: `vendor/OpenMATB`, gitignored).

`app/tasks/attention_openmatb.py` (launched via `python -m app.run_task --task attention`) is a Python wrapper that generates a scenario + config.ini and drives this as a subprocess from within our own event logging/window flow -- useful for testing that integration path, but not currently called from `app/main.py`. To just run OpenMATB directly instead (no wrapper), see the commands below.

**OpenMATB needs its own isolated Python environment -- do NOT install its requirements into this project's `data_collection`/`exg_collection` conda env.** OpenMATB requires `pyglet>=2.1,<3`, a backwards-incompatible rewrite (pyglet 2.x removed `pyglet.canvas` entirely) that breaks PsychoPy's window backend, which is hard-pinned to `pyglet==1.4.11` on Windows. Set it up with its own `.venv` instead (matches OpenMATB's own README convention):

```bash
git clone https://github.com/juliencegarra/OpenMATB vendor/OpenMATB
cd vendor/OpenMATB
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt   # Windows
# .venv/bin/python3 -m pip install -r requirements.txt        # macOS/Linux
```

To run OpenMATB directly (no wrapper), set `scenario_path` in `vendor/OpenMATB/config.ini` to the scenario file you want, then:

```bash
cd vendor/OpenMATB
.venv/Scripts/python.exe main.py   # Windows
# .venv/bin/python3 main.py        # macOS/Linux
```

`app/tasks/attention_openmatb.py` launches `vendor/OpenMATB/.venv/.../python.exe main.py` as a subprocess -- never this project's own interpreter -- so the two environments' conflicting pyglet versions never collide.

Only OpenMATB's `sysmon` (system monitoring) and `resman` (resource management) subtasks are used -- both are entirely keyboard-driven. Its `track` (tracking) subtask requires a physical joystick with no keyboard/mouse fallback in OpenMATB's code, so it's skipped (no joystick in this protocol's setup).

Each run (practice + real) generates a scenario file and `config.ini` inside the `vendor/OpenMATB` checkout, launches `python main.py` there, and afterward copies OpenMATB's own CSV log (and the generated scenario, for provenance) into the session folder. See `app/tasks/attention_openmatb.py` for the generation/orchestration logic, and `attention_nback.py` for the retired spatial n-back design it replaced (kept, commented out, for reference).

## Data

Session output (`sessions/<participant>_<timestamp>/events.jsonl`, `session_manifest.json`) is written locally and gitignored — it is participant data, not code.
