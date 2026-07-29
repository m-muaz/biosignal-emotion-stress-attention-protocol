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

```bash
python -m app.main --participant-id P001
# quick demo walkthrough, mocked devices, no hardware needed:
python -m app.main --participant-id DEMO001 --demo-scale 0.15 --devices-mode mock --skip-questionnaire
```

## Stimulus clips

Real FilmStim video clips are **not** committed to this repo (large, copyrighted commercial film files). Place them locally under `assets/clips/` (gitignored) and point `app/config/emotion_manifest.json` at that path. See `docs/reference/video_language_table.md` for the clip/valence mapping.

## Data

Session output (`sessions/<participant>_<timestamp>/events.jsonl`, `session_manifest.json`) is written locally and gitignored — it is participant data, not code.
