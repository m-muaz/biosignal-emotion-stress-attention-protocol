# Biosignal Dataset Sync & Dataloader Pipeline

Turns raw, multi-device biosignal recordings (ear-EEG, in-ear-EEG,
wristband, Polar H10) plus a task/event log into a canonical, host-clock-
synced dataset, and provides an event-windowed dataloader for DL/ML use.

This branch is self-contained: it holds only the data-processing pipeline,
not the data-collection application that produced the raw recordings.

## The data collection protocol (for context)

Each participant wears the ear-EEG (mandatory) + an optional second
in-ear-EEG + a wristband + a Polar H10 chest strap, all recording
continuously through the whole session. Devices are synced to the host
clock right before recording starts (see `docs/Dataset_Sync_Design.md`).
A session runs through these tasks in order, in one continuous app; SART
runs separately afterward, as its own standalone program.

- **`preparation`** -- consent screen, then a questionnaire (demographics,
  handedness, sleep/caffeine, a 6-item state-anxiety scale). One
  `questionnaire`-window per item (its answer + self-timed response time).
- *(optional familiarization: a short practice run of each task below, not
  part of the scored dataset)*
- **`emotion`** -- participant watches short video clips (positive/
  negative/neutral, ~3 blocks), rating each afterward on valence, arousal,
  liking, and a forced-choice emotion pick. Baseline touchpoints (start/
  mid/end) bracket each block. One `trial` window per clip (+ its
  ratings), plus `baseline` windows.
- *break*
- **`stress`** (mental arithmetic, "raindrop") -- arithmetic problems fall
  down the screen like raindrops; the participant types the answer before
  it reaches the bottom, across 3 increasingly time-pressured tiers, each
  with its own baseline and a short pause between tiers. One `trial`
  window per problem, one `block` window per tier (accuracy), plus
  baselines.
- **`attention_highway`** (no break before this -- same "stress" task,
  second half) -- a lane-based driving game; hazards appear and the
  participant switches lanes to avoid them, across 3 tiers of increasing
  difficulty, with one baseline before the first tier and a subjective
  self-report (stress/workload/control/engagement/difficulty) after each
  tier. One `trial` window per obstacle (avoided/collision/reaction time),
  one `block` window per tier, plus baseline/`self_report` windows.
- *break*
- **`attention_focus_schulte`** / **`attention_focus_stroop`** -- 3
  repetitions each, in randomized/interleaved order, each preceded by its
  own baseline:
  - *Schulte table*: find numbers 1-25 in order on a 5x5 grid as fast as
    possible. One `block` window per repetition (completion time,
    misclicks).
  - *Stroop test*: name the ink color of a color word, ignoring the word
    itself (20 stimuli/repetition). One `trial` window per stimulus
    (word, ink color, congruent, response, rt), one `block` window per
    repetition (accuracy, mean rt).
- **debrief** -- closing message, end of the main session.

**`sart`** (run separately, its own program, sometime after the main
session): Sustained Attention to Response Task -- digits 1-9 flash one at
a time; the participant presses a key for every digit except a few
designated "no-go" numbers. One continuous ~135-trial run after a
baseline, no tiers -- so its whole run already is one `block` window,
with one `trial` window per digit (responded/accuracy/rt).

## What it does

1. **Resolve** (`dataset/session_resolver.py`) -- per participant, finds the
   right raw file among restarts/duplicates/opt-outs (e.g. an aborted
   device boot, a re-synced wristband session), never silently.
2. **Sync** (`dataset/sync.py`) -- maps every device's own clock onto one
   host-UTC timeline, using each device's real sync handshake.
3. **Build** (`dataset/build_dataset.py`) -- writes canonical per-stream
   Parquet + a full audit trail (`sync_report.json`) per participant.
4. **Window** (`dataset/windows.py`) -- extracts labeled, event-defined time
   windows per task (trial/block/baseline/self-report/questionnaire, plus
   auto-detected gaps) from the synced event log.
5. **Load** (`dataset/torch_dataset.py`) -- an `EventWindowDataset` pairing
   biosignals + stimulus + labels per window, resampled on the fly.

Full design writeup (sampling rates, sync formulas, resolution rules, known
data-quality caveats): **`docs/Dataset_Sync_Design.md`**. Runnable example
end-to-end on one participant: **`notebooks/dataset_walkthrough.ipynb`**.

## Event window kinds

`dataset/windows.py` extracts several kinds of window per task (`Window.window_type`):

| `window_type` | meaning |
|---|---|
| `trial` | one fine-grained scored unit -- one arithmetic problem, one obstacle, one Stroop stimulus, one SART number, one emotion clip |
| `block` | one uninterrupted run -- one raindrop/highway tier, one Schulte/Stroop repetition, SART's single continuous run (CONTAINS its `trial` windows) |
| `baseline` | a physiological-baseline touchpoint |
| `self_report` | a post-block subjective rating (highway only) |
| `questionnaire` | one preparation-phase questionnaire item response |
| `transition` | an auto-detected gap not covered by any other window (inter-task breaks, the pre-SART gap, etc.) |

**Which one is "a trial"?** For most tasks, `trial` is too fine-grained to
be a useful atomic unit on its own -- one arithmetic problem or one
obstacle alone doesn't carry much signal. `RECOMMENDED_TRIAL_WINDOW_TYPE`
in `dataset/windows.py` is the single source of truth for what to treat as
one trial per task:

| task | recommended unit | why |
|---|---|---|
| `emotion` | `trial` (one clip) | a clip has no finer natural sub-event to fall back to |
| `stress` (raindrop) | `block` (one tier) | one arithmetic problem alone is too short a signal |
| `attention_highway` | `block` (one tier) | one obstacle alone is too short a signal |
| `attention_focus_schulte` | `block` (one grid solve) | Schulte has no per-cell trial_index at all |
| `attention_focus_stroop` | `block` (one 20-stimulus repetition) | one stimulus alone is too short a signal |
| `sart` | `block` (the single continuous run) | SART has no tiers/repetitions -- its whole run already is one uninterrupted trial |

The finer `trial` windows are still extracted for every task, for anyone
who wants sub-trial-level analysis later (e.g. per-obstacle reaction time)
-- they're just not the default answer to "what is a trial".

## Quick start

```bash
# one participant
python -m dataset.build_dataset --participant-dir <raw_session_dir> --out-dir <out_dir>

# whole cohort
python -m dataset.build_dataset --root <sessions_root> --out-dir <out_dir>
```

```python
from dataset.torch_dataset import EventWindowDataset

ds = EventWindowDataset(out_dir, ["P001", "P007"], tasks="emotion", target_hz=250.0)
sample = ds[0]  # dict of numpy arrays + stimulus/labels
```

## Requirements

`numpy`, `pandas`, `pyarrow`, `scipy`, `pyyaml`. `torch` is only needed if
you call `EventWindowDataset.__getitem__(..., to_tensor=True)`.

## Layout

```
dataset/
  raw/                one parser per raw device format
  overrides.yaml       documented per-participant manual sync fixes
  session_resolver.py   restart/opt-out/duplicate-file resolution
  sync.py               per-device clock -> host-UTC
  build_dataset.py       CLI: raw -> canonical Parquet + sync_report.json
  windows.py             event-defined window extraction, per task
  torch_dataset.py        EventWindowDataset + windowed biosignal loading
docs/Dataset_Sync_Design.md   full design writeup
notebooks/dataset_walkthrough.ipynb   runnable example (build + inspect + plot)
```

Raw participant data isn't included in this repo -- point `--root`/
`--participant-dir` at your own local copy.
