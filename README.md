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
  negative/neutral, ~3 blocks (10 videos total = 2x 3 videos/block + 1x 4 videos/block)), answers emotion choice pick (Positive/Negative/Neutral), and rating each afterward on valence ([Valence (Wikipedia)](https://en.wikipedia.org/wiki/Valence_(psychology))), arousal,
  liking. Baseline touchpoints (start/
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

Every recording gets cut up into small time segments ("windows"), each
labeled with what was happening during it. There are two levels of segment
for how well someone did on a task, plus a few special ones:

- **`trial`** -- the smallest single thing that happened: one math problem,
  one obstacle on the highway, one word shown in the Stroop test, one
  number flashed in SART, one video clip watched.
- **`block`** -- one complete, uninterrupted round of the task, start to
  finish (e.g. all the obstacles in one highway run, all 20 Stroop words
  in one sitting, the whole SART run). A block is made of many trials
  back to back.
- **`baseline`** -- a quiet moment with no task happening, so we know what
  "resting" looks like for that participant.
- **`self_report`** -- the participant rating how stressed/engaged/etc.
  they felt, right after a block (highway only).
- **`questionnaire`** -- one answer to one background/mood question, asked
  before any task starts.
- **`transition`** -- any other stretch of time that isn't part of a task
  (breaks, waiting for the next screen, etc.) -- found automatically, so
  nothing is left unlabeled.

### Which one should I actually treat as "one trial"?

A single math problem or a single highway obstacle only lasts a couple of
seconds -- too short to see much of a pattern in slower biosignals like
heart rate or skin conductance. So for most tasks, treat one whole
**block** (the complete round) as "one trial" for analysis, not the
individual events inside it. Emotion is the one exception: each video clip
is already a natural, self-contained unit, so there a `trial` (one clip) is
exactly what you want.

`RECOMMENDED_TRIAL_WINDOW_TYPE` in `dataset/windows.py` gives you this
answer directly:

| Task | Use this as "one trial" | In plain terms |
|---|---|---|
| `emotion` | `trial` (one clip) | one video clip watched + rated |
| `stress` (raindrop math) | `block` (one round) | one full ~60s round of falling math problems |
| `attention_highway` | `block` (one round) | one full ~60s round of driving/dodging |
| `attention_focus_schulte` | `block` (one round) | one full attempt at finding all 25 numbers |
| `attention_focus_stroop` | `block` (one round) | one full set of 20 color-word stimuli |
| `sart` | `block` (the whole run) | the entire ~135-number run -- there's only one round to begin with |

The individual `trial` windows are still there if you ever want to zoom
into one specific event (e.g. reaction time to one obstacle) -- they're
just not the recommended starting point.

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

### Fixed-shape `.npy` export (for external DL/ML loaders)

If you just want ready-to-load `X.npy`/`y.npy` pairs (one per task, plus
each task's related baseline windows) instead of `EventWindowDataset`'s
variable-length per-window dicts:

```bash
python -m dataset.export_npy --processed-dir <out_dir> --export-dir <out_dir>\npy_export
python -m dataset.sanity_check_npy --processed-dir <out_dir> --export-dir <out_dir>\npy_export
```

Every task-defined window (e.g. one video clip's `[t_start, t_end]`) is cut
into fixed-length, **non-overlapping epochs** (default 1.0 real second,
the unit a foundation-model-style loader consumes) before any resampling.
A clip only ever contributes `floor(duration_seconds / epoch_seconds)`
epochs -- any leftover shorter than one full epoch (e.g. the last 0.2s of a
67.2s clip at the 1.0s default) is dropped, never padded. Every epoch
becomes its own row and inherits its parent clip's label, so a clip running
67.2s and another running 70.5s both just contribute their own (different)
number of same-shaped rows -- no padding/truncation needed to concatenate
epochs from clips of different lengths, or across participants.

Writes `<export_dir>/<participant_id>/<file_prefix>_{X,y,meta}.{npy,npy,csv}`
plus `<export_dir>/pooled/<file_prefix>_*` (all participants concatenated),
for `{emotion,math,highway,stroop,schulte,sart}_{trial,baseline}` plus two
attentional-lapse labels (`sart_lapse_trial`, `stroop_lapse_trial`) -- by
default across **every device this protocol collects**: the mandatory
out-ear EEG (unsuffixed filenames), the optional in-ear EEG device plus its
onboard PPG/IMU/temp/env sensors (`_eegin`/`_inear_ppg`/`_inear_imu`/
`_inear_mlx`/`_inear_bme` suffix), every wristband modality (`_ppg`/`_imu`/
`_gsr`/`_mag`/`_mlx`/`_bme` suffix), and the Polar H10 chest strap
(`_ecg`/`_polar_acc` suffix). `X.shape == (N, C, samples_per_epoch)` --
**`N` is the total number of epochs, not clips/windows** -- `y.shape ==
(N,)`, with `C`/`samples_per_epoch` depending on the stream. Pass
`--streams <names>` to export a subset instead (stream names are the
`<device>.<role>` Parquet file names under each participant's output dir,
e.g. `ear_eeg_out.ads1299`, `wristband.gsr`, `polar_h10.ecg`), and
`--epoch-seconds <key>=<seconds>,...` to change the epoch length for one or
more file keys (default 1.0s for every key). Full design (label scheme incl.
the attentional-lapse windows, sample-rate defaults, verification method):
`docs/Dataset_Sync_Design.md` §8-9.

#### Generating just the video emotion task (e.g. to hand off to a collaborator)

The emotion task's labels are the only ones finalized so far (design-assigned
valence group -- see the table above; stress/attention lapse labels are
still under discussion, see `PROGRESS.md`). To export only `emotion_trial`/
`emotion_baseline` (skips every other task, incl. the Schulte cohort-median
pass) with every device's data attached:

```bash
# 1. Build the canonical synced dataset (skip if you already have <out_dir>)
python -m dataset.build_dataset --root <sessions_root> --out-dir <out_dir>

# 2. Export just the emotion task, every stream, to <out_dir>\npy_export
#    (1-second epochs by default; e.g. --epoch-seconds emotion_trial=2 for 2s epochs instead)
python -m dataset.export_npy --processed-dir <out_dir> --export-dir <out_dir>\npy_export --emotion-only

# 3. Independently verify the export before sharing it (recomputes a sample
#    of epochs straight from the synced Parquet and checks they're bit-exact)
python -m dataset.sanity_check_npy --processed-dir <out_dir> --export-dir <out_dir>\npy_export
```

What your collaborator gets, per participant (plus `pooled/` with every
participant concatenated):

- `emotion_trial_X.npy` / `emotion_trial_y.npy` -- one row per **1-second
  epoch** of a video clip (not one row per clip): a 67.2s clip contributes
  67 rows, a 70.5s clip contributes 70. `X.shape == (N, 8, 200)` (out-ear
  EEG, 8 channels, 200-point grid per 1s epoch). `y` is `1`/`2`/`3` for
  negative/neutral/positive (the assigned stimulus valence group -- not
  behavior-derived, so no circularity risk); every epoch from the same clip
  shares that clip's label.
- `emotion_baseline_X.npy` / `emotion_baseline_y.npy` -- same idea for the
  resting baseline windows recorded before/mid/after each block, kept as
  separate rows from the trial windows (`y` is always `0`).
- The same two pairs again per additional stream, suffixed (e.g.
  `emotion_trial_eegin_X.npy`, `emotion_trial_ecg_X.npy`,
  `emotion_trial_gsr_X.npy`) -- a stream's row *count* can differ from the
  EEG version of the same file if a device had a gap for a given window, so
  don't assume row `i` lines up across streams without checking
  `t_start`/`t_end` in the matching `_meta.csv`.
- `emotion_trial_meta.csv` -- one row per `X`/`y` row (i.e. per epoch), with
  the participant ID, block/trial index, this epoch's own `t_start`/`t_end`
  plus `epoch_index`/`n_epochs_in_window` (to group epochs back into their
  source clip -- also identifiable via the shared `window_t_start`/
  `window_t_end` columns), and (in `labels_json`) every raw post-clip
  response the participant gave: `emotion_pick` (their own pick of
  positive/negative/neutral), `valence`, `arousal`, `liking`, plus response
  times -- so nothing beyond the small integer `y` label is thrown away.

`export_manifest.json` (written alongside the per-participant folders) is
the audit trail -- per-participant/pooled epoch counts, dropped-window
reasons (incl. `too_short_for_one_epoch`, for any clip shorter than one
epoch), and the label map. `sanity_check_npy.py`'s printed report should
say `SANITY CHECK: PASSED` before sharing the export.

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
  export_npy.py            fixed-shape (B, C, T, 200) .npy export, per task
  sanity_check_npy.py       independent verification of export_npy.py output
docs/Dataset_Sync_Design.md   full design writeup
notebooks/dataset_walkthrough.ipynb   runnable example (build + inspect + plot)
```

Raw participant data isn't included in this repo -- point `--root`/
`--participant-dir` at your own local copy.
