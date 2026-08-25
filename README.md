# Biosignal Dataset Sync & Dataloader Pipeline

Turns raw, multi-device biosignal recordings (ear-EEG, in-ear-EEG,
wristband, Polar H10) plus a task/event log into a canonical, host-clock-
synced dataset, then generates labeled `.npy`/`.npz` datasets for the
emotion/stress/attention tasks.

This branch is self-contained: it holds only the data-processing pipeline,
not the data-collection application that produced the raw recordings.

Full write-ups (preview any of these with
`python docs/render_mdx_preview.py <path>`, or open the already-rendered
`.preview.html` next to each):

- **`writeup/dataset_label_pipeline.mdx`** -- start here to generate
  datasets: the EEG preprocessing/QC pipeline, canonical-tables → labels →
  versions → windowing flow, and a table of every dataset version with
  foundation-model usage notes.
- `writeup/eeg_signal_analysis.mdx` -- the earlier raw-signal QC
  investigation (clipping, per-channel SNR, session viewers) that the
  preprocessing above was validated against. Not needed to generate or use
  the label datasets.
- `docs/Dataset_Sync_Design.md` -- lower-level sync/timestamp design.

## Requirements

`numpy`, `pandas`, `pyarrow`, `scipy`, `pyyaml`. `torch` is only required
for `EventWindowDataset.__getitem__(..., to_tensor=True)`.

## 1. Build the canonical synced dataset

```bash
# one participant
python -m dataset.build_dataset --participant-dir <raw_session_dir> --out-dir <out_dir>

# whole cohort
python -m dataset.build_dataset --root <sessions_root> --out-dir <out_dir>
```

Writes `<out_dir>/<participant_id>/{events,ear_eeg_out.ads1299,...}.parquet`
plus a `sync_report.json` audit trail per participant.

## 2. Generate label datasets

Two label-generation pipelines, one per task family, each with the same
CLI shape: `build_canonical` (raw tables) → `build_datasets` (labels,
selectable by version) → `export_npy` (EEG epochs as `.npy`/`.npz`).

### Stress / highway task

```bash
python -m dataset.stress_labels.build_canonical --processed-dir <out_dir> --raw-root <sessions_root>

# see what's available
python -m dataset.stress_labels.build_datasets --processed-dir <out_dir> --out-dir <label_dir> --list-configs

# build one dataset version's label table
python -m dataset.stress_labels.build_datasets --processed-dir <out_dir> --out-dir <label_dir> --config s4_raindrop_behavior

# export that version's EEG epochs to .npy
python -m dataset.stress_labels.export_npy --processed-dir <out_dir> --export-dir <npz_dir> --config s4_raindrop_behavior
```

### Attention task (SART / Stroop / Schulte)

```bash
python -m dataset.attention_labels.build_canonical --processed-dir <out_dir>

python -m dataset.attention_labels.build_datasets --processed-dir <out_dir> --out-dir <label_dir> --list-versions
python -m dataset.attention_labels.build_datasets --processed-dir <out_dir> --out-dir <label_dir> --version a2_sart_prospective_lapse

python -m dataset.attention_labels.export_npy --processed-dir <out_dir> --export-dir <npz_dir> --config a2_sart_prospective_lapse
```

Every `build_datasets`/`export_npy` command supports `--all` (build/export
every version) or `--list-versions`/`--list-configs` (print what's
available and exit) instead of a specific name. Dataset variants are YAML
files under `configs/{stress_labels,attention_labels}/` — copy one and
change its windowing/history fields to make a new variant without editing
Python. Every version's target(s), resolution, and window mode are listed
in `writeup/dataset_label_pipeline.mdx` §3.

### Emotion task (and any other task, generic exporter)

```bash
python -m dataset.export_npy --processed-dir <out_dir> --export-dir <out_dir>\npy_export --emotion-only
python -m dataset.sanity_check_npy --processed-dir <out_dir> --export-dir <out_dir>\npy_export
```

Drop `--emotion-only` to export every task key this exporter knows about
(`{emotion,math,highway,stroop,schulte,sart}_{trial,baseline}` plus the
`sart_lapse_trial`/`stroop_lapse_trial` keys). `sanity_check_npy.py`
independently recomputes a sample of epochs from the synced Parquet and
checks they're bit-exact; its report should read `SANITY CHECK: PASSED`.

## 3. Load windows directly (no fixed-shape export)

```python
from dataset.torch_dataset import EventWindowDataset

ds = EventWindowDataset(out_dir, ["P001", "P007"], tasks="emotion", target_hz=250.0)
sample = ds[0]  # dict of numpy arrays + stimulus/labels
```

## Output shape

Every exporter writes `X.shape == (N, C, samples_per_epoch)` float32 and
`y.shape == (N,)` int64, where `N` is the total number of fixed-length
epochs (not windows/trials — a window longer than one epoch contributes
several rows). EEG channels (`ear_eeg_out.ads1299`, optionally
`ear_eeg_in.ads1299`) are ADC-count→µV converted and notch+bandpass
filtered before epoching, with per-epoch QC columns (`qc_*`) attached to
the matching `meta.csv` — never raw ADC counts. A `<prefix>_meta.csv`
alongside `X`/`y` carries every label column plus this epoch's own
timestamps, for grouping epochs back into their source window/trial.

## Layout

```
dataset/
  raw/                     one parser per raw device format
  session_resolver.py       restart/opt-out/duplicate-file resolution
  sync.py                    per-device clock -> host-UTC
  build_dataset.py            CLI: raw -> canonical Parquet + sync_report.json
  windows.py                  event-defined window extraction, per task
  torch_dataset.py             EventWindowDataset + windowed biosignal loading
  eeg_preprocess.py             ADC counts -> uV conversion + filtering (shared)
  eeg_qc.py                      per-epoch EEG QC scoring (shared)
  export_npy.py                   generic fixed-shape .npy export, every task key
  sanity_check_npy.py              independent verification of export_npy.py output
  stress_labels/                   raindrop math + highway: canonical tables -> labels -> versions -> npy
  attention_labels/                 SART / Stroop / Schulte: same pipeline shape
configs/
  stress_labels/*.yaml       one dataset-version config per file
  attention_labels/*.yaml     one dataset-version config per file
writeup/dataset_label_pipeline.mdx   dataset preprocessing + label-generation writeup
writeup/eeg_signal_analysis.mdx       raw-signal QC investigation
docs/Dataset_Sync_Design.md            sync/timestamp design details
notebooks/dataset_walkthrough.ipynb   runnable example (build + inspect + plot)
```

Raw participant data is not included in this repository; `--root`/
`--participant-dir`/`--raw-root` must point at a local copy of the raw
sessions.
