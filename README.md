# Biosignal Dataset Sync & Dataloader Pipeline

Turns raw, multi-device biosignal recordings (ear-EEG, in-ear-EEG,
wristband, Polar H10) plus a task/event log into a canonical, host-clock-
synced dataset, and provides an event-windowed dataloader for DL/ML use.

This branch is self-contained: it holds only the data-processing pipeline,
not the data-collection application that produced the raw recordings.

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
