# Progress notes: label generation (WORK IN PROGRESS)

**Update 2026-08-20:** `export_npy.py` now supports `--emotion-only` (skips
every non-emotion file key, incl. the Schulte cohort-median pass) and
defaults to exporting every device's data, not just EEG + wristband --
Polar H10 (`polar_h10.ecg`/`polar_h10.acc`) and the full in-ear device
(`ear_eeg_in.ads1299` + its onboard PPG/IMU/temp/env sensors) are now wired
in as additional streams. `sanity_check_npy.py` now reads the actual
exported file keys from `export_manifest.json["file_keys"]` instead of
hardcoding every possible key, so it verifies an `--emotion-only` export
correctly instead of reporting missing stress/attention files. See README's
"Generating just the video emotion task" section. This was specifically for
handing the emotion task off to a collaborator now, while the stress/
attention label open questions below are still unresolved -- nothing here
touches those.

**Update 2026-08-20 (2):** `export_npy.py`'s segmenting was redesigned from
fixed segment-count `T` (computed per file key from the *median* real
duration, `X.shape == (B, C, T, samples_per_segment)`) to fixed-length
**epochs** (default 1.0s, overridable via `--epoch-seconds key=N,...`):
every window is now cut into `floor(duration_s / epoch_s)` non-overlapping
epochs, any leftover shorter than one epoch dropped (never padded), and
every epoch becomes its own row -- `X.shape == (N, C, samples_per_epoch)`
with `N` = total epochs, not windows/clips. Rows from clips of different
real lengths concatenate directly with no padding/truncation. New
`meta.csv` columns: `window_t_start`/`window_t_end` (parent clip's span),
`epoch_index`/`n_epochs_in_window` (to regroup epochs into their source
clip); `t_start`/`t_end` now mean the epoch's own bounds, not the window's.
`sanity_check_npy.py` updated to match (recomputes per-epoch, cross-checks
epoch counts). `docs/Dataset_Sync_Design.md` §8's segment-count writeup is
now flagged superseded rather than rewritten in place -- see its note.

Status as of 2026-08-18. This is a handoff note so work can continue on
another machine while this one is tied up running data collection. Full
technical design/decisions already made live in `docs/Dataset_Sync_Design.md`
(§8-9) -- this file is the shorter "where things stand + what's still open"
summary on top of it.

## What's done and validated

- **Dataset sync + canonical Parquet pipeline** (`dataset/build_dataset.py`
  and friends): validated end-to-end against all 12 real participants
  (P001-P012), zero crashes. See `docs/Dataset_Sync_Design.md` §1-4.
- **Event-windowed dataloader** (`dataset/windows.py`, `dataset/torch_dataset.py`):
  all 7 tasks/sub-tasks implemented, 10,103 total windows across 9
  participants validated, per-task window counts cross-checked against real
  event-type counts. See §5.
- **Fixed-shape `.npy` export** (`dataset/export_npy.py`,
  `dataset/sanity_check_npy.py`, both new/untracked in this commit): turns
  the variable-length window dicts into `{X: (B,C,T,200), y: (B,)}` per task,
  for a downstream collaborator's DL/ML pipeline. Validated 2026-08-18 across
  all 12 participants x 14 file keys x 7 streams (EEG + 6 wristband
  modalities): 3913 recomputed sample rows, all bit-exact, 0 warnings.
  See §8-9.
- **Session-subdir naming fix** (`dataset/session_resolver.py`): fixed the
  3rd naming-convention variant (P010-P012 use `<firstname>_<ts>` instead of
  `part-<ID>_<ts>`) by matching structurally instead of hardcoding a prefix.
  Found and fixed while building the npy export against all 12 real
  participants. See §3.
- **Basic condition/stimulus labels already implemented** (§8's 12 file
  keys): emotion valence group, math/highway difficulty tier, Stroop
  congruency, SART go/no-go, Schulte speed median-split. All assigned by
  experimental design or stimulus identity -- not behavior-derived, so no
  circularity risk.
- **First attentional-lapse labels implemented** (§9, added 2026-08-18):
  `sart_lapse_trial` (commission-error definition, Robertson et al. 1997) and
  `stroop_lapse_trial` (incorrect-response definition), both with the EEG
  epoch truncated to end 0.15s before the actual keypress so the label can't
  just be reading its own motor/error-related signal back. Real cohort
  counts: SART 406 no-go trials / 77 lapses (~19%, matches literature);
  Stroop 708 trials / 12 errors (~1.7%, too imbalanced to be a primary label
  on its own).

## Open questions still to finalize (the actual blocker)

This is what's unfinished -- picking these up is the next work session's
job:

1. **Is a behavior-derived "attention" label even the right target for
   every task, or should some tasks stick to design-assigned condition
   labels only?** SART/Stroop lapse labels answer "did the participant
   error on this trial" -- that's a reasonable proxy for a lapse, but it's
   still worth deciding explicitly whether the modeling goal is
   "classify attention state" vs. "classify task condition" vs. "predict
   error probability," since they imply different label choices and
   different claims about what the EEG features would mean.
2. **Schulte's median-split circularity risk (flagged in §9, not yet
   resolved):** `completion_time_sec` is literally `t_end - t_start` for
   that window, so the label is a direct function of window duration.
   Resampling to fixed T=26 segments hides the raw number but "how much
   real time got compressed per grid point" could still leak signal. Needs
   a decision: keep as-is (documented caveat), switch to `misclicks` as the
   split variable instead (duration-independent, already present in
   `labels_json`), or drop Schulte lapse/performance labels entirely and
   keep only the design-assigned tasks for that game.
3. **Attention_highway and stress/raindrop have no lapse-style label yet**
   -- only tier/condition labels exist for them (§8). If the goal is a
   cohort-wide "attention lapse across all tasks" label set (not just
   SART/Stroop), need to define what a "lapse" means for a continuous
   avoidance game (highway) and a timed arithmetic task (raindrop) --
   plausible candidates: highway collision events, raindrop wrong-submission
   events -- neither is wired into any window/label yet.
4. **Stroop class imbalance (12 errors / 708 trials):** decide whether to
   (a) use `stroop_trial` congruency instead of `stroop_lapse_trial` when
   class balance matters more than lapse-specificity (current
   recommendation in §9), (b) pool errors across the whole cohort and accept
   the imbalance with class weighting downstream, or (c) find/define a
   softer "borderline lapse" signal (e.g. slow-but-correct RT outliers) to
   get a larger positive class -- not decided.
5. **Whether the 0.15s pre-response truncation buffer is the right value**
   for both tasks, or should be tuned per-task/per-participant based on
   actual readiness-potential timing in this cohort's own EEG -- currently a
   single literature-informed constant, not validated against this
   dataset's own motor-preparation signal.
6. **Whether/how to combine multiple streams' labels** -- now that
   `export_npy.py` exports EEG + 6 wristband modalities per file key (§9),
   decide if wristband-derived signals (e.g. GSR) should ever inform or
   cross-validate a label (e.g. arousal), or stay purely as an independent
   input stream with the label always coming from the task/EEG side.

## Where to pick this up

- Read `docs/Dataset_Sync_Design.md` §8-9 first (full label scheme + the
  circularity discussion already worked through for SART/Stroop).
- `dataset/export_npy.py`'s `_build_sart_lapse_windows` /
  `_build_stroop_lapse_windows` are the two functions to extend if adding
  more lapse-style labels for other tasks.
- `dataset/windows.py`'s `RECOMMENDED_TRIAL_WINDOW_TYPE` and per-task window
  specs are the place to check before assuming a new label needs a new
  window kind vs. reusing an existing one.
- Nothing in this commit is destructive/final -- `export_npy.py`'s label
  scheme can still change without touching the canonical Parquet or the
  dataloader underneath it.
