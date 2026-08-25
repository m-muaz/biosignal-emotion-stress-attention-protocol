"""Interactive whole-session EEG browser: raw vs. filtered, with every task
phase/trial/block/baseline/questionnaire boundary marked, using MNE's
built-in Raw viewer (scroll/zoom/pan, click-to-inspect annotations).

Unlike eeg_continuous_preprocess.py (which only loads the emotion task's
span), this loads the participant's ENTIRE recorded session so you can see
what the signal looks like across every phase before trusting any one
metric computed on a slice of it.

Usage:
    python -m analysis.eeg_session_viewer --pid P001
    python -m analysis.eeg_session_viewer --pid P017 --duration 60

Channels are grouped in two contiguous blocks -- all 8 *_raw channels,
then all 8 *_filt channels (not interleaved) -- so "all raw channels
together" / "all filtered channels together" is just the top half vs.
bottom half of the channel list; default --n-channels=8 shows one block
at a time, Page Up/Down flips between them.

Confirmed key bindings (Qt backend, from mne_qt_browser source):
    Left / Right           scroll time by 25% of the visible window
    Shift+Left / Shift+Right  scroll time by a full window (100%)
    Home / End              DEcrease / INcrease how many seconds are visible (time zoom)
    Page Up / Page Down     show one more / one fewer channel (Shift+PageUp/Down: by 10)
    - and + (or =)          decrease / increase amplitude scale (vertical zoom)
    click a channel's name label   mark/un-mark it "bad" (grays it out --
                                    the practical way to hide a channel)
    click-drag with the mouse      pyqtgraph's default box-zoom on the trace area
    scroll wheel over the traces   pyqtgraph's default zoom, same axis under the cursor
The bottom toolbar also has "Time (s)" / "Channels" spinboxes and a zoom
button doing the same thing with the mouse instead of the keyboard.
"""
import argparse

import numpy as np
import pandas as pd
import mne

from dataset.windows import extract_all_windows, load_events_parquet
from analysis.eeg_continuous_preprocess import (
    PROCESSED_DIR, CHANNELS, NATIVE_FS, UV_PER_CODE,
    resample_uniform, filter_continuous_uv,
)


def load_full_session_uv(pid: str):
    """Whole recorded session (every task, not just one), converted to uV,
    plus every window (trial/block/baseline/questionnaire/transition/...)
    across the whole event log for annotating."""
    events = load_events_parquet(PROCESSED_DIR / pid / "events.parquet")
    windows = extract_all_windows(events)

    df = pd.read_parquet(PROCESSED_DIR / pid / "ear_eeg_out.ads1299.parquet")
    sec = df.copy()
    sec[CHANNELS] = sec[CHANNELS].astype(np.float64) * UV_PER_CODE
    return sec, windows


def windows_to_annotations(windows, t0: float) -> mne.Annotations:
    """MNE colors annotations automatically, one color per unique
    description -- task/window_type(+condition) gives each phase its own
    color and label for free."""
    onsets, durations, descriptions = [], [], []
    for w in windows:
        onsets.append(w.t_start - t0)
        durations.append(max(w.t_end - w.t_start, 0.0))
        tag = w.condition_label or w.window_type
        descriptions.append(f"{w.task}/{w.window_type}:{tag}" if tag != w.window_type else f"{w.task}/{w.window_type}")
    return mne.Annotations(onset=onsets, duration=durations, description=descriptions)


def build_grouped_raw(sec_uniform: pd.DataFrame, filt: pd.DataFrame) -> mne.io.RawArray:
    """One Raw object, 16 channels in two contiguous blocks: ch1..8_raw
    then ch1..8_filt -- so "all raw together" / "all filtered together"
    is just scrolling to the top vs. bottom half of the channel list
    (MNE volts convention, so uV values are scaled by 1e-6)."""
    names = [f"{ch}_raw" for ch in CHANNELS] + [f"{ch}_filt" for ch in CHANNELS]
    cols = [sec_uniform[ch].to_numpy() for ch in CHANNELS] + [filt[ch].to_numpy() for ch in CHANNELS]
    data_v = np.stack(cols) * 1e-6
    info = mne.create_info(names, sfreq=NATIVE_FS, ch_types="eeg")
    return mne.io.RawArray(data_v, info)


def main():
    parser = argparse.ArgumentParser(description="Interactive raw-vs-filtered whole-session EEG browser.")
    parser.add_argument("--pid", required=True)
    parser.add_argument("--duration", type=float, default=30.0, help="initial visible window width (s)")
    parser.add_argument("--n-channels", type=int, default=8, help="channels shown at once (8 = one block: all-raw or all-filtered; 16 = both)")
    args = parser.parse_args()

    sec, windows = load_full_session_uv(args.pid)
    sec_uniform = resample_uniform(sec, NATIVE_FS)
    filt = filter_continuous_uv(sec_uniform)

    t0 = sec_uniform.wall_utc_s.iloc[0]
    raw = build_grouped_raw(sec_uniform, filt)
    raw.set_annotations(windows_to_annotations(windows, t0))

    n_min = sec_uniform.shape[0] / NATIVE_FS / 60
    print(f"{args.pid}: {n_min:.1f} min recorded, {len(windows)} annotated phase/trial/block windows")
    print("Channels 1-8 (top): raw. Channels 9-16 (bottom): filtered. Page Up/Down to flip between blocks (or --n-channels 16 to see both).")
    print("Time zoom: Home/End. Amplitude zoom: +/-. Scroll: Left/Right (Shift = full page). Click a channel name to hide it (marks it 'bad').")
    raw.plot(n_channels=args.n_channels, duration=args.duration, scalings="auto", block=True, group_by="original",
              title=f"{args.pid}: raw (top 8) vs. filtered (bottom 8)")


if __name__ == "__main__":
    main()
