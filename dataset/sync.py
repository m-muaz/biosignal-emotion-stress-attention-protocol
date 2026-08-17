"""Applies each device's resolved sync anchor to its raw samples, producing
absolute host-UTC-second timestamps alongside the raw values.

This is the one place every anchor formula from
docs/Dataset_Sync_Design.md §1 actually gets applied:
  - ear-EEG (ads1299 + in-ear onboard sensors): single fixed offset from the
    file's own header, wall_s = record_time + offset.
  - wristband (forward case): single fixed offset from meta-S*.csv's
    (computer_epoch_ms, sync_device_us) anchor pair.
  - wristband (P009-style backward_end override): same linear-offset math,
    just anchored at EACH file's own last device_us row instead of a shared
    meta-file row -- see overrides.yaml for why.
  - Polar H10: full-session LinearClockFit (offset + drift rate), refit from
    every checkpoint in *_polar_checkpoints.csv.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dataset.raw.ads1299_bin import parse_ads1299_bin
from dataset.raw.events_jsonl import parse_events_jsonl
from dataset.raw.polar_csv import parse_polar_acc_csv, parse_polar_checkpoints_csv, parse_polar_ecg_csv, refit_full_session
from dataset.raw.polar_csv import wall_utc_seconds as _polar_wall_utc_seconds
from dataset.raw.sbn1_bin import parse_sbn1_bin
from dataset.raw.wristband_csv import parse_wristband_meta_csv, parse_wristband_modality_csv
from dataset.session_resolver import DeviceResolution, ResolvedSession


@dataclass
class SyncedStream:
    name: str  # e.g. "ear_eeg_out.ads1299", "wristband.ppg", "polar_h10.ecg"
    sample_rate_hz: float
    wall_utc_s: np.ndarray
    data: pd.DataFrame  # value columns, aligned row-for-row with wall_utc_s
    qc: dict  # stream-specific QC info (crc_ok_fraction, drift ppm, etc.)


def sync_ear_eeg(resolution: DeviceResolution, validate_crc: bool = False) -> dict[str, SyncedStream]:
    """role -> SyncedStream. `ads1299` always, plus `ppg`/`imu`/`mlx90632`/
    `bme680` for the in-ear device when those sibling files were resolved."""
    if not resolution.available or "ads1299" not in resolution.files:
        return {}

    out: dict[str, SyncedStream] = {}
    ads = parse_ads1299_bin(resolution.files["ads1299"], validate_crc=validate_crc)
    if ads.synced:
        wall_s = ads.wall_utc_seconds()
        channels_df = pd.DataFrame(ads.channels, columns=[f"ch{i + 1}" for i in range(ads.channels.shape[1])])
        channels_df["counter"] = ads.counter
        out["ads1299"] = SyncedStream(
            f"{resolution.device}.ads1299", 250.0, wall_s, channels_df, qc={"crc_ok_fraction": ads.crc_ok_fraction}
        )

    for role in ("ppg", "imu", "mlx90632", "bme680"):
        if role not in resolution.files:
            continue
        sbn = parse_sbn1_bin(resolution.files[role])
        if not sbn.synced:
            continue
        wall_s = sbn.wall_utc_seconds()
        data = pd.DataFrame({k: v for k, v in sbn.fields.items() if k != "uptime_s"})
        out[role] = SyncedStream(f"{resolution.device}.{role}", float(sbn.sample_rate_hz), wall_s, data, qc={})

    return out


def sync_wristband(resolution: DeviceResolution) -> dict[str, SyncedStream]:
    if not resolution.available:
        return {}

    modality_rates = {"ppg": 200.0, "imu": 200.0, "gsr": 200.0, "mag": 100.0, "mlx": 1.0, "bme": 1.0}
    out: dict[str, SyncedStream] = {}

    forward_anchor = None
    if resolution.anchor_mode == "forward":
        if "meta" not in resolution.files:
            raise ValueError(
                f"wristband resolution for {resolution.device} claims anchor_mode='forward' but no meta file "
                f"was resolved -- this is a bug in session_resolver, not a data problem."
            )
        anchors = parse_wristband_meta_csv(resolution.files["meta"])
        if not anchors:
            raise ValueError(f"{resolution.files['meta']}: expected >=1 sync row for a 'forward' anchor_mode resolution")
        forward_anchor = anchors[-1]

    for mod, path in resolution.files.items():
        if mod == "meta":
            continue
        modf = parse_wristband_modality_csv(path, mod)
        if modf.device_us.size == 0:
            continue

        if resolution.anchor_mode == "forward":
            wall_s = forward_anchor.computer_epoch_ms / 1000.0 + (
                modf.device_us.astype(np.float64) - forward_anchor.sync_device_us
            ) / 1e6
        elif resolution.anchor_mode == "backward_end":
            # Anchor THIS file's own last row to the reference wall-clock
            # instant -- see overrides.yaml's end_anchor documentation for
            # why this must be per-file, not one shared value.
            last_device_us = int(modf.device_us[-1])
            offset_s = resolution.sync_anchor_wall_s - last_device_us / 1e6
            wall_s = offset_s + modf.device_us.astype(np.float64) / 1e6
        else:
            raise ValueError(f"unknown anchor_mode {resolution.anchor_mode!r}")

        out[mod] = SyncedStream(
            f"wristband.{mod}",
            modality_rates[mod],
            wall_s,
            modf.data,
            qc={"confidence": resolution.confidence},
        )

    return out


def sync_polar(resolution: DeviceResolution) -> dict[str, SyncedStream]:
    if not resolution.available or "checkpoints" not in resolution.files:
        return {}

    checkpoints = parse_polar_checkpoints_csv(resolution.files["checkpoints"])
    fit, qc = refit_full_session(checkpoints)
    qc_dict = {
        "n_checkpoints": qc.n_checkpoints,
        "rate_ppm": qc.rate_ppm,
        "max_residual_ms": qc.max_residual_ms,
        "mean_residual_ms": qc.mean_residual_ms,
    }

    out: dict[str, SyncedStream] = {}
    if "ecg" in resolution.files:
        ecg = parse_polar_ecg_csv(resolution.files["ecg"])
        wall_s = _polar_wall_utc_seconds(ecg, fit)
        out["ecg"] = SyncedStream("polar_h10.ecg", 130.0, wall_s, ecg.values, qc=qc_dict)
    if "acc" in resolution.files:
        acc = parse_polar_acc_csv(resolution.files["acc"])
        wall_s = _polar_wall_utc_seconds(acc, fit)
        out["acc"] = SyncedStream("polar_h10.acc", 200.0, wall_s, acc.values, qc=qc_dict)

    return out


def sync_events(resolved: ResolvedSession) -> pd.DataFrame:
    """Concatenates the main session's and (if present) SART sub-session's
    events.jsonl. Both are already host-UTC -- no transform needed, just a
    `session_part` column added so downstream code can tell them apart."""
    frames = []
    if resolved.main_session_dir is not None:
        df = parse_events_jsonl(resolved.main_session_dir / "events.jsonl")
        df = df.assign(session_part="main")
        frames.append(df)
    if resolved.sart_session_dir is not None:
        df = parse_events_jsonl(resolved.sart_session_dir / "events.jsonl")
        df = df.assign(session_part="sart")
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).sort_values("timestamp_host_utc", kind="stable").reset_index(drop=True)


def sync_all(resolved: ResolvedSession, validate_crc: bool = False) -> dict[str, dict[str, SyncedStream]]:
    """Top-level entry point: device name -> role -> SyncedStream, plus the
    special "events" key -> a DataFrame (not a SyncedStream, it has no fixed rate)."""
    return {
        "ear_eeg_out": sync_ear_eeg(resolved.devices["ear_eeg_out"], validate_crc=validate_crc),
        "ear_eeg_in": sync_ear_eeg(resolved.devices["ear_eeg_in"], validate_crc=validate_crc),
        "wristband": sync_wristband(resolved.devices["wristband"]),
        "polar_h10": sync_polar(resolved.devices["polar_h10"]),
        "events": sync_events(resolved),
    }
