"""Per-participant raw-file discovery + restart/opt-out resolution.

Walks one participant directory under
C:\\emotion-stress-attention-task-ear-wristband-device-data\\data-collection-sessions
and decides, for each device stream, which file(s) are authoritative --
never silently: every decision (and every excluded candidate) is recorded on
the returned ResolvedSession, and build_dataset.py writes it all out to
sync_report.json.

Resolution rules (see docs/Dataset_Sync_Design.md §2 for the full writeup):
  - Ear-EEG (out-ear/in-ear): prefer the file whose header actually has a
    nonzero sync field (`unix_time_at_sync`) -- an aborted/restarted boot
    never received CMD_SET_TIME so its header stays zero and it's excluded.
    If more than one candidate qualifies, prefer whichever's sync instant is
    closest to (and before) the participant's log-start reference timestamp
    (devices are synced, then started, then the app's own event log starts
    last -- the collection order).
  - Wristband: prefer the highest session index UNLESS overrides.yaml pins a
    specific session for this participant. If the chosen session has no
    valid forward sync anchor, apply overrides.yaml's documented
    sync_override (e.g. P009's end-anchor) if one exists, else exclude with
    a hard "cannot be time-aligned" result rather than guessing.
  - Polar H10: normally one file triplet; if more than one is found, same
    closest-to-log-start rule as ear-EEG, using each triplet's first
    checkpoint as its candidate sync instant.
  - In-ear onboard sensors (PPG/IMU/MLX90632/BME680) reuse whichever boot
    index the in-ear ADS1299 file resolved to -- firmware opens all of a
    boot's files together (confirmed via meta.txt's create-event ordering),
    so there's no independent restart risk for these to resolve separately.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from dataset.overrides import ParticipantOverrides, get_participant_overrides
from dataset.raw.ads1299_bin import peek_ads1299_header
from dataset.raw.events_jsonl import parse_events_jsonl
from dataset.raw.wristband_csv import parse_wristband_meta_csv

LOG_DIR_NAMES = ["data_collection_logs", "data-collection-logs"]
WRISTBAND_MODALITIES = ["ppg", "imu", "gsr", "mag", "mlx", "bme"]
LOG_START_GAP_WARN_S = 900.0  # sync-to-log-start gap outside [0, this] gets flagged


@dataclass
class ExcludedFile:
    path: Path
    reason: str


@dataclass
class DeviceResolution:
    device: str
    available: bool
    unavailable_reason: str = ""
    files: dict = field(default_factory=dict)  # role -> Path
    sync_anchor_wall_s: float | None = None
    sync_source: str = ""
    anchor_mode: str = "forward"  # "forward" | "backward_end" (P009-style override)
    gap_to_log_start_s: float | None = None
    confidence: str = "normal"  # "normal" | "reduced"
    warnings: list = field(default_factory=list)
    excluded: list = field(default_factory=list)  # list[ExcludedFile]


@dataclass
class ResolvedSession:
    participant_id: str
    participant_dir: Path
    log_start_reference_s: float | None
    main_session_dir: Path | None
    sart_session_dir: Path | None
    log_dir: Path | None
    devices: dict  # device name -> DeviceResolution
    global_warnings: list = field(default_factory=list)


# ── generic helpers ──────────────────────────────────────────────────────
def _iglob_ci(directory: Path, pattern: str) -> list[Path]:
    """Case-insensitive, non-recursive glob within `directory`."""
    if not directory or not directory.is_dir():
        return []
    regex = re.compile("^" + re.escape(pattern).replace(r"\*", ".*").replace(r"\?", ".") + "$", re.IGNORECASE)
    return sorted(p for p in directory.iterdir() if p.is_file() and regex.match(p.name))


def _extract_trailing_int(name: str, pattern: str) -> int:
    m = re.search(pattern, name, re.IGNORECASE)
    if not m:
        raise ValueError(f"can't extract index from filename {name!r} with pattern {pattern!r}")
    return int(m.group(1))


def _infer_participant_id(participant_dir: Path) -> str:
    m = re.match(r"part-([A-Za-z0-9]+)$", participant_dir.name, re.IGNORECASE)
    if not m:
        raise ValueError(f"can't infer participant id from folder name {participant_dir.name!r}")
    return m.group(1)


def _find_one_dir(participant_dir: Path, names: list[str]) -> Path | None:
    for name in names:
        p = participant_dir / name
        if p.is_dir():
            return p
    return None


def _find_session_subdirs(log_dir: Path | None, participant_id: str) -> tuple[Path | None, Path | None, list[str]]:
    warnings: list[str] = []
    if log_dir is None:
        return None, None, ["no data_collection_logs/data-collection-logs directory found"]

    # Naming convention has drifted: most participants' session dirs are
    # prefixed with their own folder name ("part-P007_<ts>" /
    # "part-P007_sart_<ts>"), but P010-P012 instead prefix with the
    # participant's own first name ("zijian_<ts>" / "zijian_sart_<ts>") --
    # the 3rd naming variant docs/Dataset_Sync_Design.md §6 anticipated.
    # Match structurally instead of hardcoding a prefix: any dir ending in
    # "_sart_<digits>" is the SART sub-session; any OTHER dir ending in
    # "_<digits>" is a main-session candidate. The original "part-<ID>_..."
    # convention is just a special case of "any prefix", so this covers both
    # uniformly without a per-participant override.
    main_pattern = re.compile(r"^.+_(\d+)$", re.IGNORECASE)
    sart_pattern = re.compile(r"^.+_sart_(\d+)$", re.IGNORECASE)
    subdirs = [p for p in log_dir.iterdir() if p.is_dir()]
    sarts = [p for p in subdirs if sart_pattern.match(p.name)]
    mains = [p for p in subdirs if main_pattern.match(p.name) and not sart_pattern.match(p.name)]

    main_dir = None
    if mains:
        main_dir = sorted(mains, key=lambda p: int(main_pattern.match(p.name).group(1)))[-1]
        if len(mains) > 1:
            warnings.append(f"multiple main session dirs found {sorted(p.name for p in mains)}; chose latest {main_dir.name}")
    else:
        warnings.append("no main session directory found")

    sart_dir = None
    if sarts:
        sart_dir = sorted(sarts, key=lambda p: int(sart_pattern.match(p.name).group(1)))[-1]
        if len(sarts) > 1:
            warnings.append(f"multiple SART session dirs found {sorted(p.name for p in sarts)}; chose latest {sart_dir.name}")

    return main_dir, sart_dir, warnings


def _opt_out_reason(device_dir: Path) -> str:
    for p in device_dir.iterdir():
        if not p.is_file():
            continue
        name = p.name.lower()
        if "opt" in name or "stopped" in name or "declined" in name:
            try:
                content = p.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception:
                content = ""
            return f"{p.name}" + (f": {content}" if content else " (empty marker file)")
    return "no A1299_*.bin file found and no opt-out marker file found either -- unexplained absence"


# ── ear-EEG (out-ear / in-ear) ───────────────────────────────────────────
def _resolve_ear_eeg(device_dir: Path, device_name: str, log_start_ref: float | None, has_onboard_sensors: bool) -> DeviceResolution:
    if not device_dir.is_dir():
        return DeviceResolution(device_name, available=False, unavailable_reason=f"{device_dir} does not exist")

    candidates = _iglob_ci(device_dir, "A1299_*.bin")
    if not candidates:
        return DeviceResolution(device_name, available=False, unavailable_reason=_opt_out_reason(device_dir))

    peeks = [peek_ads1299_header(p) for p in candidates]
    excluded: list[ExcludedFile] = []
    synced_peeks = []
    for peek in peeks:
        if not peek.magic_ok:
            excluded.append(ExcludedFile(peek.path, "bad magic bytes -- not a valid ADS1299 file"))
        elif not peek.synced:
            excluded.append(
                ExcludedFile(peek.path, "unix_time_at_sync == 0.0 -- this boot never received CMD_SET_TIME (aborted/restarted boot)")
            )
        else:
            synced_peeks.append(peek)

    if not synced_peeks:
        return DeviceResolution(device_name, available=False, unavailable_reason="no candidate file has a valid sync anchor", excluded=excluded)

    warnings: list[str] = []
    if len(synced_peeks) > 1:
        if log_start_ref is not None:
            chosen = min(synced_peeks, key=lambda pk: abs(pk.unix_time_at_sync - log_start_ref))
            warnings.append(
                f"{len(synced_peeks)} candidate files all had a valid sync anchor; chose {chosen.path.name} "
                f"(closest to the log-start reference timestamp)"
            )
        else:
            chosen = max(synced_peeks, key=lambda pk: pk.file_size_bytes)
            warnings.append(
                f"{len(synced_peeks)} candidate files all had a valid sync anchor and no log-start reference "
                f"was available; defaulted to {chosen.path.name} (largest file) -- verify manually"
            )
        for pk in synced_peeks:
            if pk.path != chosen.path:
                excluded.append(ExcludedFile(pk.path, "had a valid sync anchor but was not selected -- see warnings"))
    else:
        chosen = synced_peeks[0]

    gap = None
    if log_start_ref is not None:
        gap = log_start_ref - chosen.unix_time_at_sync
        if gap < 0 or gap > LOG_START_GAP_WARN_S:
            warnings.append(
                f"sync-to-log-start gap is {gap:.1f}s (expected roughly [0, {LOG_START_GAP_WARN_S:.0f}]s) -- "
                f"double-check {chosen.path.name} is really the file that belongs to this session"
            )

    index = _extract_trailing_int(chosen.path.name, r"_(\d+)\.")
    files = {"ads1299": chosen.path}
    if has_onboard_sensors:
        for role, prefix in [("ppg", "PPG"), ("imu", "IMU"), ("mlx90632", "MLX90632"), ("bme680", "BME680")]:
            sibling_candidates = _iglob_ci(device_dir, f"{prefix}_*.bin")
            sibling = next(
                (p for p in sibling_candidates if _extract_trailing_int(p.name, r"_(\d+)\.") == index), None
            )
            if sibling:
                files[role] = sibling
            else:
                warnings.append(f"expected sibling sensor file {prefix}_* (index {index}) not found next to {chosen.path.name}")

    return DeviceResolution(
        device_name,
        available=True,
        files=files,
        sync_anchor_wall_s=chosen.unix_time_at_sync,
        sync_source=f"{chosen.path.name} header (unix_time_at_sync)",
        gap_to_log_start_s=gap,
        warnings=warnings,
        excluded=excluded,
    )


# ── wristband ────────────────────────────────────────────────────────────
def _resolve_event_reference(log_dir: Path | None, session_dir_glob: str, event_type: str) -> float:
    if log_dir is None:
        raise ValueError("no data_collection_logs directory available to resolve the reference event from")
    matches = sorted(p for p in log_dir.glob(session_dir_glob) if p.is_dir())
    if not matches:
        raise ValueError(f"no session dir matching {session_dir_glob!r} under {log_dir}")
    session_dir = matches[-1]
    df = parse_events_jsonl(session_dir / "events.jsonl")
    rows = df[df["event_type"] == event_type]
    if rows.empty:
        raise ValueError(f"no {event_type!r} event found in {session_dir / 'events.jsonl'}")
    return float(rows["timestamp_host_utc"].iloc[-1])


def _resolve_wristband(
    device_dir: Path, log_start_ref: float | None, overrides: ParticipantOverrides, log_dir: Path | None
) -> DeviceResolution:
    if not device_dir.is_dir():
        return DeviceResolution("wristband", available=False, unavailable_reason=f"{device_dir} does not exist")

    modality_files = {mod: _iglob_ci(device_dir, f"{mod}-S*.csv") for mod in WRISTBAND_MODALITIES}
    session_indices = sorted(
        {_extract_trailing_int(p.name, r"-S(\d+)\.") for matches in modality_files.values() for p in matches}
    )
    if not session_indices:
        return DeviceResolution("wristband", available=False, unavailable_reason="no <modality>-S*.csv files found")

    warnings: list[str] = []
    override = overrides.wristband
    if override and override.use_session is not None:
        chosen_session = override.use_session
        warnings.append(f"session {chosen_session} forced by overrides.yaml ({override.reason.strip()})")
    else:
        chosen_session = max(session_indices)
        if len(session_indices) > 1:
            warnings.append(
                f"multiple wristband sessions found {session_indices}; no override documented -- "
                f"defaulted to highest ({chosen_session}), verify this is correct"
            )

    session_pattern = r"-S(\d+)\."
    excluded: list[ExcludedFile] = []
    for mod, matches in modality_files.items():
        for p in matches:
            p_session = _extract_trailing_int(p.name, session_pattern)
            if p_session != chosen_session:
                excluded.append(ExcludedFile(p, f"session {p_session} not selected (chose session {chosen_session})"))

    files = {}
    for mod, matches in modality_files.items():
        chosen_file = next((p for p in matches if _extract_trailing_int(p.name, r"-S(\d+)\.") == chosen_session), None)
        if chosen_file:
            files[mod] = chosen_file
        else:
            warnings.append(f"no {mod}-S{chosen_session:06d}.csv found for the chosen session")

    meta_matches = _iglob_ci(device_dir, "meta-S*.csv")
    meta_path = next((p for p in meta_matches if _extract_trailing_int(p.name, r"-S(\d+)\.") == chosen_session), None)

    sync_anchor_wall_s = None
    sync_source = ""
    anchor_mode = "forward"
    confidence = "normal"
    gap = None

    anchors = parse_wristband_meta_csv(meta_path) if meta_path else []
    if anchors:
        anchor = anchors[-1]
        sync_anchor_wall_s = anchor.computer_epoch_ms / 1000.0
        sync_source = f"{meta_path.name} row {anchor.sync_index} (forward SYNC_MS anchor)"
        files["meta"] = meta_path  # sync.py re-reads this for the exact (computer_epoch_ms, sync_device_us) pair
        if log_start_ref is not None:
            gap = log_start_ref - sync_anchor_wall_s
            if gap < 0 or gap > LOG_START_GAP_WARN_S:
                warnings.append(f"sync-to-log-start gap is {gap:.1f}s -- verify")
    elif override and override.end_anchor:
        ea = override.end_anchor
        try:
            ref_ts = _resolve_event_reference(log_dir, ea.session_dir_glob, ea.event_type)
            sync_anchor_wall_s = ref_ts
            anchor_mode = "backward_end"
            confidence = "reduced"
            sync_source = f"end-anchor override: {ea.event_type} @ {ref_ts:.3f} from {ea.session_dir_glob} (see overrides.yaml)"
            warnings.append(
                f"REDUCED CONFIDENCE: no computer-time sync ever reached this device for the chosen session; "
                f"using documented backward end-anchor instead. {ea.reason.strip()}"
            )
        except Exception as exc:
            warnings.append(f"end-anchor override is configured but failed to resolve: {exc}")
    else:
        warnings.append(
            f"no forward sync anchor available for session {chosen_session} (meta file empty/missing) and no "
            f"override documented in overrides.yaml -- this stream cannot be time-aligned"
        )

    return DeviceResolution(
        "wristband",
        available=bool(files) and sync_anchor_wall_s is not None,
        unavailable_reason="" if (files and sync_anchor_wall_s is not None) else "no usable sync anchor (see warnings)",
        files=files,
        sync_anchor_wall_s=sync_anchor_wall_s,
        sync_source=sync_source,
        anchor_mode=anchor_mode,
        gap_to_log_start_s=gap,
        confidence=confidence,
        warnings=warnings,
        excluded=excluded,
    )


# ── Polar H10 ────────────────────────────────────────────────────────────
def _first_checkpoint_wall_s(checkpoints_path: Path) -> float:
    df = pd.read_csv(checkpoints_path, nrows=1)
    return float(df["wall_ns"].iloc[0]) / 1e9


def _resolve_polar(device_dir: Path, log_start_ref: float | None) -> DeviceResolution:
    if not device_dir.is_dir():
        return DeviceResolution("polar_h10", available=False, unavailable_reason=f"{device_dir} does not exist")

    ecg_files = _iglob_ci(device_dir, "*_polar_ecg.csv")
    if not ecg_files:
        for sub in device_dir.iterdir():
            if sub.is_dir():
                ecg_files.extend(_iglob_ci(sub, "*_polar_ecg.csv"))
    if not ecg_files:
        return DeviceResolution(
            "polar_h10", available=False, unavailable_reason="no *_polar_ecg.csv found (checked device dir and one level of subfolders)"
        )

    excluded: list[ExcludedFile] = []
    scored = []
    for ecg_path in ecg_files:
        prefix = ecg_path.name[: -len("_polar_ecg.csv")]
        d = ecg_path.parent
        acc_path = d / f"{prefix}_polar_acc.csv"
        cps_path = d / f"{prefix}_polar_checkpoints.csv"
        if not cps_path.exists():
            excluded.append(ExcludedFile(ecg_path, "no matching _polar_checkpoints.csv found -- can't sync"))
            continue
        try:
            first_ts = _first_checkpoint_wall_s(cps_path)
        except Exception as exc:
            excluded.append(ExcludedFile(ecg_path, f"failed to read checkpoints file: {exc}"))
            continue
        scored.append((prefix, ecg_path, acc_path if acc_path.exists() else None, cps_path, first_ts))

    if not scored:
        return DeviceResolution("polar_h10", available=False, unavailable_reason="no candidate had a readable checkpoints file", excluded=excluded)

    warnings: list[str] = []
    if len(scored) > 1:
        if log_start_ref is not None:
            chosen = min(scored, key=lambda s: abs(s[4] - log_start_ref))
            warnings.append(f"{len(scored)} candidate Polar recordings found; chose {chosen[0]} (closest to log-start reference)")
        else:
            chosen = scored[0]
            warnings.append(f"{len(scored)} candidate Polar recordings found and no log-start reference available; defaulted to {chosen[0]} -- verify manually")
        for s in scored:
            if s[0] != chosen[0]:
                excluded.append(ExcludedFile(s[1], "another candidate recording was selected -- see warnings"))
    else:
        chosen = scored[0]

    prefix, ecg_path, acc_path, cps_path, first_ts = chosen
    gap = (log_start_ref - first_ts) if log_start_ref is not None else None
    if gap is not None and (gap < 0 or gap > LOG_START_GAP_WARN_S):
        warnings.append(f"sync-to-log-start gap is {gap:.1f}s -- verify")

    files = {"ecg": ecg_path, "checkpoints": cps_path}
    if acc_path:
        files["acc"] = acc_path
    else:
        warnings.append(f"no matching _polar_acc.csv found for {prefix} -- ACC stream unavailable for this participant")

    return DeviceResolution(
        "polar_h10",
        available=True,
        files=files,
        sync_anchor_wall_s=first_ts,
        sync_source=f"{cps_path.name} first checkpoint",
        gap_to_log_start_s=gap,
        warnings=warnings,
        excluded=excluded,
    )


# ── top-level entry point ────────────────────────────────────────────────
def resolve_participant(participant_dir: Path, overrides_path=None) -> ResolvedSession:
    participant_dir = Path(participant_dir)
    participant_id = _infer_participant_id(participant_dir)
    overrides = get_participant_overrides(participant_id) if overrides_path is None else get_participant_overrides(participant_id, overrides_path)

    log_dir = _find_one_dir(participant_dir, LOG_DIR_NAMES)
    main_session_dir, sart_session_dir, global_warnings = _find_session_subdirs(log_dir, participant_id)

    log_start_ref = None
    if main_session_dir is not None:
        try:
            df = parse_events_jsonl(main_session_dir / "events.jsonl")
            log_start_ref = float(df["timestamp_host_utc"].iloc[0])
        except Exception as exc:
            global_warnings.append(f"failed to read log-start reference from {main_session_dir}: {exc}")

    devices = {
        "ear_eeg_out": _resolve_ear_eeg(participant_dir / "out-ear", "ear_eeg_out", log_start_ref, has_onboard_sensors=False),
        "ear_eeg_in": _resolve_ear_eeg(participant_dir / "in-ear", "ear_eeg_in", log_start_ref, has_onboard_sensors=True),
        "wristband": _resolve_wristband(participant_dir / "wristband", log_start_ref, overrides, log_dir),
        "polar_h10": _resolve_polar(participant_dir / "polar-h10", log_start_ref),
    }

    return ResolvedSession(
        participant_id=participant_id,
        participant_dir=participant_dir,
        log_start_reference_s=log_start_ref,
        main_session_dir=main_session_dir,
        sart_session_dir=sart_session_dir,
        log_dir=log_dir,
        devices=devices,
        global_warnings=global_warnings,
    )
