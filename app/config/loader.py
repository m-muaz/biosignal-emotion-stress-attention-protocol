import json
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_emotion_manifest(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest["clips"]


def load_fixed_clips(path: str) -> dict[str, list[str]]:
    """Predefined per-valence-group clip_id lists (see
    scripts/select_fixed_emotion_clips.py), used when
    emotion_task.clip_selection_mode == "fixed"."""
    with open(path, "r", encoding="utf-8") as f:
        selection = json.load(f)
    return {
        valence_group: [entry["clip_id"] for entry in entries]
        for valence_group, entries in selection.items()
        if valence_group != "_comment"
    }


def resolve_path(repo_root: str, relative_path: str) -> Path:
    return Path(repo_root) / relative_path
