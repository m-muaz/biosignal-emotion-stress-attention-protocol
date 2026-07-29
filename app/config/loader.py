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


def resolve_path(repo_root: str, relative_path: str) -> Path:
    return Path(repo_root) / relative_path
