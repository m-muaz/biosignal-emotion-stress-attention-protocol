"""Phase 6 (plans/attention_task.md §46): dataset variants are chosen by
EDITING/ADDING a YAML file under configs/attention_labels/, not by writing
new Python. A config picks: which Version-1-6 label builder to use (from
dataset.attention_labels.versions.VERSION_BUILDERS), the physiological
windowing mode/bounds to cut around each label row (Phase 4,
dataset.attention_labels.export_npy), and (for SART) the rolling-history
parameters. See configs/attention_labels/*.yaml for the six shipped
variants -- copy one and change pre_sec/post_sec/history to make a new
dataset version without touching any of the label/window code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "attention_labels"


@dataclass
class WindowSpec:
    mode: str  # "concurrent" | "stimulus_locked" | "prospective_pre_stimulus" | "pre_click" | "block"
    anchor_col: str | None = None
    pre_sec: float = 0.0
    post_sec: float = 0.0
    start_col: str | None = None  # "block" mode only
    end_col: str | None = None  # "block" mode only

    def __post_init__(self):
        if self.mode == "block":
            if not (self.start_col and self.end_col):
                raise ValueError("windowing.mode 'block' requires start_col and end_col")
        elif not self.anchor_col:
            raise ValueError(f"windowing.mode {self.mode!r} requires anchor_col")


@dataclass
class DatasetVersionConfig:
    name: str
    version_builder: str
    tasks: list[str]
    windowing: WindowSpec
    history: dict = field(default_factory=dict)
    requirements: dict = field(default_factory=dict)
    split_seed: int = 42
    source_path: Path | None = None


def load_version_config(name_or_path: str) -> DatasetVersionConfig:
    """`name_or_path` is either a bare config name ("a2_sart_prospective_lapse",
    resolved under CONFIG_DIR with a .yaml suffix) or an explicit path to a
    YAML file."""
    path = Path(name_or_path)
    if not path.suffix:
        path = CONFIG_DIR / f"{name_or_path}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"dataset version config not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    windowing = WindowSpec(**raw["windowing"])
    return DatasetVersionConfig(
        name=raw["name"],
        version_builder=raw.get("version_builder", raw["name"]),
        tasks=raw.get("tasks", []),
        windowing=windowing,
        history=raw.get("history", {}),
        requirements=raw.get("requirements", {}),
        split_seed=raw.get("split", {}).get("seed", 42),
        source_path=path,
    )


def discover_configs(config_dir: Path = CONFIG_DIR) -> list[str]:
    return sorted(p.stem for p in Path(config_dir).glob("*.yaml"))


def load_all_configs(config_dir: Path = CONFIG_DIR) -> dict[str, DatasetVersionConfig]:
    return {name: load_version_config(name) for name in discover_configs(config_dir)}
