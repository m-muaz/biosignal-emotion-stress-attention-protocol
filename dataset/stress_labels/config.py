"""Dataset variants are chosen by EDITING/ADDING a YAML file under
configs/stress_labels/, mirroring dataset.attention_labels.config's Phase 6
pattern -- one file per Version, naming which VERSION_BUILDERS entry to use
and any builder kwargs (currently just exclude_highway_pre_fix). Unlike the
attention task, stress-task windows are the block/trial spans already
present as columns on each canonical table (block_start/block_end,
trial_start/response_timestamp, ...) -- auto-detected by
dataset.stress_labels.export_npy's WINDOW_BOUND_CANDIDATES, so there's no
separate windowing spec to configure here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "stress_labels"


@dataclass
class DatasetVersionConfig:
    name: str
    version_builder: str
    task: str = ""
    exclude_highway_pre_fix: bool = False
    split_seed: int = 42
    source_path: Path | None = None


def load_version_config(name_or_path: str) -> DatasetVersionConfig:
    """`name_or_path` is either a bare config name ("s4_raindrop_behavior",
    resolved under CONFIG_DIR with a .yaml suffix) or an explicit path."""
    path = Path(name_or_path)
    if not path.suffix:
        path = CONFIG_DIR / f"{name_or_path}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"dataset version config not found: {path}")

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return DatasetVersionConfig(
        name=raw["name"],
        version_builder=raw.get("version_builder", raw["name"]),
        task=raw.get("task", ""),
        exclude_highway_pre_fix=raw.get("exclude_highway_pre_fix", False),
        split_seed=raw.get("split", {}).get("seed", 42),
        source_path=path,
    )


def discover_configs(config_dir: Path = CONFIG_DIR) -> list[str]:
    return sorted(p.stem for p in Path(config_dir).glob("*.yaml"))


def load_all_configs(config_dir: Path = CONFIG_DIR) -> dict[str, DatasetVersionConfig]:
    return {name: load_version_config(name) for name in discover_configs(config_dir)}
