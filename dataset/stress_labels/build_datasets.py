"""CLI: assemble canonical stress-task tables across the whole cohort, apply
the participant-level split, and build ONLY the dataset version(s) the
caller asks for -- each version's sample table + manifest + distribution
report get written to `<out_dir>/<version_name>/`.

Building every version by default was more than most callers need. Pass
`--list-versions` to see what's available, or `--all` to build everything
at once (the old default behavior).

Usage:
    python -m dataset.stress_labels.build_datasets \\
        --processed-dir data/processed --out-dir data/stress_datasets \\
        --list-versions

    python -m dataset.stress_labels.build_datasets \\
        --processed-dir data/processed --out-dir data/stress_datasets \\
        --version s4_raindrop_behavior s5_highway_behavior

Or build from a YAML config under configs/stress_labels/ (Phase 6 -- lets a
colleague pick a dataset variant without knowing any Python/CLI flags,
mirroring dataset.attention_labels's config-driven workflow):

    python -m dataset.stress_labels.build_datasets \\
        --processed-dir data/processed --out-dir data/stress_datasets \\
        --config s5_highway_behavior
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dataset.stress_labels import qc
from dataset.stress_labels.config import discover_configs, load_version_config
from dataset.stress_labels.splits import split_by_participant
from dataset.stress_labels.versions import VERSION_BUILDERS, load_all_canonical_tables

LABEL_PIPELINE_VERSION = "v1"


def discover_participants(processed_dir: Path) -> list[str]:
    return sorted(p.name for p in Path(processed_dir).iterdir() if p.is_dir() and (p / "stress_blocks.parquet").exists())


def _print_available_versions():
    print("Available dataset versions:")
    for name, builder in VERSION_BUILDERS.items():
        doc = (builder.__doc__ or "").strip().splitlines()[0] if builder.__doc__ else ""
        print(f"  {name:<32} {doc}")


def build_selected_versions(
    processed_dir: Path,
    out_dir: Path,
    version_names: list[str],
    seed: int = 42,
    exclude_highway_pre_fix: bool = False,
    per_version_exclude_highway_pre_fix: dict[str, bool] | None = None,
) -> dict:
    """`per_version_exclude_highway_pre_fix` (if given) overrides the global
    `exclude_highway_pre_fix` flag per version name -- used by
    build_selected_configs so each YAML config's own setting wins."""
    processed_dir, out_dir = Path(processed_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    per_version_exclude_highway_pre_fix = per_version_exclude_highway_pre_fix or {}

    participant_ids = discover_participants(processed_dir)
    split = split_by_participant(participant_ids, seed=seed)
    print(f"{len(participant_ids)} participants -- split: train={len(split.train)} val={len(split.val)} test={len(split.test)}")

    tables = load_all_canonical_tables(processed_dir, participant_ids)

    # sanity checks required before any modeling (plan §49, §51's raindrop/highway
    # QC tables) -- written once for the whole cohort, independent of any one version
    qc.raindrop_qc_report(tables["raindrop_trials"]).to_csv(out_dir / "raindrop_qc_report.csv", index=False)
    qc.raindrop_wrong_submission_counts(tables["raindrop_unmatched_events"]).to_csv(out_dir / "raindrop_wrong_submission_counts.csv", index=False)
    qc.highway_qc_report(tables["highway_obstacles"], tables["highway_self_report"]).to_csv(out_dir / "highway_qc_report.csv", index=False)

    report = {}
    for version_name in version_names:
        if version_name not in VERSION_BUILDERS:
            raise KeyError(f"unknown dataset version {version_name!r}; available: {sorted(VERSION_BUILDERS)}")
        builder = VERSION_BUILDERS[version_name]
        effective_exclude = per_version_exclude_highway_pre_fix.get(version_name, exclude_highway_pre_fix)
        kwargs = {"exclude_highway_pre_fix": effective_exclude} if "highway" in version_name else {}
        try:
            samples = builder(tables, **kwargs) if kwargs else builder(tables)
        except TypeError:
            samples = builder(tables)

        version_dir = out_dir / version_name
        version_dir.mkdir(parents=True, exist_ok=True)
        samples.to_parquet(version_dir / "samples.parquet", index=False)

        target_col = "model_target" if "model_target" in samples.columns else None
        dist_report = qc.distribution_report(samples, target_col=target_col) if target_col else qc.distribution_report(samples, target_col="__none__")

        manifest = {
            "dataset_version": version_name,
            "label_pipeline_version": LABEL_PIPELINE_VERSION,
            "sample_count": len(samples),
            "participants": split.to_dict(),
            "protocol_filter": {"exclude_highway_pre_fix": effective_exclude},
            "distribution": dist_report,
        }
        with open(version_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)

        report[version_name] = len(samples)
        print(f"{version_name}: {len(samples)} samples -- {version_dir}")

    return report


def build_selected_configs(processed_dir: Path, out_dir: Path, config_names: list[str], seed_override: int | None = None) -> dict:
    """Same as build_selected_versions, but reads each version's settings
    (currently just exclude_highway_pre_fix) from its YAML config under
    configs/stress_labels/ instead of CLI flags."""
    configs = [load_version_config(name) for name in config_names]
    version_names = [c.version_builder for c in configs]
    per_version_exclude = {c.version_builder: c.exclude_highway_pre_fix for c in configs}
    seed = seed_override if seed_override is not None else configs[0].split_seed
    return build_selected_versions(processed_dir, out_dir, version_names, seed=seed, per_version_exclude_highway_pre_fix=per_version_exclude)


def _print_available_configs():
    print("Available configs (configs/stress_labels/*.yaml):")
    for name in discover_configs():
        print(f"  {name}")


def main():
    parser = argparse.ArgumentParser(description="Build stress-task label Dataset Versions from canonical tables.")
    parser.add_argument("--processed-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None, help="overrides each config's own split.seed (or the default 42 when using --version/--all)")
    parser.add_argument("--exclude-highway-pre-fix", action="store_true", help="exclude the 4 pre-fix highway participants from highway-behavior-dependent versions (--version/--all only; use the YAML field with --config)")
    parser.add_argument("--version", type=str, nargs="+", default=None, choices=sorted(VERSION_BUILDERS), help="one or more dataset versions to build")
    parser.add_argument("--config", type=str, nargs="+", default=None, choices=discover_configs(), help="one or more configs/stress_labels/*.yaml names to build from (Phase 6)")
    parser.add_argument("--all", action="store_true", help="build every available version")
    parser.add_argument("--list-versions", action="store_true", help="print available dataset versions and exit")
    parser.add_argument("--list-configs", action="store_true", help="print available configs/stress_labels/*.yaml names and exit")
    args = parser.parse_args()

    if args.list_versions:
        _print_available_versions()
        return
    if args.list_configs:
        _print_available_configs()
        return

    if args.config:
        build_selected_configs(Path(args.processed_dir), Path(args.out_dir), args.config, seed_override=args.seed)
        return

    if args.all:
        version_names = list(VERSION_BUILDERS)
    elif args.version:
        version_names = args.version
    else:
        print("No --version/--config specified (and --all not passed) -- nothing to build.\n")
        _print_available_versions()
        raise SystemExit(1)

    build_selected_versions(Path(args.processed_dir), Path(args.out_dir), version_names, seed=args.seed if args.seed is not None else 42, exclude_highway_pre_fix=args.exclude_highway_pre_fix)


if __name__ == "__main__":
    main()
