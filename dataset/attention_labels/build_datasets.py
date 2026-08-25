"""CLI: assemble canonical SART/Stroop/Schulte tables across the whole
cohort, apply the participant-level split, and build ONLY the dataset
version(s) the caller asks for -- each version's sample table + manifest +
distribution report get written to `<out_dir>/<version_name>/`.

Building every version by default was more than most callers need (and
silently re-running six builders -- one of which fits a train-only RT-
residual model -- on every invocation makes it easy to lose track of what
you're actually looking at). Pass `--list-versions` to see what's
available, or `--all` to build everything at once.

Usage:
    python -m dataset.attention_labels.build_datasets \\
        --processed-dir data/processed --out-dir data/attention_datasets \\
        --list-versions

    python -m dataset.attention_labels.build_datasets \\
        --processed-dir data/processed --out-dir data/attention_datasets \\
        --version a2_sart_prospective_lapse a5_schulte_block
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

from dataset.attention_labels import qc
from dataset.attention_labels.versions import VERSION_BUILDERS, load_all_canonical_tables
from dataset.stress_labels.splits import split_by_participant

LABEL_PIPELINE_VERSION = "v1"


def discover_participants(processed_dir: Path) -> list[str]:
    return sorted(p.name for p in Path(processed_dir).iterdir() if p.is_dir() and (p / "attention_blocks.parquet").exists())


def _print_available_versions():
    print("Available dataset versions:")
    for name, builder in VERSION_BUILDERS.items():
        doc = (builder.__doc__ or "").strip().splitlines()[0] if builder.__doc__ else ""
        print(f"  {name:<28} {doc}")


def build_selected_versions(processed_dir: Path, out_dir: Path, version_names: list[str], seed: int = 42) -> dict:
    processed_dir, out_dir = Path(processed_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    participant_ids = discover_participants(processed_dir)
    split = split_by_participant(participant_ids, seed=seed)
    print(f"{len(participant_ids)} participants -- split: train={len(split.train)} val={len(split.val)} test={len(split.test)}")

    tables = load_all_canonical_tables(processed_dir, participant_ids)

    qc.sart_qc_report(tables["sart_trials"]).to_csv(out_dir / "sart_qc_report.csv", index=False)
    qc.stroop_qc_report(tables["stroop_trials"]).to_csv(out_dir / "stroop_qc_report.csv", index=False)
    qc.schulte_qc_report(tables["schulte_blocks"], tables["schulte_clicks"]).to_csv(out_dir / "schulte_qc_report.csv", index=False)

    report = {}
    for version_name in version_names:
        if version_name not in VERSION_BUILDERS:
            raise KeyError(f"unknown dataset version {version_name!r}; available: {sorted(VERSION_BUILDERS)}")
        builder = VERSION_BUILDERS[version_name]
        kwargs = {"split": split} if "split" in inspect.signature(builder).parameters else {}
        samples = builder(tables, **kwargs)

        version_dir = out_dir / version_name
        version_dir.mkdir(parents=True, exist_ok=True)
        samples.to_parquet(version_dir / "samples.parquet", index=False)

        target_col = "model_target" if "model_target" in samples.columns else "__none__"
        dist_report = qc.distribution_report(samples, target_col=target_col)

        manifest = {
            "dataset_version": version_name,
            "task": "sart" if "sart" in version_name else ("stroop" if "stroop" in version_name else "schulte"),
            "label_pipeline_version": LABEL_PIPELINE_VERSION,
            "sample_count": len(samples),
            "participants": split.to_dict(),
            "distribution": dist_report,
        }
        with open(version_dir / "manifest.json", "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, default=str)

        report[version_name] = len(samples)
        print(f"{version_name}: {len(samples)} samples -- {version_dir}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Build SART/Stroop/Schulte label Dataset Versions from canonical tables.")
    parser.add_argument("--processed-dir", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--version", type=str, nargs="+", default=None, choices=sorted(VERSION_BUILDERS), help="one or more dataset versions to build")
    parser.add_argument("--all", action="store_true", help="build every available version")
    parser.add_argument("--list-versions", action="store_true", help="print available dataset versions and exit")
    args = parser.parse_args()

    if args.list_versions:
        _print_available_versions()
        return

    if args.all:
        version_names = list(VERSION_BUILDERS)
    elif args.version:
        version_names = args.version
    else:
        print("No --version specified (and --all not passed) -- nothing to build.\n")
        _print_available_versions()
        raise SystemExit(1)

    build_selected_versions(Path(args.processed_dir), Path(args.out_dir), version_names, seed=args.seed)


if __name__ == "__main__":
    main()
