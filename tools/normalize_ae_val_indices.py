#!/usr/bin/env python
"""Normalize CARLA AE validation index files to three-column relative paths."""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, List, Optional, Sequence, Tuple


DATASET_ROOT = Path("/home/lgz/dataset/ADEC")


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    reference_val: Path
    ae_methods_root: Path


@dataclass(frozen=True)
class ReferenceSample:
    experiment: str
    image_name: str
    disparity_name: str


@dataclass(frozen=True)
class RewriteResult:
    index_file: Path
    line_count: int
    changed: bool
    backup_file: Optional[Path]


def parse_reference_samples(reference_val: Path) -> List[ReferenceSample]:
    """Read the canonical val split and keep experiment/sample ordering."""
    if not reference_val.is_file():
        raise FileNotFoundError(f"Reference file does not exist: {reference_val}")

    samples: List[ReferenceSample] = []
    with reference_val.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            parts = stripped.split()
            if len(parts) != 3:
                raise ValueError(
                    f"{reference_val}:{line_no} should have 3 columns, "
                    f"but got {len(parts)}."
                )

            left_path = PurePosixPath(parts[0])
            disparity_path = PurePosixPath(parts[2])
            if len(left_path.parts) < 4:
                raise ValueError(
                    f"{reference_val}:{line_no} has an invalid left path: "
                    f"{parts[0]}"
                )

            samples.append(
                ReferenceSample(
                    experiment=left_path.parts[-3],
                    image_name=left_path.name,
                    disparity_name=disparity_path.name,
                )
            )

    if not samples:
        raise ValueError(f"Reference file is empty: {reference_val}")
    return samples


def iter_index_files(ae_methods_root: Path) -> Iterable[Path]:
    if not ae_methods_root.is_dir():
        raise NotADirectoryError(f"AE methods root does not exist: {ae_methods_root}")

    for method_dir in sorted(ae_methods_root.iterdir()):
        if not method_dir.is_dir():
            continue
        index_file = method_dir / "val.txt"
        if index_file.is_file():
            yield index_file


def resolve_experiment_dir(method_dir: Path, experiment: str) -> Path:
    split_experiment_dir = method_dir / "val" / experiment
    if split_experiment_dir.is_dir():
        return split_experiment_dir

    direct_experiment_dir = method_dir / experiment
    if direct_experiment_dir.is_dir():
        return direct_experiment_dir

    raise FileNotFoundError(
        f"Cannot locate experiment directory for {method_dir.name}/{experiment}"
    )


def resolve_pair_dirs(experiment_dir: Path) -> Tuple[Path, Path]:
    candidates = (
        ("ldr_left", "ldr_right"),
        ("hdr_left", "hdr_right"),
    )
    for left_name, right_name in candidates:
        left_dir = experiment_dir / left_name
        right_dir = experiment_dir / right_name
        if left_dir.is_dir() and right_dir.is_dir():
            return left_dir, right_dir

    raise FileNotFoundError(
        f"Cannot find left/right image directories under {experiment_dir}"
    )


def resolve_image_file(image_dir: Path, reference_name: str) -> Path:
    stem = Path(reference_name).stem
    suffixes = (".png", Path(reference_name).suffix)

    for suffix in suffixes:
        candidate = image_dir / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate

    matches = sorted(image_dir.glob(f"{stem}.*"))
    if len(matches) == 1 and matches[0].is_file():
        return matches[0]

    raise FileNotFoundError(
        f"Cannot resolve image file for {reference_name} under {image_dir}"
    )


def build_index_lines(
    dataset_root: Path,
    method_dir: Path,
    samples: Sequence[ReferenceSample],
) -> List[str]:
    lines: List[str] = []

    for sample in samples:
        experiment_dir = resolve_experiment_dir(method_dir, sample.experiment)
        left_dir, right_dir = resolve_pair_dirs(experiment_dir)
        left_image = resolve_image_file(left_dir, sample.image_name)
        right_image = resolve_image_file(right_dir, sample.image_name)
        disparity = (
            experiment_dir
            / "ground_truth_disparity_left"
            / sample.disparity_name
        )

        if not disparity.is_file():
            raise FileNotFoundError(f"Disparity file does not exist: {disparity}")

        relative_paths = (
            left_image.relative_to(dataset_root).as_posix(),
            right_image.relative_to(dataset_root).as_posix(),
            disparity.relative_to(dataset_root).as_posix(),
        )
        lines.append(" ".join(relative_paths))

    return lines


def rewrite_index_file(
    dataset_root: Path,
    index_file: Path,
    samples: Sequence[ReferenceSample],
    dry_run: bool,
    backup: bool,
) -> RewriteResult:
    method_dir = index_file.parent
    new_text = "\n".join(build_index_lines(dataset_root, method_dir, samples)) + "\n"
    old_text = index_file.read_text(encoding="utf-8")
    changed = old_text != new_text
    backup_file = None

    if changed and not dry_run:
        if backup:
            backup_file = index_file.with_name(f"{index_file.name}.bak")
            if not backup_file.exists():
                shutil.copy2(index_file, backup_file)

        temp_file = index_file.with_name(f"{index_file.name}.tmp")
        temp_file.write_text(new_text, encoding="utf-8")
        temp_file.replace(index_file)

    return RewriteResult(
        index_file=index_file,
        line_count=len(samples),
        changed=changed,
        backup_file=backup_file,
    )


def normalize_dataset(
    config: DatasetConfig,
    dry_run: bool,
    backup: bool,
) -> List[RewriteResult]:
    samples = parse_reference_samples(config.reference_val)
    results: List[RewriteResult] = []

    for index_file in iter_index_files(config.ae_methods_root):
        results.append(
            rewrite_index_file(
                dataset_root=DATASET_ROOT,
                index_file=index_file,
                samples=samples,
                dry_run=dry_run,
                backup=backup,
            )
        )

    if not results:
        raise FileNotFoundError(f"No val.txt files found under {config.ae_methods_root}")
    return results


def build_configs(workspace_root: Path) -> List[DatasetConfig]:
    return [
        DatasetConfig(
            name="carla_600x800",
            reference_val=workspace_root / "dataset_split/carla_600x800/val.txt",
            ae_methods_root=DATASET_ROOT / "carla_600x800/ae_methods",
        ),
        DatasetConfig(
            name="carla_1280x384",
            reference_val=workspace_root / "dataset_split/carla_1280x384/val.txt",
            ae_methods_root=DATASET_ROOT / "carla_1280x384/ae_methods",
        ),
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize CARLA AE val.txt files to three-column relative paths."
    )
    parser.add_argument(
        "--workspace-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="OpenStereo workspace root.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and report changes without writing files.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create val.txt.bak before overwriting changed files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    all_results: List[RewriteResult] = []

    for config in build_configs(args.workspace_root):
        results = normalize_dataset(
            config=config,
            dry_run=args.dry_run,
            backup=not args.no_backup,
        )
        all_results.extend(results)
        changed_count = sum(result.changed for result in results)
        print(
            f"{config.name}: checked {len(results)} val.txt files, "
            f"{changed_count} need updates."
        )

        for result in results:
            action = "would update" if args.dry_run and result.changed else "updated"
            if not result.changed:
                action = "unchanged"
            print(f"  {action}: {result.index_file} ({result.line_count} lines)")

    total_changed = sum(result.changed for result in all_results)
    mode = "validated" if args.dry_run else "finished"
    print(f"{mode}: {len(all_results)} files checked, {total_changed} changed.")


if __name__ == "__main__":
    main()
