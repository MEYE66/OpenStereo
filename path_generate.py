from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, List, Sequence


DEFAULT_DATA_ROOT = Path("/home/lgz/dataset/ADEC")
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "dataset_split"
DEFAULT_DATASETS = ("carla_600x800", "carla_1280x384")
DEFAULT_SPLITS = ("train", "val")
DEFAULT_METHOD_ROOT_NAME = "ae_methods"
DEFAULT_METHODS = ("gradient", "mixed", "pid", "semantic")

DATASET_ALIASES = {
	"carla600x800": "carla_600x800",
	"carla_600x800": "carla_600x800",
	"carla1280x384": "carla_1280x384",
	"carla_1280x384": "carla_1280x384",
}


def _normalize_dataset_name(name: str) -> str:
	normalized = DATASET_ALIASES.get(name.strip())
	if normalized is None:
		raise ValueError(f"Unsupported dataset name: {name}")
	return normalized


def _experiment_sort_key(path: Path) -> int:
	match = re.fullmatch(r"Experiment(\d+)", path.name)
	if match is None:
		raise ValueError(f"Unexpected experiment directory name: {path}")
	return int(match.group(1))


def _frame_sort_key(path: Path) -> int:
	if not path.stem.isdigit():
		raise ValueError(f"Unexpected frame file name: {path}")
	return int(path.stem)


def _iter_experiments(split_root: Path) -> Iterable[Path]:
	experiments = [path for path in split_root.iterdir() if path.is_dir() and path.name.startswith("Experiment")]
	yield from sorted(experiments, key=_experiment_sort_key)


def _build_entries_from_experiment_root(
	data_root: Path,
	experiment_root: Path,
	left_suffix: str,
	right_suffix: str,
) -> List[str]:
	entries: List[str] = []
	for experiment_dir in _iter_experiments(experiment_root):
		hdr_left_dir = experiment_dir / "hdr_left"
		hdr_right_dir = experiment_dir / "hdr_right"
		disp_dir = experiment_dir / "ground_truth_disparity_left"
		if not hdr_left_dir.is_dir() or not hdr_right_dir.is_dir() or not disp_dir.is_dir():
			raise FileNotFoundError(f"Incomplete experiment layout under: {experiment_dir}")

		left_files = sorted(hdr_left_dir.glob(f"*{left_suffix}"), key=_frame_sort_key)
		if not left_files:
			raise FileNotFoundError(f"No {left_suffix} files found under: {hdr_left_dir}")

		for left_path in left_files:
			frame_id = left_path.stem
			right_path = hdr_right_dir / f"{frame_id}{right_suffix}"
			disp_path = disp_dir / f"disparity_map_{frame_id}.npy"
			if not right_path.is_file():
				raise FileNotFoundError(f"Missing right image file: {right_path}")
			if not disp_path.is_file():
				raise FileNotFoundError(f"Missing disparity file: {disp_path}")

			entry = " ".join(
				[
					left_path.relative_to(data_root).as_posix(),
					right_path.relative_to(data_root).as_posix(),
					disp_path.relative_to(data_root).as_posix(),
				]
			)
			entries.append(entry)

	return entries


def _build_split_entries(data_root: Path, dataset_name: str, split_name: str) -> List[str]:
	split_root = data_root / dataset_name / split_name
	if not split_root.is_dir():
		raise FileNotFoundError(f"Missing split directory: {split_root}")
	return _build_entries_from_experiment_root(
		data_root=data_root,
		experiment_root=split_root,
		left_suffix=".hdr",
		right_suffix=".hdr",
	)


def _build_ae_method_entries(data_root: Path, dataset_name: str, method_root_name: str, method_name: str) -> List[str]:
	method_root = data_root / dataset_name / method_root_name / method_name
	if not method_root.is_dir():
		raise FileNotFoundError(f"Missing method directory: {method_root}")
	return _build_entries_from_experiment_root(
		data_root=data_root,
		experiment_root=method_root,
		left_suffix=".png",
		right_suffix=".png",
	)


def _write_split_file(entries: Sequence[str], output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	content = "\n".join(entries)
	if entries:
		content += "\n"
	output_path.write_text(content, encoding="utf-8")


def generate_dataset_split(data_root: Path, output_root: Path, dataset_name: str, split_name: str) -> Path:
	entries = _build_split_entries(data_root=data_root, dataset_name=dataset_name, split_name=split_name)
	output_path = output_root / dataset_name / f"{split_name}.txt"
	_write_split_file(entries, output_path)
	print(f"[{dataset_name}] {split_name}: wrote {len(entries)} entries -> {output_path}")
	return output_path


def generate_ae_method_split(
	data_root: Path,
	dataset_name: str,
	method_root_name: str,
	method_name: str,
	split_name: str = "val",
) -> Path:
	entries = _build_ae_method_entries(
		data_root=data_root,
		dataset_name=dataset_name,
		method_root_name=method_root_name,
		method_name=method_name,
	)
	output_path = data_root / dataset_name / method_root_name / method_name / f"{split_name}.txt"
	_write_split_file(entries, output_path)
	print(f"[{dataset_name}/{method_name}] {split_name}: wrote {len(entries)} entries -> {output_path}")
	return output_path


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Generate Carla dataset split txt files from the ADEC directory layout.")
	parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
	parser.add_argument("--layout", choices=("split", "ae_methods"), default="split")
	parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
	parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
	parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS), choices=list(DEFAULT_SPLITS))
	parser.add_argument("--method-root-name", default=DEFAULT_METHOD_ROOT_NAME)
	parser.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
	return parser


def main() -> None:
	args = build_parser().parse_args()
	dataset_names = [_normalize_dataset_name(name) for name in args.datasets]

	if args.layout == "split":
		for dataset_name in dataset_names:
			for split_name in args.splits:
				generate_dataset_split(
					data_root=args.data_root,
					output_root=args.output_root,
					dataset_name=dataset_name,
					split_name=split_name,
				)
		return

	for dataset_name in dataset_names:
		for method_name in args.methods:
			generate_ae_method_split(
				data_root=args.data_root,
				dataset_name=dataset_name,
				method_root_name=args.method_root_name,
				method_name=method_name,
			)


if __name__ == "__main__":
	main()
