import os
from pathlib import Path
import sys

import cv2

if __package__ in {None, ""}:
	sys.path.append(str(Path(__file__).resolve().parent.parent))

from vis_tools.vis_real import process_left, process_right


SOURCE_ROOT = Path("/home/lgz/dataset/ADEC/real/train")
DEST_ROOT = Path("/home/lgz/dataset/ADEC/real/val_vis")


def iter_stereo_dirs(root: Path):
	for current_dir, _, files in os.walk(root):
		file_set = set(files)
		if "left.npy" in file_set and "right.npy" in file_set:
			yield Path(current_dir)


def process_directory(source_dir: Path, source_root: Path, dest_root: Path):
	relative_dir = source_dir.relative_to(source_root)
	dest_dir = dest_root / relative_dir
	dest_dir.mkdir(parents=True, exist_ok=True)

	left_image = process_left(str(source_dir / "left.npy"))
	right_image = process_right(str(source_dir / "right.npy"))

	cv2.imwrite(str(dest_dir / "left.png"), cv2.cvtColor(left_image, cv2.COLOR_RGB2BGR))
	cv2.imwrite(str(dest_dir / "right.png"), cv2.cvtColor(right_image, cv2.COLOR_RGB2BGR))


def main(source_root: Path = SOURCE_ROOT, dest_root: Path = DEST_ROOT):
	source_root = Path(source_root)
	dest_root = Path(dest_root)

	if not source_root.exists():
		raise FileNotFoundError(f"Source directory does not exist: {source_root}")

	processed_count = 0
	for source_dir in iter_stereo_dirs(source_root):
		process_directory(source_dir, source_root, dest_root)
		processed_count += 1
		print(f"Processed {source_dir}")

	print(f"Finished processing {processed_count} directories into {dest_root}")


if __name__ == '__main__':
	main()
