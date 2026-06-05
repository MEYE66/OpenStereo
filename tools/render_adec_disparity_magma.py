#!/usr/bin/env python3
"""Render ADEC disparity .npy files as MAGMA-colored PNG images."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np


DEFAULT_DATASETS = ("carla_600x800", "carla_1280x384")
DEFAULT_DISPARITY_ROOT = Path("output_disparity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Regenerate ADEC disparity PNGs with cv2.COLORMAP_MAGMA."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset folders under --disparity-root.",
    )
    parser.add_argument(
        "--disparity-root",
        type=Path,
        default=DEFAULT_DISPARITY_ROOT,
        help="Root containing carla_*/adec disparity outputs.",
    )
    parser.add_argument(
        "--clip-percentile",
        type=float,
        default=None,
        help="Optional high percentile for robust visualization, e.g. 99.0.",
    )
    return parser.parse_args()


def find_disparity_files(disparity_root: Path, datasets: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for dataset in datasets:
        adec_dir = disparity_root / dataset / "adec"
        if not adec_dir.is_dir():
            raise FileNotFoundError(f"ADEC disparity directory not found: {adec_dir}")
        files.extend(sorted(adec_dir.glob("Experiment*/disparity_map_*.npy")))
    if not files:
        raise FileNotFoundError("No ADEC disparity .npy files found.")
    return files


def normalize_to_uint8(disparity: np.ndarray, clip_percentile: float | None) -> np.ndarray:
    if disparity.ndim != 2:
        raise ValueError(f"Expected a 2D disparity array, got shape {disparity.shape}")

    disp = disparity.astype(np.float32, copy=False)
    finite = np.isfinite(disp)
    if not finite.any():
        return np.zeros(disp.shape, dtype=np.uint8)

    valid_values = disp[finite]
    low = float(valid_values.min())
    high = float(valid_values.max())
    if clip_percentile is not None:
        if not 0.0 < clip_percentile <= 100.0:
            raise ValueError("--clip-percentile must be in (0, 100].")
        high = float(np.percentile(valid_values, clip_percentile))

    if high <= low:
        return np.zeros(disp.shape, dtype=np.uint8)

    clipped = np.clip(disp, low, high)
    scaled = (clipped - low) / (high - low)
    scaled[~finite] = 0.0
    return (scaled * 255.0).round().astype(np.uint8)


def render_magma_png(npy_path: Path, clip_percentile: float | None) -> Path:
    disparity = np.load(npy_path)
    gray = normalize_to_uint8(disparity, clip_percentile)
    colored = cv2.applyColorMap(gray, cv2.COLORMAP_MAGMA)
    png_path = npy_path.with_suffix(".png")
    if not cv2.imwrite(str(png_path), colored):
        raise IOError(f"Failed to write PNG: {png_path}")
    return png_path


def main() -> int:
    args = parse_args()
    files = find_disparity_files(args.disparity_root, args.datasets)

    rendered = 0
    for npy_path in files:
        render_magma_png(npy_path, args.clip_percentile)
        rendered += 1

    print(f"Rendered {rendered} MAGMA disparity PNG files.")
    for dataset in args.datasets:
        count = len(list((args.disparity_root / dataset / "adec").glob("Experiment*/disparity_map_*.png")))
        print(f"{dataset}: {count} PNG files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
