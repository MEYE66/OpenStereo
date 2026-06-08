# Created: 2025-06-03
# Refactored: 2026-04-10

import argparse
import multiprocessing
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
import sys

import cv2
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from optimizer.nm_optimizer import NelderMeadOptimizer
from optimizer.pid_optimizer import PIDOptimizer
from optimizer.utils import (
    ImageContrastMetric,
    ImageEntropyMetric,
    ImageFormationModel,
    ImageGradientMetric,
    ImageSemanticMetric,
    MixedImageMetric,
    minmax_norm,
    np_to_image,
    inverse_tmo,
    radiance_scale
)


def natural_sort_key(name: str):
    import re

    parts = re.split(r"(\d+)", name)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def list_hdr_files(folder: Path) -> Dict[str, Path]:
    exts = [".hdr", ".exr", ".HDR", ".EXR"]
    files: Dict[str, Path] = {}
    for ext in exts:
        for p in folder.glob(f"*{ext}"):
            files[p.stem] = p
    return files


def resolve_disp_left_path(exp_dir: Path, frame_id: str) -> Optional[Path]:
    disp_dir = exp_dir / "ground_truth_disparity_left"
    if not disp_dir.exists():
        return None
    candidates = [
        disp_dir / f"disparity_map_{frame_id}.npy",
        disp_dir / f"{frame_id}.npy",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def collect_records(exp_dir: Path) -> List[Tuple[str, Path, Path, Optional[Path]]]:
    left_dir = exp_dir / "hdr_left"
    right_dir = exp_dir / "hdr_right"
    if not left_dir.exists() or not right_dir.exists():
        return []

    left_map = list_hdr_files(left_dir)
    right_map = list_hdr_files(right_dir)
    frame_ids = sorted(set(left_map.keys()) & set(right_map.keys()), key=natural_sort_key)

    records: List[Tuple[str, Path, Path, Optional[Path]]] = []
    for frame_id in frame_ids:
        disp_left = resolve_disp_left_path(exp_dir, frame_id)
        records.append((frame_id, left_map[frame_id], right_map[frame_id], disp_left))
    return records


def build_metric(mode: str):
    mode = mode.lower()
    if mode == "pid":
        return ImageEntropyMetric
    if mode == "gradient":
        return ImageGradientMetric
    if mode == "mixed":
        return MixedImageMetric
    if mode == "semantic":
        return ImageSemanticMetric
    raise ValueError(f"Unsupported mode: {mode}")


def build_optimizer(mode: str):
    if mode.lower() == "pid":
        return PIDOptimizer
    return NelderMeadOptimizer


def make_optimizer(optimizer_cls, metric_cls, args):
    image_model = ImageFormationModel(nbits=args.nbits, seed=args.seed)
    metric_model = metric_cls()

    bounds = [(args.exp_min, args.exp_max), (args.gain_min, args.gain_max)]
    if optimizer_cls is PIDOptimizer:
        optimizer = optimizer_cls(
            image_model,
            metric_model,
            bounds=bounds,
            setpoint=args.pid_setpoint,
            max_iter=args.max_iters,
            tol=args.tol,
        )
    else:
        optimizer = optimizer_cls(
            image_model,
            metric_model,
            bounds=bounds,
            max_iter=args.max_iters,
            tol=args.tol,
        )
    return image_model, optimizer


def relative_source_path(src_path: Path, source_root: Path) -> str:
    # source_root is expected like .../ADEC/aaa/val. We want aaa/val/...
    try:
        base = source_root.parents[1]
        return src_path.relative_to(base).as_posix()
    except Exception:
        return src_path.as_posix()


def process_one_frame(
    frame_id: str,
    left_path: Path,
    right_path: Path,
    disp_left_path: Optional[Path],
    exp_out: Path,
    mode: str,
    source_root: Path,
    args,
):
    metric_cls = build_metric(mode)
    optimizer_cls = build_optimizer(mode)
    image_model, optimizer = make_optimizer(optimizer_cls, metric_cls, args)

    left_hdr = cv2.imread(str(left_path), cv2.IMREAD_UNCHANGED)
    right_hdr = cv2.imread(str(right_path), cv2.IMREAD_UNCHANGED)
    if left_hdr is None or right_hdr is None:
        raise FileNotFoundError(f"Failed to read HDR pair: {left_path}, {right_path}")
    

    left_hdr = inverse_tmo(minmax_norm(left_hdr))
    right_hdr = inverse_tmo(minmax_norm(right_hdr))

    left_hdr = radiance_scale(minmax_norm(left_hdr), capacity=1.)
    right_hdr = radiance_scale(minmax_norm(right_hdr), capacity=1.)

    # left_hdr = np.clip(left_hdr.astype(np.float32), 0.0, None)
    # right_hdr = np.clip(right_hdr.astype(np.float32), 0.0, None)

    left_opt = left_hdr
    if args.downsample_factor > 1:
        h, w = left_hdr.shape[:2]
        nh = max(1, h // args.downsample_factor)
        nw = max(1, w // args.downsample_factor)
        left_opt = cv2.resize(left_hdr, (nw, nh), interpolation=cv2.INTER_CUBIC)

    best_exp, best_gain = optimizer.optimize(image=left_opt)
    best_exp = float(np.clip(best_exp, args.exp_min, args.exp_max))
    best_gain = float(np.clip(best_gain, args.gain_min, args.gain_max))

    left_ldr = image_model.simulate(left_hdr, exp_time=best_exp, analog_gain=best_gain)
    right_ldr = image_model.simulate(right_hdr, exp_time=best_exp, analog_gain=best_gain)

    out_left_dir = exp_out / "hdr_left"
    out_right_dir = exp_out / "hdr_right"
    out_disp_left_dir = exp_out / "ground_truth_disparity_left"
    out_left_dir.mkdir(parents=True, exist_ok=True)
    out_right_dir.mkdir(parents=True, exist_ok=True)
    out_disp_left_dir.mkdir(parents=True, exist_ok=True)

    out_left = out_left_dir / f"{frame_id}.png"
    out_right = out_right_dir / f"{frame_id}.png"
    cv2.imwrite(str(out_left), np_to_image(left_ldr, rgb2bgr=False))
    cv2.imwrite(str(out_right), np_to_image(right_ldr, rgb2bgr=False))

    if disp_left_path is not None and disp_left_path.exists():
        out_disp_left = out_disp_left_dir / disp_left_path.name
        disp_arr = np.load(str(disp_left_path))
        np.save(str(out_disp_left), disp_arr)

    left_rel = relative_source_path(left_path, source_root)
    right_rel = relative_source_path(right_path, source_root)
    if disp_left_path is not None:
        disp_rel = relative_source_path(disp_left_path, source_root)
    else:
        disp_rel = ""

    line = " ".join([p for p in [left_rel, right_rel, disp_rel] if p])
    return frame_id, best_exp, best_gain, line


def run_mode(args):
    source_root = Path(args.source_root).expanduser().resolve()
    mode_root = Path(args.output_root).expanduser().resolve() / args.mode
    mode_root.mkdir(parents=True, exist_ok=True)

    if not source_root.exists():
        raise FileNotFoundError(f"source-root does not exist: {source_root}")

    exp_dirs = sorted([p for p in source_root.iterdir() if p.is_dir()], key=lambda p: natural_sort_key(p.name))
    all_lines: List[str] = []

    total_processed = 0
    for exp_dir in exp_dirs:
        records = collect_records(exp_dir)
        if not records:
            continue

        if args.max_samples is not None:
            remaining = max(args.max_samples - total_processed, 0)
            records = records[:remaining]
            if not records:
                break

        exp_out = mode_root / exp_dir.name
        exp_out.mkdir(parents=True, exist_ok=True)

        tasks = []
        for frame_id, left_path, right_path, disp_left_path in records:
            out_left = exp_out / "hdr_left" / f"{frame_id}.png"
            out_right = exp_out / "hdr_right" / f"{frame_id}.png"
            out_disp = None
            if disp_left_path is not None:
                out_disp = exp_out / "ground_truth_disparity_left" / disp_left_path.name

            files_exist = out_left.exists() and out_right.exists() and (out_disp is None or out_disp.exists())
            if files_exist and not args.overwrite:
                left_rel = relative_source_path(left_path, source_root)
                right_rel = relative_source_path(right_path, source_root)
                disp_rel = relative_source_path(disp_left_path, source_root) if disp_left_path is not None else ""
                all_lines.append(" ".join([p for p in [left_rel, right_rel, disp_rel] if p]))
                total_processed += 1
                continue

            tasks.append((frame_id, left_path, right_path, disp_left_path))

        if tasks:
            results = Parallel(n_jobs=args.n_workers, backend="loky")(
                delayed(process_one_frame)(
                    frame_id=t[0],
                    left_path=t[1],
                    right_path=t[2],
                    disp_left_path=t[3],
                    exp_out=exp_out,
                    mode=args.mode,
                    source_root=source_root,
                    args=args,
                )
                for t in tqdm(tasks, desc=f"[{args.mode}] {exp_dir.name}")
            )

            rows = ["# frame_id exp gain"]
            exp_lines: List[str] = []
            for frame_id, best_exp, best_gain, line in results:
                rows.append(f"{frame_id} {best_exp:.6f} {best_gain:.6f}")
                exp_lines.append(line)
                all_lines.append(line)
                total_processed += 1

            (exp_out / "exposure_params.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
            (exp_out / "val.txt").write_text("\n".join(exp_lines) + ("\n" if exp_lines else ""), encoding="utf-8")

        if args.max_samples is not None and total_processed >= args.max_samples:
            break

    (mode_root / "val.txt").write_text("\n".join(all_lines) + ("\n" if all_lines else ""), encoding="utf-8")
    print(f"Done mode={args.mode}, total={total_processed}, output={mode_root}")


def build_parser():
    parser = argparse.ArgumentParser(description="Offline AE pipeline for Experiment*/hdr_left,hdr_right dataset")
    parser.add_argument("--source-root", type=str, default="~/dataset/ADEC/carla_600x800/val")
    parser.add_argument("--output-root", type=str, default="~/dataset/ADEC/carla_600x800/ae_methods")
    # parser.add_argument("--mode", type=str, default="mixed", choices=["pid", "gradient", "mixed", "semantic"])
    parser.add_argument("--mode", type=str, default="semantic", choices=["pid", "gradient", "mixed",])

    parser.add_argument("--exp-min", type=float, default=5.0)
    parser.add_argument("--exp-max", type=float, default=20.0)
    parser.add_argument("--gain-min", type=float, default=1.0)
    parser.add_argument("--gain-max", type=float, default=20.0)
    parser.add_argument("--max-iters", type=int, default=20)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--pid-setpoint", type=float, default=0.62)

    parser.add_argument("--nbits", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--downsample-factor", type=int, default=1)

    parser.add_argument("--n-workers", type=int, default=max(1, multiprocessing.cpu_count() // 8))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main():
    args = build_parser().parse_args()
    run_mode(args)


if __name__ == "__main__":
    main()
