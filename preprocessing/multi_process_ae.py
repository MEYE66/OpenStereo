# Created: 2025-06-03
# Refactored: 2026-03-28

import argparse
import multiprocessing
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from np_optimizer import GridSearchOptimizer, NelderMeadOptimizer
from pid_optimizer import PIDOptimizer
from utils import (
    ImageContrastMetric,
    ImageEntropyMetric,
    ImageFormationModel,
    ImageGradientMetric,
    ImageSemanticMetric,
    MixedImageMetric,
    load_hdr_image,
    inverse_tmo,
    radiance_scale,
    save_ldr,
)


def build_metric(metric_name: str):
    name = metric_name.lower()
    if name == "entropy":
        return ImageEntropyMetric()
    if name == "gradient":
        return ImageGradientMetric()
    if name == "mixed":
        return MixedImageMetric()
    if name == "contrast":
        return ImageContrastMetric()
    if name == "semantic":
        return ImageSemanticMetric()
    raise ValueError(f"Unsupported metric: {metric_name}")


def default_target_score(metric_name: str) -> float:
    name = metric_name.lower()
    if name == "entropy":
        return 0.85
    if name == "gradient":
        return 0.35
    if name == "mixed":
        return 0.40
    if name == "contrast":
        return 0.50
    if name == "semantic":
        return 2.50
    return 0.50


def available_hdr_files(folder: Path) -> Dict[str, Path]:
    exts = (".hdr", ".exr", ".tiff", ".tif", ".png", ".npy")
    mapping: Dict[str, Path] = {}
    for ext in exts:
        for p in folder.glob(f"*{ext}"):
            mapping[p.stem] = p
    return mapping


def collect_experiment_records(
    exp_dir: Path,
) -> List[Tuple[str, Path, Path, Optional[Path], Optional[str]]]:
    """Return (frame_id, left_path, right_path, aux_path, aux_out_name)."""
    records: List[Tuple[str, Path, Path, Optional[Path], Optional[str]]] = []

    left_dir = exp_dir / "hdr_left"
    right_dir = exp_dir / "hdr_right"
    if left_dir.exists() and right_dir.exists():
        disp_dir = exp_dir / "ground_truth_disparity_left"
        left_map = available_hdr_files(left_dir)
        right_map = available_hdr_files(right_dir)
        frame_ids = sorted(set(left_map.keys()) & set(right_map.keys()))
        for frame_id in frame_ids:
            disp_path = disp_dir / f"disparity_map_{frame_id}.npy"
            if not disp_path.exists():
                disp_path = None
            disp_out_name = f"disparity_map_{frame_id}.npy" if disp_path is not None else None
            records.append((frame_id, left_map[frame_id], right_map[frame_id], disp_path, disp_out_name))
        return records

    candidate_dirs: List[Path] = []
    if (exp_dir / "left_rectified.npy").exists() and (exp_dir / "right_rectified.npy").exists():
        candidate_dirs.append(exp_dir)
    for child in sorted([p for p in exp_dir.iterdir() if p.is_dir()]):
        if (child / "left_rectified.npy").exists() and (child / "right_rectified.npy").exists():
            candidate_dirs.append(child)

    for frame_dir in candidate_dirs:
        frame_id = frame_dir.name
        left_path = frame_dir / "left_rectified.npy"
        right_path = frame_dir / "right_rectified.npy"
        aux_path = frame_dir / "points.npy"
        if not aux_path.exists():
            aux_path = None
        aux_out_name = f"{frame_id}_points.npy" if aux_path is not None else None
        records.append((frame_id, left_path, right_path, aux_path, aux_out_name))

    return records


def optimize_left_image(
    left_hdr: np.ndarray,
    metric_name: str,
    optimizer_name: str,
    controller_name: str,
    exposure_bounds: Tuple[float, float],
    gain_bounds: Tuple[int, int],
    target_score: Optional[float],
    pid_max_iter: int,
    pid_tol: float,
    nm_max_iter: int,
    nm_tol: float,
    grid_exp_steps: int,
    grid_gain_step: int,
    init_exposure: float,
    init_gain: int,
    nbits: int,
    seed: int,
):
    metric = build_metric(metric_name)
    model = ImageFormationModel(nbits=nbits, seed=seed)
    bounds = [exposure_bounds, gain_bounds]

    controller_name = controller_name.lower()
    optimizer_name = optimizer_name.lower()

    if controller_name == "fixed":
        exp = float(np.clip(init_exposure, exposure_bounds[0], exposure_bounds[1]))
        gain = int(np.clip(init_gain, gain_bounds[0], gain_bounds[1]))
        image = model.subframes_fusion(left_hdr, exp_time=exp, analog_gain=gain)
        score = float(metric.evaluate(image))
        return exp, gain, score, image, model

    if optimizer_name == "pid":
        score_target = default_target_score(metric_name) if target_score is None else target_score
        optimizer = PIDOptimizer(
            image_formation_model=model,
            metric=metric,
            bounds=bounds,
            setpoint=float(score_target),
            max_iter=pid_max_iter,
            tol=pid_tol,
        )
        result = optimizer.optimize(left_hdr)
        return result.exposure, result.gain, result.score, result.image, model

    if optimizer_name == "nelder_mead":
        optimizer = NelderMeadOptimizer(
            image_formation_model=model,
            metric=metric,
            bounds=bounds,
            max_iter=nm_max_iter,
            tol=nm_tol,
        )
        result = optimizer.optimize(left_hdr)
        return result.exposure, result.gain, result.score, result.image, model

    if optimizer_name == "grid":
        optimizer = GridSearchOptimizer(
            image_formation_model=model,
            metric=metric,
            bounds=bounds,
            exposure_steps=grid_exp_steps,
            gain_step=grid_gain_step,
        )
        result = optimizer.optimize(left_hdr)
        return result.exposure, result.gain, result.score, result.image, model

    raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def process_one_frame(
    frame_id: str,
    left_path: Path,
    right_path: Path,
    disp_path: Optional[Path],
    out_left: Path,
    out_right: Path,
    out_disp: Optional[Path],
    args,
):
    t0 = time.time()
    left_hdr = load_hdr_image(str(left_path))
    right_hdr = load_hdr_image(str(right_path))

    left_hdr = inverse_tmo(left_hdr)
    right_hdr = inverse_tmo(right_hdr)


    left_hdr = radiance_scale(left_hdr, capacity=args.radiance_capacity)
    right_hdr = radiance_scale(right_hdr, capacity=args.radiance_capacity)

    # Downsample for optimization if specified
    if args.downsample_factor > 1:
        h, w = left_hdr.shape[:2]
        new_h, new_w = h // args.downsample_factor, w // args.downsample_factor
        left_hdr_opt = cv2.resize(left_hdr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        right_hdr_opt = cv2.resize(right_hdr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    else:
        left_hdr_opt = left_hdr
        right_hdr_opt = right_hdr

    seed = int(args.seed + abs(hash(frame_id)) % 100000)
    exp, gain, score, ldr_left_opt, model = optimize_left_image(
        left_hdr=left_hdr_opt,
        metric_name=args.metric,
        optimizer_name=args.optimizer,
        controller_name=args.controller,
        exposure_bounds=(args.exp_min, args.exp_max),
        gain_bounds=(args.gain_min, args.gain_max),
        target_score=args.target_score,
        pid_max_iter=args.pid_max_iters,
        pid_tol=args.pid_tol,
        nm_max_iter=args.nm_max_iters,
        nm_tol=args.nm_tol,
        grid_exp_steps=args.grid_exp_steps,
        grid_gain_step=args.grid_gain_step,
        init_exposure=args.init_exposure,
        init_gain=args.init_gain,
        nbits=args.nbits,
        seed=seed,
    )

    # Apply optimized parameters to full-resolution images
    ldr_left = model.subframes_fusion(left_hdr, exp_time=exp, analog_gain=gain)
    ldr_right = model.subframes_fusion(right_hdr, exp_time=exp, analog_gain=gain)

    out_left.parent.mkdir(parents=True, exist_ok=True)
    out_right.parent.mkdir(parents=True, exist_ok=True)
    save_ldr(str(out_left), ldr_left)
    save_ldr(str(out_right), ldr_right)

    if disp_path is not None and out_disp is not None and disp_path.exists():
        out_disp.parent.mkdir(parents=True, exist_ok=True)
        disp_arr = np.load(str(disp_path))
        np.save(str(out_disp), disp_arr)

    rel_left = out_left.as_posix()
    rel_right = out_right.as_posix()
    rel_disp = out_disp.as_posix() if out_disp is not None else ""

    return {
        "frame_id": frame_id,
        "exp": float(exp),
        "gain": int(gain),
        "score": float(score),
        "line": f"{rel_left} {rel_right} {rel_disp}".strip(),
        "elapsed": float(time.time() - t0),
    }


def process_split(split: str, args, run_root: Path) -> Dict:
    split_src = args.source_root / split
    split_dst = run_root / split
    split_dst.mkdir(parents=True, exist_ok=True)

    if not split_src.exists():
        return {
            "split": split,
            "total": 0,
            "success": 0,
            "failed": 0,
            "lines": [],
            "errors": [f"split path does not exist: {split_src}"],
            "times": [],
        }

    summary = {
        "split": split,
        "total": 0,
        "success": 0,
        "failed": 0,
        "lines": [],
        "errors": [],
        "times": [],
    }

    experiments = sorted([p for p in split_src.iterdir() if p.is_dir()])
    for exp_dir in experiments:
        records = collect_experiment_records(exp_dir)
        if not records:
            continue

        out_exp = split_dst / exp_dir.name
        out_left_dir = out_exp / "ldr_left"
        out_right_dir = out_exp / "ldr_right"
        out_disp_dir = out_exp / "ground_truth_disparity_left"

        if args.max_samples is not None:
            remaining = max(args.max_samples - summary["total"], 0)
            records = records[:remaining]
        if not records:
            continue

        tasks = []
        for frame_id, left_path, right_path, disp_path, disp_out_name in records:

            out_left = out_left_dir / f"{frame_id}.png"
            out_right = out_right_dir / f"{frame_id}.png"
            out_disp = out_disp_dir / disp_out_name if (disp_path is not None and disp_out_name is not None) else None

            if (
                not args.overwrite
                and out_left.exists()
                and out_right.exists()
                and (out_disp is None or out_disp.exists())
            ):
                summary["lines"].append(
                    f"{out_left.as_posix()} {out_right.as_posix()} {(out_disp.as_posix() if out_disp is not None else '')}".strip()
                )
                continue

            tasks.append((frame_id, left_path, right_path, disp_path, out_left, out_right, out_disp))

        if not tasks:
            if args.max_samples is not None and summary["total"] >= args.max_samples:
                break
            continue

        results = Parallel(n_jobs=args.n_workers, backend="loky")(
            delayed(process_one_frame)(
                frame_id=t[0],
                left_path=t[1],
                right_path=t[2],
                disp_path=t[3],
                out_left=t[4],
                out_right=t[5],
                out_disp=t[6],
                args=args,
            )
            for t in tqdm(tasks, desc=f"[{split}] {exp_dir.name}")
        )

        exposure_rows = ["# frame_id exposure gain score"]
        for result in results:
            summary["total"] += 1
            summary["success"] += 1
            summary["times"].append(result["elapsed"])
            summary["lines"].append(result["line"])
            exposure_rows.append(
                f"{result['frame_id']} {result['exp']:.6f} {result['gain']} {result['score']:.6f}"
            )

        exposure_file = out_exp / "exposure_params.txt"
        exposure_file.parent.mkdir(parents=True, exist_ok=True)
        exposure_file.write_text("\n".join(exposure_rows) + "\n", encoding="utf-8")

        if args.max_samples is not None and summary["total"] >= args.max_samples:
            break

    split_txt = run_root / f"{split}.txt"
    split_txt.write_text("\n".join(summary["lines"]) + ("\n" if summary["lines"] else ""), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline auto exposure pipeline with numpy image formation model.")
    parser.add_argument("--source-root", type=str, default="~/dataset/ADEC/aaa")
    parser.add_argument("--output-root", type=str, default="~/dataset/ADEC")
    parser.add_argument("--splits", nargs="+", default=["val"])

    parser.add_argument("--controller", type=str, default="pid", choices=["pid", "fixed"])
    parser.add_argument("--optimizer", type=str, default="pid", choices=["pid", "nelder_mead", "grid"])
    parser.add_argument("--metric", type=str, default="entropy", choices=["entropy", "gradient", "mixed", "contrast", "semantic"])

    parser.add_argument("--target-score", type=float, default=None)

    parser.add_argument("--exp-min", type=float, default=5.0)
    parser.add_argument("--exp-max", type=float, default=20.0)
    parser.add_argument("--gain-min", type=int, default=1)
    parser.add_argument("--gain-max", type=int, default=20)

    parser.add_argument("--init-exposure", type=float, default=12.5)
    parser.add_argument("--init-gain", type=int, default=8)

    parser.add_argument("--pid-max-iters", type=int, default=30)
    parser.add_argument("--pid-tol", type=float, default=0.02)

    parser.add_argument("--nm-max-iters", type=int, default=30)
    parser.add_argument("--nm-tol", type=float, default=1e-5)

    parser.add_argument("--grid-exp-steps", type=int, default=16)
    parser.add_argument("--grid-gain-step", type=int, default=1)

    parser.add_argument("--radiance-capacity", type=float, default=12.0)
    parser.add_argument("--nbits", type=int, default=10)

    parser.add_argument("--n-workers", type=int, default=max(1, multiprocessing.cpu_count() // 2))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--downsample-factor", type=int, default=2, help="Downsample factor for optimization (1=no downsampling)")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    np.random.seed(args.seed)

    args.source_root = Path(args.source_root).expanduser().resolve()
    args.output_root = Path(args.output_root).expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    run_name = f"ae_{args.controller}_{args.optimizer}_{args.metric}"
    run_root = args.output_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    print(f"source_root={args.source_root}")
    print(f"output_root={run_root}")
    print(f"splits={args.splits}")
    print(f"controller={args.controller} optimizer={args.optimizer} metric={args.metric}")
    print(f"workers={args.n_workers} max_samples={args.max_samples}")

    start = time.time()
    for split in args.splits:
        result = process_split(split=split, args=args, run_root=run_root)
        mean_time = float(np.mean(result["times"])) if result["times"] else 0.0
        print(
            f"[split={split}] total={result['total']} success={result['success']} "
            f"failed={result['failed']} avg_time={mean_time:.4f}s"
        )
        if result["errors"]:
            for err in result["errors"]:
                print(f"  error: {err}")

    print(f"Done. elapsed={time.time() - start:.2f}s")


if __name__ == "__main__":
    main()
