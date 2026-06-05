# Created: 2026-04-09

import argparse
import multiprocessing
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from utils import ImageFormationModel, inverse_tmo, load_hdr_image, radiance_scale, save_ldr


def available_hdr_files(folder: Path, extensions=(".hdr", ".exr")) -> Dict[str, Path]:
    mapping: Dict[str, Path] = {}
    for ext in extensions:
        for p in folder.glob(f"*{ext}"):
            mapping[p.stem] = p
    return mapping


def frame_id_sort_key(frame_id: str):
    parts = re.split(r"(\d+)", frame_id)
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def resolve_disparity_path(disp_dir: Path, frame_id: str) -> Optional[Tuple[Path, str]]:
    candidates = [
        (disp_dir / f"disparity_map_{frame_id}.npy", f"disparity_map_{frame_id}.npy"),
        (disp_dir / f"{frame_id}.npy", f"{frame_id}.npy"),
    ]
    for path, out_name in candidates:
        if path.exists():
            return path, out_name
    return None


def collect_experiment_records(
    exp_dir: Path,
) -> List[Tuple[str, Path, Path, Optional[Path], Optional[Path], Optional[str], Optional[str]]]:
    records: List[Tuple[str, Path, Path, Optional[Path], Optional[Path], Optional[str], Optional[str]]] = []

    left_dir = exp_dir / "hdr_left"
    right_dir = exp_dir / "hdr_right"
    if left_dir.exists() and right_dir.exists():
        disp_left_dir = exp_dir / "ground_truth_disparity_left"
        disp_right_dir = exp_dir / "ground_truth_disparity_right"
        left_map = available_hdr_files(left_dir)
        right_map = available_hdr_files(right_dir)
        frame_ids = sorted(set(left_map.keys()) & set(right_map.keys()), key=frame_id_sort_key)

        for frame_id in frame_ids:
            disp_left_path: Optional[Path] = None
            disp_left_out_name: Optional[str] = None
            if disp_left_dir.exists():
                disp_left_resolved = resolve_disparity_path(disp_left_dir, frame_id)
                if disp_left_resolved is not None:
                    disp_left_path, disp_left_out_name = disp_left_resolved

            disp_right_path: Optional[Path] = None
            disp_right_out_name: Optional[str] = None
            if disp_right_dir.exists():
                disp_right_resolved = resolve_disparity_path(disp_right_dir, frame_id)
                if disp_right_resolved is not None:
                    disp_right_path, disp_right_out_name = disp_right_resolved

            records.append(
                (
                    frame_id,
                    left_map[frame_id],
                    right_map[frame_id],
                    disp_left_path,
                    disp_right_path,
                    disp_left_out_name,
                    disp_right_out_name,
                )
            )
        return records

    return records


def brightness_channel(image: np.ndarray) -> np.ndarray:
    return 0.299 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.114 * image[:, :, 2]


def histogram_subimage_brightness(
    image: np.ndarray,
    grid_size: int,
    bins: int = 256,
    value_min: float = 0.0,
    value_max: float = 1.0,
) -> np.ndarray:
    bright = brightness_channel(image)
    height, width = bright.shape

    if grid_size <= 1 or height < grid_size or width < grid_size:
        hist, _ = np.histogram(bright, bins=bins, range=(value_min, value_max))
        return hist.astype(np.float64)

    grid_h = height // grid_size
    grid_w = width // grid_size
    if grid_h <= 0 or grid_w <= 0:
        hist, _ = np.histogram(bright, bins=bins, range=(value_min, value_max))
        return hist.astype(np.float64)

    hists: List[np.ndarray] = []
    for i in range(grid_size):
        for j in range(grid_size):
            sub = bright[i * grid_h:(i + 1) * grid_h, j * grid_w:(j + 1) * grid_w]
            if sub.size == 0:
                continue
            sub = np.clip(sub, value_min, value_max)
            hist, _ = np.histogram(sub, bins=bins, range=(value_min, value_max))
            hists.append(hist.astype(np.float64))

    if not hists:
        hist, _ = np.histogram(bright, bins=bins, range=(value_min, value_max))
        return hist.astype(np.float64)
    return np.mean(np.stack(hists, axis=0), axis=0)


def multi_scale_histogram(
    image: np.ndarray,
    bins: int = 256,
    value_min: float = 0.0,
    value_max: float = 1.0,
) -> np.ndarray:
    scales = [1, 3, 7]
    hists = [
        histogram_subimage_brightness(
            image,
            grid_size=s,
            bins=bins,
            value_min=value_min,
            value_max=value_max,
        )
        for s in scales
    ]
    return np.mean(np.stack(hists, axis=0), axis=0)


def skewness_from_hist(hist: np.ndarray, fixed_mean: Optional[float] = None, eps: float = 1e-6) -> float:
    if fixed_mean is None:
        fixed_mean = (hist.shape[0] - 1) / 2.0

    pixel_values = np.arange(hist.shape[0], dtype=np.float64)
    diff = pixel_values - float(fixed_mean)
    total = float(hist.sum()) + eps
    numerator = float(np.sum(hist * (diff ** 3)))
    variance = float(np.sum(hist * (diff ** 2)) / total)
    std = float(np.sqrt(variance + eps))
    return numerator / (total * (std ** 3 + eps))


def clamping_ratio_from_hist(
    hist: np.ndarray,
    low_threshold: float = 0.05,
    high_threshold: float = 0.95,
    eps: float = 1e-6,
) -> Tuple[float, float]:
    bins = hist.shape[0]
    low_idx = int(low_threshold * (bins - 1))
    high_idx = int(high_threshold * (bins - 1))

    total = float(hist.sum()) + eps
    low_ratio = float(hist[:low_idx].sum() / total)
    high_ratio = float(hist[high_idx:].sum() / total)
    return low_ratio, high_ratio


class StereoExposureController:
    def __init__(
        self,
        low_threshold: float = 0.05,
        high_threshold: float = 0.95,
        hdr_ratio_threshold: float = 0.05,
        alpha_skew: float = 0.1,
        exp_gap_threshold: float = 2.0,
        hist_bins: int = 256,
        hist_value_min: float = 0.0,
        hist_value_max: float = 1.0,
        alpha: float = 0.1,
    ):
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.hdr_ratio_threshold = hdr_ratio_threshold
        self.alpha_skew = alpha_skew
        self.exp_gap_threshold = exp_gap_threshold
        self.hist_bins = hist_bins
        self.hist_value_min = hist_value_min
        self.hist_value_max = hist_value_max
        self.alpha = alpha
        self.fixed_mean = (hist_bins - 1) / 2.0

    def update(
        self,
        left_image: np.ndarray,
        right_image: np.ndarray,
        current_exp_left: float,
        current_exp_right: float,
        alpha_left: float,
        alpha_right: float,
    ) -> Tuple[float, float, Dict[str, float]]:
        left_image = np.clip(left_image.astype(np.float64), self.hist_value_min, self.hist_value_max)
        right_image = np.clip(right_image.astype(np.float64), self.hist_value_min, self.hist_value_max)

        hist_left = multi_scale_histogram(
            left_image,
            bins=self.hist_bins,
            value_min=self.hist_value_min,
            value_max=self.hist_value_max,
        )
        hist_right = multi_scale_histogram(
            right_image,
            bins=self.hist_bins,
            value_min=self.hist_value_min,
            value_max=self.hist_value_max,
        )

        skew_left = skewness_from_hist(hist_left, fixed_mean=self.fixed_mean)
        skew_right = skewness_from_hist(hist_right, fixed_mean=self.fixed_mean)

        low_left, high_left = clamping_ratio_from_hist(hist_left, self.low_threshold, self.high_threshold)
        low_right, high_right = clamping_ratio_from_hist(hist_right, self.low_threshold, self.high_threshold)

        hdr_left = (low_left > self.hdr_ratio_threshold) and (high_left > self.hdr_ratio_threshold)
        hdr_right = (low_right > self.hdr_ratio_threshold) and (high_right > self.hdr_ratio_threshold)
        hdr_scene = hdr_left or hdr_right

        exp_diff = abs(current_exp_left - current_exp_right)
        widen_mask = hdr_scene and (exp_diff < self.exp_gap_threshold)

        left_higher = current_exp_left > current_exp_right
        if left_higher:
            new_left_hdr = current_exp_left + alpha_left * low_left
            new_right_hdr = current_exp_right - alpha_right * high_right
        else:
            new_left_hdr = current_exp_left - alpha_left * high_left
            new_right_hdr = current_exp_right + alpha_right * low_right

        new_left_ldr = current_exp_left - self.alpha_skew * skew_left
        new_right_ldr = current_exp_right - self.alpha_skew * skew_right

        new_left = new_left_hdr if widen_mask else new_left_ldr
        new_right = new_right_hdr if widen_mask else new_right_ldr

        metrics = {
            "skew_left": float(skew_left),
            "skew_right": float(skew_right),
            "low_ratio_left": float(low_left),
            "high_ratio_left": float(high_left),
            "low_ratio_right": float(low_right),
            "high_ratio_right": float(high_right),
            "hdr_scene": 1.0 if hdr_scene else 0.0,
            "widen_mask": 1.0 if widen_mask else 0.0,
            "exp_diff": float(exp_diff),
        }
        return float(new_left), float(new_right), metrics


def run_stereo_exposure_iteration(
    left_hdr: np.ndarray,
    right_hdr: np.ndarray,
    model: ImageFormationModel,
    controller: StereoExposureController,
    exp_bounds: Tuple[float, float],
    gain_bounds: Tuple[float, float],
    init_exposure: float,
    init_gain: float,
    iters: int,
) -> Tuple[np.ndarray, np.ndarray, float, float, float, float, Dict[str, float]]:
    exp_left = float(np.clip(init_exposure, exp_bounds[0], exp_bounds[1]))
    exp_right = float(np.clip(init_exposure, exp_bounds[0], exp_bounds[1]))
    gain_seed = float(np.clip(init_gain, gain_bounds[0], gain_bounds[1]))

    # Keep exactly the same control flow as stereo_ae_2:
    # seed images use init exposure + init gain before iterative updates.
    img_left = model.subframes_fusion(left_hdr, exp_time=exp_left, analog_gain=gain_seed)
    img_right = model.subframes_fusion(right_hdr, exp_time=exp_right, analog_gain=gain_seed)

    alpha_left = float(controller.alpha)
    alpha_right = float(controller.alpha)
    # max_ev = float(exp_bounds[1] * gain_bounds[1])
    max_ev = float(exp_bounds[1] * gain_bounds[0])  # Use max exposure with min gain as max EV reference
    # print(f"MAX EV for this scene: {max_ev:.2f}")
    # exit(0)  # --- IGNORE ---

    last_metrics: Dict[str, float] = {
        "skew_left": 0.0,
        "skew_right": 0.0,
        "low_ratio_left": 0.0,
        "high_ratio_left": 0.0,
        "low_ratio_right": 0.0,
        "high_ratio_right": 0.0,
        "hdr_scene": 0.0,
        "widen_mask": 0.0,
        "exp_diff": 0.0,
    }

    for _ in range(max(int(iters), 1)):
        new_exp_left, new_exp_right, last_metrics = controller.update(
            left_image=img_left,
            right_image=img_right,
            current_exp_left=exp_left,
            current_exp_right=exp_right,
            alpha_left=alpha_left,
            alpha_right=alpha_right,
        )

        exp_left = float(np.clip(new_exp_left, exp_bounds[0], exp_bounds[1]))
        exp_right = float(np.clip(new_exp_right, exp_bounds[0], exp_bounds[1]))
        gain_left = float(np.clip(max_ev / exp_left, gain_bounds[0], gain_bounds[1]))
        gain_right = float(np.clip(max_ev / exp_right, gain_bounds[0], gain_bounds[1]))

        img_left = model.subframes_fusion(left_hdr, exp_time=exp_left, analog_gain=gain_left)
        img_right = model.subframes_fusion(right_hdr, exp_time=exp_right, analog_gain=gain_right)

    gain_left = float(np.clip(max_ev / exp_left, gain_bounds[0], gain_bounds[1]))
    gain_right = float(np.clip(max_ev / exp_right, gain_bounds[0], gain_bounds[1]))

    # print(f"In this turn:{exp_left:.2f} {gain_left:.2f} {exp_right:.2f} {gain_right:.2f} \n")
    return img_left, img_right, exp_left, exp_right, gain_left, gain_right, last_metrics


def process_one_frame(
    frame_id: str,
    left_path: Path,
    right_path: Path,
    disp_left_path: Optional[Path],
    disp_right_path: Optional[Path],
    out_left: Path,
    out_right: Path,
    out_disp_left: Optional[Path],
    out_disp_right: Optional[Path],
    args,
):
    t0 = time.time()
    left_hdr = load_hdr_image(str(left_path))
    right_hdr = load_hdr_image(str(right_path))

    # left_hdr = inverse_tmo(left_hdr)
    # right_hdr = inverse_tmo(right_hdr)

    left_hdr = radiance_scale(left_hdr, capacity=args.radiance_capacity)
    right_hdr = radiance_scale(right_hdr, capacity=args.radiance_capacity)

    if args.downsample_factor > 1:
        h, w = left_hdr.shape[:2]
        new_h, new_w = h // args.downsample_factor, w // args.downsample_factor
        left_ctrl = cv2.resize(left_hdr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        right_ctrl = cv2.resize(right_hdr, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    else:
        left_ctrl = left_hdr
        right_ctrl = right_hdr

    seed = int(args.seed + abs(hash(frame_id)) % 100000)
    model = ImageFormationModel(nbits=args.nbits, seed=seed)
    controller = StereoExposureController(
        low_threshold=args.low_threshold,
        high_threshold=args.high_threshold,
        hdr_ratio_threshold=args.hdr_ratio_threshold,
        alpha_skew=args.alpha_skew,
        exp_gap_threshold=args.exp_gap_threshold,
        hist_bins=args.hist_bins,
        hist_value_min=args.hist_value_min,
        hist_value_max=args.hist_value_max,
        alpha=args.alpha,
    )

    _, _, exp_left, exp_right, gain_left, gain_right, metrics = run_stereo_exposure_iteration(
        left_hdr=left_ctrl,
        right_hdr=right_ctrl,
        model=model,
        controller=controller,
        exp_bounds=(args.exp_min, args.exp_max),
        gain_bounds=(args.gain_min, args.gain_max),
        init_exposure=args.init_exposure,
        init_gain=args.init_gain,
        iters=args.iters,
    )

    ldr_left = model.subframes_fusion(left_hdr, exp_time=exp_left, analog_gain=gain_left)
    ldr_right = model.subframes_fusion(right_hdr, exp_time=exp_right, analog_gain=gain_right)

    out_left.parent.mkdir(parents=True, exist_ok=True)
    out_right.parent.mkdir(parents=True, exist_ok=True)
    save_ldr(str(out_left), ldr_left)
    save_ldr(str(out_right), ldr_right)

    if disp_left_path is not None and out_disp_left is not None and disp_left_path.exists():
        out_disp_left.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_disp_left), np.load(str(disp_left_path)))

    if disp_right_path is not None and out_disp_right is not None and disp_right_path.exists():
        out_disp_right.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_disp_right), np.load(str(disp_right_path)))

    rel_left = out_left.as_posix()
    rel_right = out_right.as_posix()
    rel_disp_left = out_disp_left.as_posix() if out_disp_left is not None else ""
    rel_disp_right = out_disp_right.as_posix() if out_disp_right is not None else ""
    line_parts = [p for p in [rel_left, rel_right, rel_disp_left, rel_disp_right] if p]

    return {
        "frame_id": frame_id,
        "exp_left": float(exp_left),
        "gain_left": float(gain_left),
        "exp_right": float(exp_right),
        "gain_right": float(gain_right),
        "metrics": metrics,
        "line": " ".join(line_parts),
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
        out_disp_left_dir = out_exp / "ground_truth_disparity_left"
        out_disp_right_dir = out_exp / "ground_truth_disparity_right"

        if args.max_samples is not None:
            remaining = max(args.max_samples - summary["total"], 0)
            records = records[:remaining]
        if not records:
            continue

        tasks = []
        for frame_id, left_path, right_path, disp_left_path, disp_right_path, disp_left_out_name, disp_right_out_name in records:
            out_left = out_left_dir / f"{frame_id}.png"
            out_right = out_right_dir / f"{frame_id}.png"
            out_disp_left = out_disp_left_dir / disp_left_out_name if (disp_left_path is not None and disp_left_out_name is not None) else None
            out_disp_right = out_disp_right_dir / disp_right_out_name if (disp_right_path is not None and disp_right_out_name is not None) else None

            files_exist = (
                out_left.exists()
                and out_right.exists()
                and (out_disp_left is None or out_disp_left.exists())
                and (out_disp_right is None or out_disp_right.exists())
            )

            if not args.overwrite and files_exist:
                line_parts = [
                    out_left.as_posix(),
                    out_right.as_posix(),
                    out_disp_left.as_posix() if out_disp_left is not None else "",
                    out_disp_right.as_posix() if out_disp_right is not None else "",
                ]
                summary["lines"].append(" ".join([p for p in line_parts if p]))
                continue

            tasks.append((frame_id, left_path, right_path, disp_left_path, disp_right_path, out_left, out_right, out_disp_left, out_disp_right))

        if not tasks:
            if args.max_samples is not None and summary["total"] >= args.max_samples:
                break
            continue

        results = Parallel(n_jobs=args.n_workers, backend="loky")(
            delayed(process_one_frame)(
                frame_id=t[0],
                left_path=t[1],
                right_path=t[2],
                disp_left_path=t[3],
                disp_right_path=t[4],
                out_left=t[5],
                out_right=t[6],
                out_disp_left=t[7],
                out_disp_right=t[8],
                args=args,
            )
            for t in tqdm(tasks, desc=f"[{split}] {exp_dir.name}")
        )

        rows = [
            "# frame_id exp_left gain_left exp_right gain_right "
            "skew_left skew_right low_ratio_left high_ratio_left low_ratio_right high_ratio_right hdr_scene widen_mask"
        ]
        for result in results:
            summary["total"] += 1
            summary["success"] += 1
            summary["times"].append(result["elapsed"])
            summary["lines"].append(result["line"])

            m = result["metrics"]
            rows.append(
                f"{result['frame_id']} "
                f"{result['exp_left']:.6f} {result['gain_left']:.6f} "
                f"{result['exp_right']:.6f} {result['gain_right']:.6f} "
                f"{m['skew_left']:.6f} {m['skew_right']:.6f} "
                f"{m['low_ratio_left']:.6f} {m['high_ratio_left']:.6f} "
                f"{m['low_ratio_right']:.6f} {m['high_ratio_right']:.6f} "
                f"{m['hdr_scene']:.0f} {m['widen_mask']:.0f}"
            )

        out_exp.mkdir(parents=True, exist_ok=True)
        (out_exp / "stereo_exposure_params.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")

        if args.max_samples is not None and summary["total"] >= args.max_samples:
            break

    split_txt = run_root / f"{split}.txt"
    split_txt.write_text("\n".join(summary["lines"]) + ("\n" if summary["lines"] else ""), encoding="utf-8")
    return summary


def run_stereo_exposure(args) -> Dict[str, Dict]:
    args.source_root = Path(args.source_root).expanduser().resolve()
    args.output_root = Path(args.output_root).expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    run_name = f"stereo_ae_{args.metric_name}"
    run_root = args.output_root / run_name
    run_root.mkdir(parents=True, exist_ok=True)

    print(f"source_root={args.source_root}")
    print(f"output_root={run_root}")
    print(f"splits={args.splits}")
    print(
        f"iters={args.iters} alpha={args.alpha} alpha_skew={args.alpha_skew} "
        f"exp_gap_threshold={args.exp_gap_threshold}"
    )
    print(f"workers={args.n_workers} max_samples={args.max_samples}")

    all_results: Dict[str, Dict] = {}
    start = time.time()
    for split in args.splits:
        result = process_split(split=split, args=args, run_root=run_root)
        all_results[split] = result
        mean_time = float(np.mean(result["times"])) if result["times"] else 0.0
        print(
            f"[split={split}] total={result['total']} success={result['success']} "
            f"failed={result['failed']} avg_time={mean_time:.4f}s"
        )
        if result["errors"]:
            for err in result["errors"]:
                print(f"  error: {err}")

    print(f"Done. elapsed={time.time() - start:.2f}s")
    return all_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone stereo exposure controller offline pipeline. "
        "Metrics and update formulas are reimplemented from stereonet/stereo_ae_2.py."
    )
    parser.add_argument("--source-root", type=str, default="~/dataset/ADEC/aaa")
    parser.add_argument("--output-root", type=str, default="~/dataset/ADEC/aaa/carla_ae")
    parser.add_argument("--splits", nargs="+", default=["val"])

    parser.add_argument("--metric-name", type=str, default="hist_skew_hdr")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=1)
    parser.add_argument("--alpha-skew", type=float, default=1)
    parser.add_argument("--low-threshold", type=float, default=0.05)
    parser.add_argument("--high-threshold", type=float, default=0.95)
    parser.add_argument("--hdr-ratio-threshold", type=float, default=0.05)
    parser.add_argument("--exp-gap-threshold", type=float, default=3.0)
    parser.add_argument("--hist-bins", type=int, default=256)
    parser.add_argument("--hist-value-min", type=float, default=0.0)
    parser.add_argument("--hist-value-max", type=float, default=1.0)

    parser.add_argument("--exp-min", type=float, default=1.0)
    parser.add_argument("--exp-max", type=float, default=20.0)
    parser.add_argument("--gain-min", type=float, default=1.0)
    parser.add_argument("--gain-max", type=float, default=20.0)
    parser.add_argument("--init-exposure", type=float, default=10.0)
    parser.add_argument("--init-gain", type=float, default=10.0)

    parser.add_argument("--radiance-capacity", type=float, default=12.0)
    parser.add_argument("--nbits", type=int, default=10)
    parser.add_argument("--downsample-factor", type=int, default=2)

    parser.add_argument("--n-workers", type=int, default=max(1, multiprocessing.cpu_count() // 2))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--seed", type=int, default=2026)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    np.random.seed(args.seed)
    run_stereo_exposure(args)


if __name__ == "__main__":
    main()
