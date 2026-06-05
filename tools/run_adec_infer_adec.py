#!/usr/bin/env python3
"""Run ADEC CARLA inference and export OpenStereo-compatible files."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


OPENSTEREO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADEC_REPO = Path("/home/lgz/workspace/ADEC-main")
DEFAULT_ADEC_DATA_ROOT = Path("/home/lgz/dataset/ADEC")
DEFAULT_DISPARITY_ROOT = OPENSTEREO_ROOT / "output_disparity"
DEFAULT_RUN_ROOT = OPENSTEREO_ROOT / "output" / "adec_infer_runs"

DATASET_CHECKPOINTS = {
    "carla_600x800": "carla_600x800_adec.pth",
    "carla_1280x384": "carla_1280x384_adec.pth",
}


@dataclass(frozen=True)
class ExportPaths:
    ae_root: Path
    disparity_root: Path
    val_txt: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run ADEC inference for CARLA datasets and export results."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["carla_600x800", "carla_1280x384"],
        choices=sorted(DATASET_CHECKPOINTS),
        help="Datasets to process in launcher mode.",
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        default=["7", "6"],
        help="GPU ids assigned to datasets in order.",
    )
    parser.add_argument("--adec-repo", type=Path, default=DEFAULT_ADEC_REPO)
    parser.add_argument("--adec-data-root", type=Path, default=DEFAULT_ADEC_DATA_ROOT)
    parser.add_argument("--disparity-root", type=Path, default=DEFAULT_DISPARITY_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--valid-iters", type=int, default=32)
    parser.add_argument("--limit", type=int, default=None, help="Optional worker sample limit.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--dataset", choices=sorted(DATASET_CHECKPOINTS), help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--summary-json", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def ensure_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def ensure_dir(path: Path, label: str) -> None:
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path}")


def setup_adec_imports(adec_repo: Path):
    adec_repo = adec_repo.resolve()
    ensure_dir(adec_repo, "ADEC repository")
    sys.path.insert(0, str(adec_repo))
    sys.path.insert(0, str(adec_repo / "core"))
    os.chdir(adec_repo)

    import numpy as np
    import torch
    import torch.utils.data as data
    from PIL import Image
    from tqdm import tqdm

    from core.combine_model_dual import CombineModel
    from core.stereo_datasets import CARLASequenceDataset, resolve_carla_root
    from test_sequence_carla import (
        compute_batch_metrics,
        init_metric_tracker,
        summarize_metric_tracker,
        update_metric_tracker,
    )

    return SimpleNamespace(
        np=np,
        torch=torch,
        data=data,
        Image=Image,
        tqdm=tqdm,
        CombineModel=CombineModel,
        CARLASequenceDataset=CARLASequenceDataset,
        resolve_carla_root=resolve_carla_root,
        compute_batch_metrics=compute_batch_metrics,
        init_metric_tracker=init_metric_tracker,
        summarize_metric_tracker=summarize_metric_tracker,
        update_metric_tracker=update_metric_tracker,
    )


def make_model_args(dataset: str, checkpoint: Path, valid_iters: int) -> SimpleNamespace:
    return SimpleNamespace(
        name=f"ADEC_{dataset}",
        restore_ckpt=str(checkpoint),
        mixed_precision=False,
        train_datasets=[dataset],
        test_dataset=dataset,
        eval_split="val",
        output_root="test_results_carla",
        num_workers=2,
        metrics_only=False,
        batch_size=1,
        valid_iters=valid_iters,
        device="cuda:0",
        corr_implementation="reg",
        shared_backbone=False,
        corr_levels=4,
        corr_radius=4,
        n_downsample=2,
        context_norm="batch",
        slow_fast_gru=False,
        n_gru_layers=3,
        hidden_dims=[128, 128, 128],
    )


def tensor_to_uint8(tensor, torch_module, np_module):
    image = tensor.detach().cpu().float().clamp(0.0, 1.0)
    if image.ndim == 4:
        image = image[0]
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0)
    array = (image.numpy() * 255.0).round().astype(np_module.uint8)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    return array


def save_image_tensor(path: Path, tensor, modules) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = tensor_to_uint8(tensor, modules.torch, modules.np)
    modules.Image.fromarray(image).save(path)


def save_disparity_png(path: Path, disparity, np_module, image_cls) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    disp = disparity.astype("float32")
    finite = np_module.isfinite(disp)
    if not finite.any():
        scaled = np_module.zeros(disp.shape, dtype=np_module.uint8)
    else:
        lo = float(np_module.nanmin(disp[finite]))
        hi = float(np_module.nanmax(disp[finite]))
        if hi <= lo:
            scaled = np_module.zeros(disp.shape, dtype=np_module.uint8)
        else:
            scaled = ((disp - lo) / (hi - lo) * 255.0).clip(0, 255).astype(np_module.uint8)
    image_cls.fromarray(scaled).save(path)


def flatten_collated_paths(collated_paths) -> List[str]:
    paths: List[str] = []
    for item in collated_paths:
        if isinstance(item, (list, tuple)):
            if len(item) != 1:
                raise ValueError(f"Expected batch_size=1 path item, got: {item}")
            paths.append(str(item[0]))
        else:
            paths.append(str(item))
    return paths


def frame_stem(path: Path) -> str:
    return path.stem


def experiment_name_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("Experiment"):
            return part
    raise ValueError(f"Could not find Experiment* in path: {path}")


def matching_gt_path(source_disp_path: Path, image_path: Path) -> Path:
    stem = frame_stem(image_path)
    candidate = source_disp_path.parent / f"disparity_map_{stem}.npy"
    if not candidate.is_file():
        raise FileNotFoundError(f"Ground-truth disparity not found: {candidate}")
    return candidate


def copy_gt_to_method(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() == destination.resolve():
        return
    shutil.copy2(source, destination)


def make_export_paths(dataset: str, adec_data_root: Path, disparity_root: Path) -> ExportPaths:
    ae_root = adec_data_root / dataset / "ae_methods" / "adec"
    return ExportPaths(
        ae_root=ae_root,
        disparity_root=disparity_root / dataset / "adec",
        val_txt=ae_root / "val.txt",
    )


def relative_to_adec_root(path: Path, adec_data_root: Path) -> str:
    return path.resolve().relative_to(adec_data_root.resolve()).as_posix()


def append_val_line(
    lines: List[str],
    left_path: Path,
    right_path: Path,
    gt_path: Path,
    adec_data_root: Path,
) -> None:
    lines.append(
        " ".join(
            [
                relative_to_adec_root(left_path, adec_data_root),
                relative_to_adec_root(right_path, adec_data_root),
                relative_to_adec_root(gt_path, adec_data_root),
            ]
        )
    )


def load_model(modules, model_args: SimpleNamespace):
    torch = modules.torch
    device = torch.device(model_args.device)
    model = modules.CombineModel(model_args).to(device)
    ensure_file(Path(model_args.restore_ckpt), "Checkpoint")
    checkpoint = torch.load(model_args.restore_ckpt, map_location=device)
    checkpoint = {key.replace("module.", ""): value for key, value in checkpoint.items()}
    model.disp_recon_net.load_state_dict(checkpoint, strict=False)
    model.eval()
    return model, device


def run_worker(args: argparse.Namespace) -> Dict[str, object]:
    if args.dataset is None:
        raise ValueError("--dataset is required in worker mode")
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required in worker mode")

    modules = setup_adec_imports(args.adec_repo)
    torch = modules.torch
    np = modules.np

    torch.manual_seed(1234)
    np.random.seed(1234)

    model_args = make_model_args(args.dataset, args.checkpoint, args.valid_iters)
    model, device = load_model(modules, model_args)

    dataset_root = modules.resolve_carla_root(args.dataset)
    dataset = modules.CARLASequenceDataset(
        root=dataset_root,
        image_set="val",
        apply_blur=False,
    )
    loader = modules.data.DataLoader(
        dataset,
        batch_size=1,
        pin_memory=True,
        shuffle=False,
        num_workers=args.num_workers,
        drop_last=False,
    )

    export_paths = make_export_paths(args.dataset, args.adec_data_root, args.disparity_root)
    export_paths.ae_root.mkdir(parents=True, exist_ok=True)
    export_paths.disparity_root.mkdir(parents=True, exist_ok=True)

    metric_tracker = modules.init_metric_tracker()
    val_lines: List[str] = []
    saved_images = 0
    saved_disparities = 0

    initial_exp1 = torch.tensor([2.0], dtype=torch.float32, device=device).unsqueeze(0)
    initial_exp2 = torch.tensor([2.0], dtype=torch.float32, device=device).unsqueeze(0)

    iterable = modules.tqdm(loader, desc=args.dataset, dynamic_ncols=True)
    for batch_idx, batch in enumerate(iterable):
        if args.limit is not None and batch_idx >= args.limit:
            break

        collated_paths, *data_blob = batch
        paths = flatten_collated_paths(collated_paths)
        left0_src, right0_src, left1_src, right1_src, disp0_src = [Path(p) for p in paths]
        experiment = experiment_name_from_path(left0_src)
        stem0 = frame_stem(left0_src)
        stem1 = frame_stem(left1_src)

        left_hdr, right_hdr, left_next_hdr, right_next_hdr, disp, valid = [
            tensor.to(device) for tensor in data_blob
        ]

        with torch.no_grad():
            (
                fused_disparity,
                _original_img_list,
                _captured_rand_img_list,
                captured_adj_img_list,
                exp1,
                exp2,
                _fmap_list,
                _mask_list,
                _flow_l,
                _fixed_range_middle,
            ) = model(
                left_hdr,
                right_hdr,
                left_next_hdr,
                right_next_hdr,
                initial_exp1,
                initial_exp2,
            )

        batch_metrics = modules.compute_batch_metrics(fused_disparity[-1], disp, valid)
        modules.update_metric_tracker(metric_tracker, batch_metrics)
        initial_exp1, initial_exp2 = exp1, exp2

        frame_exports = [
            (stem0, captured_adj_img_list[0][0], captured_adj_img_list[2][0], disp0_src),
            (
                stem1,
                captured_adj_img_list[1][0],
                captured_adj_img_list[3][0],
                matching_gt_path(disp0_src, left1_src),
            ),
        ]

        for stem, left_tensor, right_tensor, gt_src in frame_exports:
            left_out = export_paths.ae_root / experiment / "hdr_left" / f"{stem}.png"
            right_out = export_paths.ae_root / experiment / "hdr_right" / f"{stem}.png"
            gt_out = (
                export_paths.ae_root
                / experiment
                / "ground_truth_disparity_left"
                / gt_src.name
            )
            save_image_tensor(left_out, left_tensor, modules)
            save_image_tensor(right_out, right_tensor, modules)
            copy_gt_to_method(gt_src, gt_out)
            append_val_line(val_lines, left_out, right_out, gt_out, args.adec_data_root)
            saved_images += 2

        pred = fused_disparity[-1][0].detach().cpu().float().numpy()
        if pred.ndim == 3 and pred.shape[0] == 1:
            pred = pred[0]
        positive_disp = (-pred).astype("float32")
        pred_base = export_paths.disparity_root / experiment / f"disparity_map_{stem0}"
        pred_base.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(pred_base.with_suffix(".npy")), positive_disp)
        save_disparity_png(pred_base.with_suffix(".png"), positive_disp, np, modules.Image)
        saved_disparities += 1

    export_paths.val_txt.write_text("\n".join(val_lines) + "\n", encoding="utf-8")
    summary = {
        "dataset": args.dataset,
        "checkpoint": str(args.checkpoint),
        "samples": len(metric_tracker["epe"]),
        "saved_val_lines": len(val_lines),
        "saved_images": saved_images,
        "saved_disparities": saved_disparities,
        "ae_root": str(export_paths.ae_root),
        "val_txt": str(export_paths.val_txt),
        "disparity_root": str(export_paths.disparity_root),
        "metrics": modules.summarize_metric_tracker(metric_tracker),
    }

    if args.summary_json:
        args.summary_json.parent.mkdir(parents=True, exist_ok=True)
        args.summary_json.write_text(
            json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
        )

    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def checkpoint_for_dataset(adec_repo: Path, dataset: str) -> Path:
    return adec_repo / "checkpoints" / DATASET_CHECKPOINTS[dataset]


def launcher(args: argparse.Namespace) -> int:
    ensure_dir(args.adec_repo, "ADEC repository")
    if len(args.gpus) < len(args.datasets):
        raise ValueError("Number of --gpus must be >= number of --datasets")

    run_dir = args.run_root / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    processes: List[Tuple[str, Path, subprocess.Popen]] = []

    for dataset, gpu in zip(args.datasets, args.gpus):
        checkpoint = checkpoint_for_dataset(args.adec_repo, dataset)
        ensure_file(checkpoint, f"{dataset} checkpoint")
        summary_json = run_dir / f"{dataset}_summary.json"
        log_path = run_dir / f"{dataset}.log"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--dataset",
            dataset,
            "--checkpoint",
            str(checkpoint),
            "--summary-json",
            str(summary_json),
            "--adec-repo",
            str(args.adec_repo),
            "--adec-data-root",
            str(args.adec_data_root),
            "--disparity-root",
            str(args.disparity_root),
            "--num-workers",
            str(args.num_workers),
            "--valid-iters",
            str(args.valid_iters),
        ]
        if args.limit is not None:
            command.extend(["--limit", str(args.limit)])

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env.setdefault("PYTHONUNBUFFERED", "1")
        log_handle = log_path.open("w", encoding="utf-8")
        process = subprocess.Popen(
            command,
            cwd=str(OPENSTEREO_ROOT),
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        log_handle.close()
        print(f"Started {dataset} on GPU {gpu}; log: {log_path}")
        processes.append((dataset, summary_json, process))

    failures = []
    for dataset, summary_json, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failures.append((dataset, return_code))

    summaries = []
    for dataset, summary_json, _process in processes:
        if summary_json.is_file():
            summaries.append(json.loads(summary_json.read_text(encoding="utf-8")))
        else:
            failures.append((dataset, "missing summary"))

    combined_path = run_dir / "summary.json"
    combined_path.write_text(
        json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"\nCombined summary: {combined_path}")
    print_summary_table(summaries)

    if failures:
        print(f"Failures: {failures}", file=sys.stderr)
        return 1
    return 0


def print_summary_table(summaries: Sequence[Dict[str, object]]) -> None:
    if not summaries:
        print("No summaries were produced.")
        return
    headers = ["dataset", "samples", "val_lines", "d1_all", "epe", "thres_1", "thres_2", "thres_3"]
    print(" | ".join(headers))
    print(" | ".join(["---"] * len(headers)))
    for summary in summaries:
        metrics = summary["metrics"]
        row = [
            str(summary["dataset"]),
            str(summary["samples"]),
            str(summary["saved_val_lines"]),
            f"{metrics['d1_all']:.4f}",
            f"{metrics['epe']:.4f}",
            f"{metrics['thres_1']:.4f}",
            f"{metrics['thres_2']:.4f}",
            f"{metrics['thres_3']:.4f}",
        ]
        print(" | ".join(row))


def main() -> int:
    args = parse_args()
    if args.worker:
        run_worker(args)
        return 0
    return launcher(args)


if __name__ == "__main__":
    raise SystemExit(main())
