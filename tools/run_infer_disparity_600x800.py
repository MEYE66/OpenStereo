#!/usr/bin/env python3
"""Batch RAFTStereo disparity visualization for CARLA 600x800 AE methods."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


METHODS = ("pid", "semantic", "mixed", "gradient", "exposure_agent", "drl")
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path("/home/lgz/dataset/ADEC")
DEFAULT_CFG = Path("cfgs/raftstereo/raftstereo_carla_600x800_rgb.yaml")
DEFAULT_CKPT = Path(
    "./output/CarlaStereoDataset/RAFTStereo/"
    "raftstereo_carla_600x800_rgb/default/ckpt/checkpoint_epoch_19.pth"
)
DEFAULT_OUTPUT_ROOT = Path("output_disparity/carla_600x800")
DEFAULT_RUN_ROOT = Path("output/infer_disparity_runs")


@dataclass
class Job:
    method: str
    gpu: str
    cfg_path: Path
    output_dir: Path
    log_path: Path
    process: Optional[subprocess.Popen] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tools/infer.py for CARLA 600x800 AE methods."
    )
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    parser.add_argument("--gpus", nargs="+", default=["7", "6", "5", "4"])
    parser.add_argument("--cfg-file", type=Path, default=DEFAULT_CFG)
    parser.add_argument("--pretrained-model", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def check_inputs(args: argparse.Namespace) -> None:
    if not args.cfg_file.is_file():
        raise FileNotFoundError(f"Model cfg not found: {args.cfg_file}")
    if not args.pretrained_model.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.pretrained_model}")
    for method in args.methods:
        val_txt = DATA_ROOT / "carla_600x800" / "ae_methods" / method / "val.txt"
        if not val_txt.is_file():
            raise FileNotFoundError(f"{method} val.txt not found: {val_txt}")


def write_eval_cfg(run_dir: Path, method: str, batch_size: int) -> Path:
    val_txt = DATA_ROOT / "carla_600x800" / "ae_methods" / method / "val.txt"
    cfg_path = run_dir / "cfgs" / f"carla_600x800_{method}.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(
        "\n".join(
            [
                "DATA_CONFIG:",
                "    DATA_INFOS:",
                "        -   DATASET: CarlaStereoDataset",
                "            DATA_SPLIT: {",
                f"                EVALUATING: {val_txt.as_posix()},",
                "            }",
                "",
                "            RETURN_RIGHT_DISP: false",
                "            ENABLE_RGB: false",
                "            MINMAX_NORM: false",
                "",
                "    DATA_TRANSFORM:",
                "        EVALUATING:",
                "            - { NAME: RightTopPad, SIZE: [ 608, 800 ] }",
                "            - { NAME: TransposeImage }",
                "            - { NAME: ToTensor }",
                "",
                "EVALUATOR:",
                f"    BATCH_SIZE_PER_GPU: {batch_size}",
                "    APPLY_OCC_MASK: true",
                "    MAX_DISP: 192",
                "    METRIC:",
                "        - d1_all",
                "        - epe",
                "        - thres_1",
                "        - thres_2",
                "        - thres_3",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return cfg_path


def build_jobs(args: argparse.Namespace, run_dir: Path) -> List[Job]:
    jobs: List[Job] = []
    for index, method in enumerate(args.methods):
        cfg_path = write_eval_cfg(run_dir, method, args.batch_size)
        jobs.append(
            Job(
                method=method,
                gpu=args.gpus[index % len(args.gpus)],
                cfg_path=cfg_path,
                output_dir=args.output_root / method,
                log_path=run_dir / f"{method}.log",
            )
        )
    return jobs


def start_job(job: Job, args: argparse.Namespace) -> None:
    job.output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "tools/infer.py",
        "--cfg_file",
        str(args.cfg_file),
        "--eval_data_cfg_file",
        str(job.cfg_path),
        "--pretrained_model",
        str(args.pretrained_model),
        "--save_root_dir",
        str(job.output_dir),
        "--workers",
        str(args.workers),
        "--pin_memory",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = job.gpu
    env.setdefault("PYTHONUNBUFFERED", "1")
    log_handle = job.log_path.open("w", encoding="utf-8")
    job.process = subprocess.Popen(
        command,
        cwd=str(WORKSPACE_ROOT),
        env=env,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    log_handle.close()
    print(f"Started {job.method} on GPU {job.gpu}; log: {job.log_path}")


def output_count(output_dir: Path) -> int:
    return len(list(output_dir.glob("Experiment*/disparity_map_*.png")))


def run_jobs(args: argparse.Namespace, jobs: List[Job]) -> Dict[str, Dict[str, object]]:
    pending = list(jobs)
    running: List[Job] = []
    completed: Dict[str, Dict[str, object]] = {}
    max_parallel = min(len(args.gpus), len(jobs))

    while pending or running:
        while pending and len(running) < max_parallel:
            job = pending.pop(0)
            start_job(job, args)
            running.append(job)

        time.sleep(5)
        still_running: List[Job] = []
        for job in running:
            assert job.process is not None
            return_code = job.process.poll()
            if return_code is None:
                still_running.append(job)
                continue
            count = output_count(job.output_dir)
            completed[job.method] = {
                "gpu": job.gpu,
                "return_code": return_code,
                "output_dir": str(job.output_dir),
                "png_count": count,
                "log_path": str(job.log_path),
            }
            print(f"Finished {job.method}: return_code={return_code}, png_count={count}")
        running = still_running

    return completed


def main() -> int:
    args = parse_args()
    check_inputs(args)

    run_dir = args.run_root / time.strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args, run_dir)
    summary = run_jobs(args, jobs)

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print(f"\nSummary: {summary_path}")
    print("method | gpu | return_code | png_count | output_dir")
    print("--- | --- | --- | --- | ---")
    failed = False
    for method in args.methods:
        row = summary[method]
        print(
            f"{method} | {row['gpu']} | {row['return_code']} | "
            f"{row['png_count']} | {row['output_dir']}"
        )
        failed = failed or row["return_code"] != 0
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
