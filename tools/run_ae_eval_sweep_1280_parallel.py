#!/usr/bin/env python
"""Run CARLA 1280x384 AE validation jobs in parallel on multiple GPUs."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


METHOD_SPLITS = {
    "pid": Path("/home/lgz/dataset/ADEC/carla_1280x384/ae_methods/pid/val.txt"),
    "semantic": Path("/home/lgz/dataset/ADEC/carla_1280x384/ae_methods/semantic/val.txt"),
    "mixed": Path("/home/lgz/dataset/ADEC/carla_1280x384/ae_methods/mixed/val.txt"),
    "gradient": Path("/home/lgz/dataset/ADEC/carla_1280x384/ae_methods/gradient/val.txt"),
    "exposure_agent": Path("/home/lgz/dataset/ADEC/carla_1280x384/ae_methods/exposure_agent/val.txt"),
}
METRIC_NAMES = ("d1_all", "epe", "thres_1", "thres_2", "thres_3")


@dataclass
class Job:
    method: str
    split_file: Path
    gpu: int
    cfg_file: Path
    save_root: Path
    stdout_log: Path
    command: List[str]
    process: subprocess.Popen


@dataclass(frozen=True)
class JobResult:
    method: str
    split_file: Path
    gpu: int
    return_code: int
    metrics: Dict[str, float]
    stdout_log: Path
    eval_log: Optional[Path]
    renamed_log: Optional[Path]
    command: List[str]


def replace_active_line(lines: List[str], key: str, value: str) -> List[str]:
    replaced = False
    new_lines: List[str] = []
    for line in lines:
        stripped = line.lstrip()
        if not replaced and stripped.startswith(f"{key}:"):
            indent = line[: len(line) - len(stripped)]
            comma = "," if line.rstrip().endswith(",") else ""
            new_lines.append(f"{indent}{key}: {value}{comma}")
            replaced = True
        else:
            new_lines.append(line)

    if not replaced:
        raise ValueError(f"No active {key} line found.")
    return new_lines


def replace_active_right_top_pad(lines: List[str], size: str) -> List[str]:
    replaced = False
    pattern = re.compile(r"^(\s*)-\s*\{\s*NAME:\s*RightTopPad,\s*SIZE:")
    new_lines: List[str] = []
    for line in lines:
        if not replaced and pattern.search(line):
            indent = pattern.search(line).group(1)
            new_lines.append(f"{indent}- {{ NAME: RightTopPad, SIZE: {size} }}")
            replaced = True
        else:
            new_lines.append(line)

    if not replaced:
        raise ValueError("No active RightTopPad transform found.")
    return new_lines


def write_eval_cfg(template_cfg: Path, output_cfg: Path, split_file: Path) -> None:
    if not template_cfg.is_file():
        raise FileNotFoundError(f"Missing template eval config: {template_cfg}")
    if not split_file.is_file():
        raise FileNotFoundError(f"Missing split file: {split_file}")

    lines = template_cfg.read_text(encoding="utf-8").splitlines()
    lines = replace_active_line(lines, "EVALUATING", str(split_file))
    lines = replace_active_right_top_pad(lines, "[ 384, 1280 ]")
    output_cfg.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_metrics(text: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for name in METRIC_NAMES:
        match = re.search(rf"'{re.escape(name)}': tensor\(([-+0-9.eE]+)", text)
        if match:
            metrics[name] = float(match.group(1))
    return metrics


def find_eval_log(save_root: Path) -> Optional[Path]:
    eval_dir = save_root / "CarlaStereoDataset" / "RAFTStereo" / "eval"
    logs = sorted(eval_dir.glob("eval_*.log"), key=lambda path: path.stat().st_mtime)
    if not logs:
        return None
    return logs[-1]


def build_command(args: argparse.Namespace, eval_cfg: Path, save_root: Path) -> List[str]:
    return [
        sys.executable,
        "tools/eval.py",
        "--cfg_file",
        str(args.cfg_file),
        "--eval_data_cfg_file",
        str(eval_cfg),
        "--pretrained_model",
        str(args.pretrained_model),
        "--save_root_dir",
        str(save_root),
    ]


def start_job(
    method: str,
    split_file: Path,
    gpu: int,
    args: argparse.Namespace,
    run_dir: Path,
) -> Job:
    cfg_file = run_dir / "cfgs" / f"carla_eval_1280x384_{method}.yaml"
    save_root = run_dir / "eval_outputs" / method
    stdout_log = run_dir / f"{method}.stdout.log"
    save_root.mkdir(parents=True, exist_ok=True)
    write_eval_cfg(args.eval_data_cfg_file, cfg_file, split_file)

    command = build_command(args, cfg_file, save_root)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env.setdefault("PYTHONUNBUFFERED", "1")

    stdout_handle = stdout_log.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(args.workspace_root),
        env=env,
        text=True,
        stdout=stdout_handle,
        stderr=subprocess.STDOUT,
    )
    stdout_handle.close()
    print(f"[start] {method:14s} gpu={gpu} pid={process.pid} split={split_file}")
    return Job(
        method=method,
        split_file=split_file,
        gpu=gpu,
        cfg_file=cfg_file,
        save_root=save_root,
        stdout_log=stdout_log,
        command=command,
        process=process,
    )


def collect_job(job: Job, central_eval_dir: Path) -> JobResult:
    return_code = job.process.wait()
    stdout_text = job.stdout_log.read_text(encoding="utf-8", errors="replace")
    metrics = parse_metrics(stdout_text)
    eval_log = find_eval_log(job.save_root)
    renamed_log = None
    if eval_log is not None:
        central_eval_dir.mkdir(parents=True, exist_ok=True)
        renamed_log = central_eval_dir / f"1280x384_{job.method}.log"
        shutil.copy2(eval_log, renamed_log)
        if not metrics:
            metrics = parse_metrics(eval_log.read_text(encoding="utf-8", errors="replace"))

    status = "ok" if return_code == 0 else "failed"
    metric_text = ", ".join(f"{name}={metrics[name]:.6f}" for name in metrics)
    print(f"[done]  {job.method:14s} gpu={job.gpu} status={status} {metric_text}")
    if renamed_log is not None:
        print(f"        log -> {renamed_log}")

    return JobResult(
        method=job.method,
        split_file=job.split_file,
        gpu=job.gpu,
        return_code=return_code,
        metrics=metrics,
        stdout_log=job.stdout_log,
        eval_log=eval_log,
        renamed_log=renamed_log,
        command=job.command,
    )


def write_summary(results: List[JobResult], run_dir: Path) -> None:
    rows = []
    for result in results:
        rows.append(
            {
                "method": result.method,
                "gpu": result.gpu,
                "return_code": result.return_code,
                "split_file": str(result.split_file),
                "metrics": result.metrics,
                "stdout_log": str(result.stdout_log),
                "eval_log": str(result.eval_log) if result.eval_log else None,
                "renamed_log": str(result.renamed_log) if result.renamed_log else None,
                "command": result.command,
            }
        )
    (run_dir / "summary.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    header = ["method", "gpu", "return_code", *METRIC_NAMES, "renamed_log"]
    csv_rows = [",".join(header)]
    for result in results:
        csv_rows.append(
            ",".join(
                [
                    result.method,
                    str(result.gpu),
                    str(result.return_code),
                    *[
                        f"{result.metrics[name]:.8f}"
                        if name in result.metrics
                        else ""
                        for name in METRIC_NAMES
                    ],
                    str(result.renamed_log) if result.renamed_log else "",
                ]
            )
        )
    (run_dir / "summary.csv").write_text("\n".join(csv_rows) + "\n", encoding="utf-8")

    print("\nFinal summary")
    print("| method | gpu | return | d1_all | epe | thres_1 | thres_2 | thres_3 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for result in results:
        metric_values = [
            f"{result.metrics[name]:.6f}" if name in result.metrics else "NA"
            for name in METRIC_NAMES
        ]
        print(
            f"| {result.method} | {result.gpu} | {result.return_code} | "
            f"{metric_values[0]} | {metric_values[1]} | {metric_values[2]} | "
            f"{metric_values[3]} | {metric_values[4]} |"
        )
    print(f"\nSaved summary to: {run_dir}")


def parse_args() -> argparse.Namespace:
    workspace_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=workspace_root)
    parser.add_argument(
        "--cfg-file",
        type=Path,
        default=workspace_root / "cfgs/raftstereo/raftstereo_carla_1280x384_rgb.yaml",
    )
    parser.add_argument(
        "--eval-data-cfg-file",
        type=Path,
        default=workspace_root / "cfgs/carla_eval.yaml",
    )
    parser.add_argument(
        "--pretrained-model",
        type=Path,
        default=(
            workspace_root
            / "output/CarlaStereoDataset/RAFTStereo/raftstereo_carla_1280x384_rgb"
            / "default/ckpt/checkpoint_epoch_19.pth"
        ),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["pid", "semantic", "mixed", "gradient", "exposure_agent"],
        choices=sorted(METHOD_SPLITS.keys()),
    )
    parser.add_argument(
        "--gpus",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3, 4],
        help="GPU ids used in round-robin order.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=workspace_root / "output/ae_eval_runs_1280x384",
    )
    parser.add_argument(
        "--central-eval-dir",
        type=Path,
        default=workspace_root / "output/CarlaStereoDataset/RAFTStereo/eval",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    (run_dir / "cfgs").mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.eval_data_cfg_file, run_dir / f"{args.eval_data_cfg_file.name}.bak")

    jobs: List[Job] = []
    for index, method in enumerate(args.methods):
        jobs.append(
            start_job(
                method=method,
                split_file=METHOD_SPLITS[method],
                gpu=args.gpus[index % len(args.gpus)],
                args=args,
                run_dir=run_dir,
            )
        )
        time.sleep(1)

    results = [collect_job(job, args.central_eval_dir) for job in jobs]
    write_summary(results, run_dir)

    failed = [result.method for result in results if result.return_code != 0]
    if failed:
        raise SystemExit(f"Evaluation failed for: {', '.join(failed)}")


if __name__ == "__main__":
    main()
