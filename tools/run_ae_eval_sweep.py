#!/usr/bin/env python
"""Run RAFTStereo evaluation on several CARLA AE method split files."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


METHOD_SPLITS = {
    "pid": Path("/home/lgz/dataset/ADEC/carla_600x800/ae_methods/ae_pid_pid_entropy/val.txt"),
    "semantic": Path("/home/lgz/dataset/ADEC/carla_600x800/ae_methods/semantic/val.txt"),
    "mixed": Path("/home/lgz/dataset/ADEC/carla_600x800/ae_methods/mixed/val.txt"),
    "gradient": Path("/home/lgz/dataset/ADEC/carla_600x800/ae_methods/gradient/val.txt"),
    "exposure_agent": Path("/home/lgz/dataset/ADEC/carla_600x800/ae_methods/exposure_agent/val.txt"),
}

METRIC_NAMES = ("d1_all", "epe", "thres_1", "thres_2", "thres_3")


@dataclass(frozen=True)
class EvalResult:
    method: str
    split_file: Path
    return_code: int
    metrics: Dict[str, float]
    output_log: Path
    command: List[str]


def replace_evaluating_path(eval_cfg: Path, split_file: Path) -> None:
    if not eval_cfg.is_file():
        raise FileNotFoundError(f"Evaluation config does not exist: {eval_cfg}")
    if not split_file.is_file():
        raise FileNotFoundError(f"Split file does not exist: {split_file}")

    lines = eval_cfg.read_text(encoding="utf-8").splitlines()
    replaced = False
    new_lines: List[str] = []
    for line in lines:
        stripped = line.lstrip()
        if not replaced and stripped.startswith("EVALUATING:"):
            indent = line[: len(line) - len(stripped)]
            comma = "," if line.rstrip().endswith(",") else ""
            new_lines.append(f"{indent}EVALUATING: {split_file}{comma}")
            replaced = True
        else:
            new_lines.append(line)

    if not replaced:
        raise ValueError(f"No active EVALUATING line found in {eval_cfg}")

    eval_cfg.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def backup_config(eval_cfg: Path, output_dir: Path) -> Path:
    backup_path = output_dir / f"{eval_cfg.name}.bak"
    shutil.copy2(eval_cfg, backup_path)
    return backup_path


def build_eval_command(args: argparse.Namespace) -> List[str]:
    return [
        sys.executable,
        "tools/eval.py",
        "--cfg_file",
        str(args.cfg_file),
        "--eval_data_cfg_file",
        str(args.eval_data_cfg_file),
        "--pretrained_model",
        str(args.pretrained_model),
    ]


def parse_metrics(text: str) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for name in METRIC_NAMES:
        pattern = rf"'{re.escape(name)}': tensor\(([-+0-9.eE]+)"
        match = re.search(pattern, text)
        if match:
            metrics[name] = float(match.group(1))
    return metrics


def run_one_eval(
    method: str,
    split_file: Path,
    args: argparse.Namespace,
    output_dir: Path,
) -> EvalResult:
    replace_evaluating_path(args.eval_data_cfg_file, split_file)

    command = build_eval_command(args)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    env.setdefault("PYTHONUNBUFFERED", "1")

    print(f"\n[{method}] EVALUATING = {split_file}", flush=True)
    print(f"[{method}] command: CUDA_VISIBLE_DEVICES={args.gpu} {' '.join(command)}", flush=True)

    process = subprocess.run(
        command,
        cwd=str(args.workspace_root),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )

    output_log = output_dir / f"{method}.log"
    output_log.write_text(process.stdout, encoding="utf-8")
    metrics = parse_metrics(process.stdout)

    print(f"[{method}] return_code = {process.returncode}", flush=True)
    if metrics:
        metric_text = ", ".join(f"{name}={metrics[name]:.6f}" for name in metrics)
        print(f"[{method}] metrics: {metric_text}", flush=True)
    else:
        print(f"[{method}] metrics were not parsed; see {output_log}", flush=True)

    return EvalResult(
        method=method,
        split_file=split_file,
        return_code=process.returncode,
        metrics=metrics,
        output_log=output_log,
        command=command,
    )


def write_summary(results: List[EvalResult], output_dir: Path) -> None:
    json_rows = [
        {
            "method": result.method,
            "split_file": str(result.split_file),
            "return_code": result.return_code,
            "metrics": result.metrics,
            "output_log": str(result.output_log),
            "command": result.command,
        }
        for result in results
    ]
    (output_dir / "summary.json").write_text(
        json.dumps(json_rows, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    header = ["method", "return_code", *METRIC_NAMES, "log"]
    rows = [",".join(header)]
    for result in results:
        values = [
            result.method,
            str(result.return_code),
            *[
                f"{result.metrics[name]:.8f}" if name in result.metrics else ""
                for name in METRIC_NAMES
            ],
            str(result.output_log),
        ]
        rows.append(",".join(values))
    (output_dir / "summary.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    print("\nFinal summary")
    print("| method | return | d1_all | epe | thres_1 | thres_2 | thres_3 |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for result in results:
        metric_values = [
            f"{result.metrics[name]:.6f}" if name in result.metrics else "NA"
            for name in METRIC_NAMES
        ]
        print(
            f"| {result.method} | {result.return_code} | "
            f"{metric_values[0]} | {metric_values[1]} | {metric_values[2]} | "
            f"{metric_values[3]} | {metric_values[4]} |"
        )
    print(f"\nSaved summary to: {output_dir}")


def parse_args() -> argparse.Namespace:
    workspace_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=workspace_root)
    parser.add_argument(
        "--cfg-file",
        type=Path,
        default=workspace_root / "cfgs/raftstereo/raftstereo_carla_600x800_rgb.yaml",
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
            / "output/CarlaStereoDataset/RAFTStereo/raftstereo_carla_600x800_rgb"
            / "default/ckpt/checkpoint_epoch_19.pth"
        ),
    )
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["pid", "semantic", "mixed", "gradient", "exposure_agent"],
        choices=sorted(METHOD_SPLITS.keys()),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=workspace_root / "output/ae_eval_runs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    backup_path = backup_config(args.eval_data_cfg_file, run_dir)
    print(f"Backed up config to: {backup_path}")

    results: List[EvalResult] = []
    try:
        for method in args.methods:
            results.append(run_one_eval(method, METHOD_SPLITS[method], args, run_dir))
    finally:
        write_summary(results, run_dir)

    failed = [result.method for result in results if result.return_code != 0]
    if failed:
        raise SystemExit(f"Evaluation failed for: {', '.join(failed)}")


if __name__ == "__main__":
    main()
