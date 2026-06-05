#!/usr/bin/env python
"""Run DRL exposure-control inference for ADEC CARLA splits and validate outputs."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


DRL_REPO = Path("/home/lgz/workspace/drl_exposure_ctrl")
ADEC_ROOT = Path("/home/lgz/dataset/ADEC")


@dataclass(frozen=True)
class InferJobSpec:
    dataset_profile: str
    gpu: int
    model_path: Path
    save_root: Path
    reference_split: Path


@dataclass
class InferJob:
    spec: InferJobSpec
    stdout_log: Path
    command: List[str]
    process: subprocess.Popen


def build_job_specs(workspace_root: Path) -> List[InferJobSpec]:
    return [
        InferJobSpec(
            dataset_profile="carla_600x800",
            gpu=7,
            model_path=DRL_REPO / "model/actor_stat_600x800_000009000.pth",
            save_root=ADEC_ROOT / "carla_600x800/ae_methods/drl",
            reference_split=workspace_root / "dataset_split/carla_600x800/val.txt",
        ),
        InferJobSpec(
            dataset_profile="carla_1280x384",
            gpu=6,
            model_path=DRL_REPO / "model/actor_stat_1280x384_000009000.pth",
            save_root=ADEC_ROOT / "carla_1280x384/ae_methods/drl",
            reference_split=workspace_root / "dataset_split/carla_1280x384/val.txt",
        ),
    ]


def count_nonempty_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as file_obj:
        return sum(1 for line in file_obj if line.strip())


def build_command(spec: InferJobSpec) -> List[str]:
    return [
        sys.executable,
        "infer.py",
        "--dataset-profile",
        spec.dataset_profile,
        "--model-path",
        str(spec.model_path),
        "--save-root",
        str(spec.save_root),
        "--device",
        "cuda",
    ]


def start_job(spec: InferJobSpec, run_dir: Path) -> InferJob:
    if not spec.model_path.is_file():
        raise FileNotFoundError(f"Missing actor checkpoint: {spec.model_path}")
    if not spec.reference_split.is_file():
        raise FileNotFoundError(f"Missing reference split: {spec.reference_split}")

    stdout_log = run_dir / f"infer_{spec.dataset_profile}.log"
    command = build_command(spec)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(spec.gpu)
    env.setdefault("PYTHONUNBUFFERED", "1")

    stdout_handle = stdout_log.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(DRL_REPO),
        env=env,
        text=True,
        stdout=stdout_handle,
        stderr=subprocess.STDOUT,
    )
    stdout_handle.close()
    print(
        f"[start] {spec.dataset_profile} gpu={spec.gpu} "
        f"pid={process.pid} model={spec.model_path.name}"
    )
    return InferJob(
        spec=spec,
        stdout_log=stdout_log,
        command=command,
        process=process,
    )


def resolve_output_path(relative_path: str) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return ADEC_ROOT / path


def validate_val_file(spec: InferJobSpec) -> Dict[str, object]:
    val_path = spec.save_root / "val.txt"
    if not val_path.is_file():
        raise FileNotFoundError(f"Missing generated val.txt: {val_path}")

    expected_count = count_nonempty_lines(spec.reference_split)
    line_count = 0
    first_line: Optional[str] = None
    last_line: Optional[str] = None
    missing_paths: List[str] = []

    with val_path.open("r", encoding="utf-8") as file_obj:
        for line_no, raw_line in enumerate(file_obj, start=1):
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 3:
                raise ValueError(f"{val_path}:{line_no} expected 3 columns, got {len(parts)}")
            if any(Path(part).is_absolute() for part in parts):
                raise ValueError(f"{val_path}:{line_no} contains absolute paths: {line}")

            for part in parts:
                if not resolve_output_path(part).exists():
                    missing_paths.append(part)
                    if len(missing_paths) >= 10:
                        break
            if missing_paths:
                break

            first_line = first_line or line
            last_line = line
            line_count += 1

    if missing_paths:
        raise FileNotFoundError(
            f"{val_path} references missing files, first examples: {missing_paths}"
        )
    if line_count != expected_count:
        raise ValueError(
            f"{val_path} line count mismatch: expected {expected_count}, got {line_count}"
        )

    return {
        "dataset_profile": spec.dataset_profile,
        "val_path": str(val_path),
        "line_count": line_count,
        "expected_count": expected_count,
        "first_line": first_line,
        "last_line": last_line,
    }


def collect_job(job: InferJob) -> Dict[str, object]:
    return_code = job.process.wait()
    stdout_tail = ""
    if job.stdout_log.is_file():
        lines = job.stdout_log.read_text(encoding="utf-8", errors="replace").splitlines()
        stdout_tail = "\n".join(lines[-8:])

    print(f"[done]  {job.spec.dataset_profile} gpu={job.spec.gpu} return={return_code}")
    if return_code != 0:
        print(stdout_tail)
        raise RuntimeError(f"inference failed for {job.spec.dataset_profile}")

    validation = validate_val_file(job.spec)
    print(
        f"[valid] {job.spec.dataset_profile} "
        f"{validation['line_count']} lines -> {validation['val_path']}"
    )
    return {
        "dataset_profile": job.spec.dataset_profile,
        "gpu": job.spec.gpu,
        "return_code": return_code,
        "model_path": str(job.spec.model_path),
        "save_root": str(job.spec.save_root),
        "stdout_log": str(job.stdout_log),
        "command": job.command,
        "validation": validation,
        "stdout_tail": stdout_tail,
    }


def parse_args() -> argparse.Namespace:
    workspace_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace-root", type=Path, default=workspace_root)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=workspace_root / "output/drl_infer_runs",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)

    jobs = []
    for spec in build_job_specs(args.workspace_root):
        jobs.append(start_job(spec, run_dir))
        time.sleep(1)

    results = [collect_job(job) for job in jobs]
    summary_path = run_dir / "summary.json"
    summary_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\nFinal summary")
    print("| dataset | gpu | lines | val.txt |")
    print("|---|---:|---:|---|")
    for result in results:
        validation = result["validation"]
        print(
            f"| {result['dataset_profile']} | {result['gpu']} | "
            f"{validation['line_count']} | {validation['val_path']} |"
        )
    print(f"\nSaved run summary to: {summary_path}")


if __name__ == "__main__":
    main()
