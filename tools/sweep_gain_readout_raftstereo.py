import argparse
import copy
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stereo.modeling.models.ae_util import ImageFormationModel, dB_to_ratio, radiance_scale  # noqa: E402


METRIC_KEYS = ['d1_all', 'epe', 'thres_1', 'thres_2', 'thres_3']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Sweep gain and readout noise for RAFTStereo on Carla Experiment1.')
    parser.add_argument(
        '--cfg-file',
        type=str,
        default=str(REPO_ROOT / 'cfgs' / 'raftstereo' / 'raftstereo_carla_600x800_rgb.yaml'),
        help='Base model config used by tools/eval.py.')
    parser.add_argument(
        '--pretrained-model',
        type=str,
        default=str(
            REPO_ROOT / 'output' / 'CarlaStereoDataset' / 'RAFTStereo' /
            'raftstereo_carla_600x800_rgb' / 'default' / 'ckpt' / 'checkpoint_epoch_19.pth'
        ),
        help='Checkpoint used for evaluation.')
    parser.add_argument(
        '--split-file',
        type=str,
        default=str(REPO_ROOT / 'dataset_split' / 'carla_600x800' / 'val.txt'),
        help='Source split file. Only Experiment1 entries are used by default.')
    parser.add_argument(
        '--dataset-root',
        type=str,
        default='/home/lgz/dataset/ADEC/',
        help='Dataset root corresponding to CarlaStereoDataset.')
    parser.add_argument(
        '--experiment-name',
        type=str,
        nargs='+',
        default=['Experiment1'],
        help='Only entries containing any of these exact path tokens will be evaluated.')
    parser.add_argument(
        '--aggregate-level',
        type=str,
        choices=['sample', 'experiment'],
        default='experiment',
        help='How to aggregate metrics when multiple experiments are selected.')
    parser.add_argument(
        '--output-root',
        type=str,
        default=str(REPO_ROOT / 'output' / 'gain_readout_sweeps' / 'raftstereo_carla_600x800_experiment1'),
        help='Directory used for generated cases and aggregated metrics.')
    parser.add_argument(
        '--exposure-ms',
        type=float,
        default=10.0,
        help='Exposure time passed to ImageFormationModel.')
    parser.add_argument(
        '--fixed-gaussian-var',
        type=float,
        default=5e1,
        help='Readout-noise variance used during the gain sweep.')
    parser.add_argument(
        '--fixed-gain',
        type=float,
        default=1.0,
        help='Gain in dB used during the gaussian-variance sweep.')
    parser.add_argument(
        '--gain-values',
        type=float,
        nargs='+',
        default=[1.0, 2.0, 4.0, 8.0, 12.0, 16.0],
        help='Gain values in dB scanned while gaussian_var is fixed.')
    parser.add_argument(
        '--gaussian-values',
        type=float,
        nargs='+',
        default=[1.0e-3, 1.0e-2, 1.0e-1, 3.0e-1, 1.0, 3.0],
        help='Readout-noise variances scanned while gain is fixed.')
    parser.add_argument(
        '--poisson-scale',
        type=float,
        default=3.3e-4,
        help='Poisson scale used by ImageFormationModel.')
    parser.add_argument(
        '--radiance-capacity',
        type=float,
        default=12.0,
        help='Capacity used by radiance_scale before image formation.')
    parser.add_argument(
        '--disable-radiance-scale',
        action='store_true',
        help='Skip radiance_scale and feed normalized HDR directly into ImageFormationModel.')
    parser.add_argument(
        '--disable-motion-blur',
        action='store_true',
        help='Disable motion blur to isolate sensor-noise effects.')
    parser.add_argument(
        '--motion-angle-deg',
        type=float,
        default=10.0,
        help='Motion blur angle in degrees.')
    parser.add_argument(
        '--motion-velocity-scale',
        type=float,
        default=0.8,
        help='Exposure-to-kernel-length scale for motion blur.')
    parser.add_argument(
        '--motion-min-length',
        type=int,
        default=1,
        help='Minimum odd kernel length for motion blur.')
    parser.add_argument(
        '--motion-max-length',
        type=int,
        default=13,
        help='Maximum odd kernel length for motion blur.')
    parser.add_argument(
        '--motion-canvas-size',
        type=int,
        default=31,
        help='Canvas size used to rasterize the motion PSF.')
    parser.add_argument(
        '--motion-line-sigma',
        type=float,
        default=0.55,
        help='Gaussian width of the motion PSF line profile.')
    parser.add_argument(
        '--motion-edge-softness',
        type=float,
        default=0.75,
        help='Softness of the motion PSF endpoints.')
    generated_gtm_group = parser.add_mutually_exclusive_group()
    generated_gtm_group.add_argument(
        '--enable-generated-gtm',
        dest='enable_generated_gtm',
        action='store_true',
        help='Apply GTM after loading generated npy inputs before RAFTStereo.')
    generated_gtm_group.add_argument(
        '--disable-generated-gtm',
        dest='enable_generated_gtm',
        action='store_false',
        help='Deprecated alias. Generated-input GTM is disabled by default.')
    parser.add_argument(
        '--skip-reference',
        action='store_true',
        help='Skip evaluation on the original Experiment1 HDR inputs.')
    parser.add_argument(
        '--skip-gain-sweep',
        action='store_true',
        help='Skip the gain sweep.')
    parser.add_argument(
        '--skip-gaussian-sweep',
        action='store_true',
        help='Skip the gaussian_var sweep.')
    parser.add_argument(
        '--keep-generated',
        action='store_true',
        help='Keep per-case generated left/right arrays after evaluation.')
    parser.add_argument(
        '--force',
        action='store_true',
        help='Regenerate cases and rerun evaluation even if metrics already exist.')
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Optional cap for Experiment1 samples, useful for smoke runs.')
    parser.add_argument(
        '--workers',
        type=int,
        default=0,
        help='workers argument passed to tools/eval.py.')
    parser.add_argument(
        '--batch-size-per-gpu',
        type=int,
        default=4,
        help='Batch size written into generated eval yamls.')
    parser.add_argument(
        '--cuda-visible-devices',
        type=str,
        default=None,
        help='Optional CUDA_VISIBLE_DEVICES passed to eval subprocesses, for example 7.')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Base random seed used for synthetic image generation.')
    parser.add_argument(
        '--nbits',
        type=int,
        default=10,
        help='Quantization bits for ImageFormationModel.')
    parser.set_defaults(enable_generated_gtm=False)
    return parser.parse_args()


def slugify_float(value):
    if value == 0:
        return '0'
    text = f'{value:.6g}'
    text = text.replace('+', '')
    text = text.replace('.', 'p')
    return text


def load_yaml(path):
    with open(path, 'r') as handle:
        return yaml.safe_load(handle)


def ensure_dir(path):
    path.mkdir(parents=True, exist_ok=True)


def normalize_experiment_names(experiment_name):
    if isinstance(experiment_name, str):
        tokens = [experiment_name]
    else:
        tokens = list(experiment_name)

    normalized = []
    seen = set()
    for token in tokens:
        token = str(token).strip()
        if not token or token in seen:
            continue
        normalized.append(token)
        seen.add(token)
    if not normalized:
        raise ValueError('At least one experiment token is required.')
    return normalized


def extract_experiment_name(entry):
    left_rel = entry['left_rel'] if isinstance(entry, dict) else str(entry)
    for token in Path(left_rel).parts:
        if token.startswith('Experiment'):
            return token
    raise ValueError(f'Could not infer experiment name from path: {left_rel}')


def experiment_sort_key(name):
    match = re.search(r'(\d+)$', str(name))
    if match is None:
        return (1, str(name))
    return (0, int(match.group(1)))


def group_entries_by_experiment(entries):
    grouped = {}
    for entry in entries:
        experiment_name = extract_experiment_name(entry)
        grouped.setdefault(experiment_name, []).append(entry)
    return {
        key: grouped[key]
        for key in sorted(grouped.keys(), key=experiment_sort_key)
    }


def should_aggregate_by_experiment(args, entries):
    if args.aggregate_level != 'experiment':
        return False
    return len(group_entries_by_experiment(entries)) > 1


def build_aggregated_row(case_name, gain, gaussian_var, exposure_ms, effective_readout_std, per_experiment_rows, args):
    if not per_experiment_rows:
        raise ValueError('Expected at least one per-experiment row to aggregate.')

    row = {
        'sweep_type': per_experiment_rows[0]['sweep_type'],
        'case_name': case_name,
        'gain': gain,
        'gaussian_var': gaussian_var,
        'exposure_ms': exposure_ms,
        'effective_readout_std': effective_readout_std,
        'aggregation_level': args.aggregate_level,
        'experiment_count': len(per_experiment_rows),
        'sample_count': sum(item['sample_count'] for item in per_experiment_rows),
    }
    for key in METRIC_KEYS:
        row[key] = sum(item[key] for item in per_experiment_rows) / float(len(per_experiment_rows))
    return row


def load_split_entries(split_path, dataset_root, experiment_name, max_samples=None):
    experiment_names = set(normalize_experiment_names(experiment_name))
    entries = []
    with open(split_path, 'r') as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split(' ')
            if len(parts) != 3:
                raise ValueError(f'Unexpected split entry: {line}')
            left_rel, right_rel, disp_rel = parts
            left_rel_path = Path(left_rel)
            if not any(token in left_rel_path.parts for token in experiment_names):
                continue
            left_abs = Path(dataset_root) / left_rel
            right_abs = Path(dataset_root) / right_rel
            disp_abs = Path(dataset_root) / disp_rel
            entries.append({
                'left_rel': left_rel,
                'right_rel': right_rel,
                'disp_rel': disp_rel,
                'left_abs': left_abs,
                'right_abs': right_abs,
                'disp_abs': disp_abs,
            })
            if max_samples is not None and len(entries) >= max_samples:
                break
    if not entries:
        raise ValueError(f'No entries matching {experiment_name} were found in {split_path}.')
    return entries


def load_hdr_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f'Failed to read HDR image: {path}')
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    if img.shape[2] >= 3:
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)
    return img.astype(np.float32)


def normalize_hdr(img):
    img = img.astype(np.float32)
    max_val = float(img.max())
    if max_val <= 0:
        return img
    return img / max_val


def make_renderer(args):
    motion_blur_params = {
        'velocity_scale': float(args.motion_velocity_scale),
        'min_length': int(args.motion_min_length),
        'max_length': int(args.motion_max_length),
        'canvas_size': int(args.motion_canvas_size),
        'line_sigma': float(args.motion_line_sigma),
        'edge_softness': float(args.motion_edge_softness),
    }
    renderer = ImageFormationModel(
        nbits=args.nbits,
        motion_blur=not args.disable_motion_blur,
        motion_angle_deg=float(args.motion_angle_deg),
        motion_blur_params=motion_blur_params,
    ).cpu()
    renderer.eval()
    with torch.no_grad():
        renderer.poisson_scale.fill_(float(args.poisson_scale))
    return renderer


def set_case_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def render_noisy_image(renderer, hdr_rgb, exposure_ms, gain, gaussian_var, use_radiance_scale, capacity):
    hdr_rgb = normalize_hdr(hdr_rgb)
    if use_radiance_scale:
        hdr_rgb = radiance_scale(hdr_rgb, capacity=capacity)

    tensor = torch.from_numpy(hdr_rgb).permute(2, 0, 1).unsqueeze(0).float()
    exposure_tensor = torch.tensor([exposure_ms], dtype=torch.float32)
    gain_tensor = torch.tensor([gain], dtype=torch.float32)
    with torch.no_grad():
        renderer.gaussian_var.fill_(float(gaussian_var))
        rendered = renderer(tensor, exposure_tensor, gain_tensor)
    rendered = rendered.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return rendered.astype(np.float32)


def write_split_file(path, rows):
    with open(path, 'w') as handle:
        for row in rows:
            handle.write(f"{row['left']} {row['right']} {row['disp']}\n")


def build_eval_config(base_cfg, split_path, batch_size, enable_rgb):
    eval_cfg = {
        'DATA_CONFIG': copy.deepcopy(base_cfg['DATA_CONFIG']),
        'EVALUATOR': copy.deepcopy(base_cfg['EVALUATOR']),
    }
    data_info = eval_cfg['DATA_CONFIG']['DATA_INFOS'][0]
    data_info['DATA_SPLIT']['EVALUATING'] = str(split_path)
    data_info['DATA_SPLIT']['TESTING'] = str(split_path)
    data_info['ENABLE_RGB'] = bool(enable_rgb)
    eval_cfg['EVALUATOR']['BATCH_SIZE_PER_GPU'] = int(batch_size)
    return eval_cfg


def write_yaml(path, content):
    with open(path, 'w') as handle:
        yaml.safe_dump(content, handle, sort_keys=False)


def parse_metrics(eval_output_text):
    metrics_line = None
    for line in reversed(eval_output_text.splitlines()):
        if 'Epoch 0 metrics:' in line:
            metrics_line = line
            break
    if metrics_line is None:
        raise RuntimeError('Could not find metrics line in eval output.')

    metrics = {}
    matches = re.findall(r"'([^']+)': tensor\(([^)]+)\)", metrics_line)
    for key, value in matches:
        if key in METRIC_KEYS:
            metrics[key] = float(value)
    missing = [key for key in METRIC_KEYS if key not in metrics]
    if missing:
        raise RuntimeError(f'Failed to parse metrics {missing} from line: {metrics_line}')
    return metrics


def run_eval(case_dir, args, eval_yaml_path):
    run_dir = case_dir / 'run'
    ensure_dir(run_dir)
    command = [
        sys.executable,
        str(REPO_ROOT / 'tools' / 'eval.py'),
        '--cfg_file', str(args.cfg_file),
        '--eval_data_cfg_file', str(eval_yaml_path),
        '--pretrained_model', str(args.pretrained_model),
        '--workers', str(args.workers),
        '--save_root_dir', str(run_dir),
    ]

    env = os.environ.copy()
    if args.cuda_visible_devices is not None:
        env['CUDA_VISIBLE_DEVICES'] = str(args.cuda_visible_devices)

    completed = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    case_log_path = case_dir / 'eval_stdout.log'
    with open(case_log_path, 'w') as handle:
        handle.write(completed.stdout)

    if completed.returncode != 0:
        raise RuntimeError(
            f'Evaluation failed for {case_dir.name}. See {case_log_path} for details.')

    return parse_metrics(completed.stdout)


def compute_effective_readout_std(gaussian_var, exposure_ms, gain_db):
    gain_ratio = float(dB_to_ratio(float(gain_db)))
    return math.sqrt(max(gaussian_var, 0.0)) * exposure_ms * gain_ratio


def attach_deltas(rows):
    if not rows:
        return rows
    baseline = rows[0]
    for row in rows:
        for key in METRIC_KEYS:
            delta = row[key] - baseline[key]
            row[f'{key}_delta'] = delta
            row[f'{key}_delta_pct'] = 0.0 if baseline[key] == 0 else delta / baseline[key] * 100.0
    return rows


def write_rows_csv(path, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_rows_json(path, rows):
    with open(path, 'w') as handle:
        json.dump(rows, handle, indent=2)


def write_json(path, content):
    with open(path, 'w') as handle:
        json.dump(content, handle, indent=2)


def create_reference_case_single(case_dir, entries, base_cfg, args):
    ensure_dir(case_dir)
    split_path = case_dir / 'Experiment1_eval.txt'
    eval_yaml_path = case_dir / 'eval.yaml'

    split_rows = []
    for entry in entries:
        split_rows.append({
            'left': str(entry['left_abs']),
            'right': str(entry['right_abs']),
            'disp': str(entry['disp_abs']),
        })

    write_split_file(split_path, split_rows)
    eval_cfg = build_eval_config(
        base_cfg=base_cfg,
        split_path=split_path,
        batch_size=args.batch_size_per_gpu,
        enable_rgb=base_cfg['DATA_CONFIG']['DATA_INFOS'][0].get('ENABLE_RGB', True),
    )
    write_yaml(eval_yaml_path, eval_cfg)
    metrics = run_eval(case_dir, args, eval_yaml_path)
    row = {
        'sweep_type': 'reference',
        'case_name': case_dir.name,
        'gain': None,
        'gaussian_var': None,
        'exposure_ms': args.exposure_ms,
        'effective_readout_std': None,
        'sample_count': len(entries),
    }
    row.update(metrics)
    return row


def create_reference_case(case_dir, entries, base_cfg, args):
    if not should_aggregate_by_experiment(args, entries):
        row = create_reference_case_single(case_dir, entries, base_cfg, args)
        row['aggregation_level'] = args.aggregate_level
        row['experiment_count'] = len(group_entries_by_experiment(entries))
        return row

    if case_dir.exists() and args.force:
        shutil.rmtree(case_dir)
    ensure_dir(case_dir)

    per_experiment_rows = []
    for experiment_index, (experiment_name, experiment_entries) in enumerate(group_entries_by_experiment(entries).items()):
        experiment_case_dir = case_dir / experiment_name
        row = create_reference_case_single(experiment_case_dir, experiment_entries, base_cfg, args)
        row['case_name'] = f'{case_dir.name}_{experiment_name}'
        row['experiment_name'] = experiment_name
        row['experiment_index'] = experiment_index
        per_experiment_rows.append(row)

    write_rows_json(case_dir / 'per_experiment_metrics.json', per_experiment_rows)
    aggregated_row = build_aggregated_row(
        case_name=case_dir.name,
        gain=None,
        gaussian_var=None,
        exposure_ms=args.exposure_ms,
        effective_readout_std=None,
        per_experiment_rows=per_experiment_rows,
        args=args,
    )
    return aggregated_row


def generate_case_split(case_dir, entries, renderer, gain, gaussian_var, args, case_seed):
    generated_dir = case_dir / 'generated'
    left_dir = generated_dir / 'left'
    right_dir = generated_dir / 'right'
    ensure_dir(left_dir)
    ensure_dir(right_dir)

    split_rows = []
    for index, entry in enumerate(entries):
        set_case_seed(case_seed + index)
        left_hdr = load_hdr_rgb(entry['left_abs'])
        right_hdr = load_hdr_rgb(entry['right_abs'])

        left_generated = render_noisy_image(
            renderer=renderer,
            hdr_rgb=left_hdr,
            exposure_ms=args.exposure_ms,
            gain=gain,
            gaussian_var=gaussian_var,
            use_radiance_scale=not args.disable_radiance_scale,
            capacity=args.radiance_capacity,
        )
        right_generated = render_noisy_image(
            renderer=renderer,
            hdr_rgb=right_hdr,
            exposure_ms=args.exposure_ms,
            gain=gain,
            gaussian_var=gaussian_var,
            use_radiance_scale=not args.disable_radiance_scale,
            capacity=args.radiance_capacity,
        )

        sample_name = Path(entry['left_rel']).stem
        left_path = left_dir / f'{sample_name}.npy'
        right_path = right_dir / f'{sample_name}.npy'
        np.save(left_path, left_generated)
        np.save(right_path, right_generated)

        split_rows.append({
            'left': str(left_path.resolve()),
            'right': str(right_path.resolve()),
            'disp': str(entry['disp_abs'].resolve()),
        })
    return split_rows


def run_generated_case_single(case_dir, case_name, entries, base_cfg, renderer, gain, gaussian_var, args, case_seed):
    split_rows = generate_case_split(
        case_dir=case_dir,
        entries=entries,
        renderer=renderer,
        gain=gain,
        gaussian_var=gaussian_var,
        args=args,
        case_seed=case_seed,
    )
    split_path = case_dir / 'Experiment1_eval.txt'
    eval_yaml_path = case_dir / 'eval.yaml'
    write_split_file(split_path, split_rows)

    eval_cfg = build_eval_config(
        base_cfg=base_cfg,
        split_path=split_path,
        batch_size=args.batch_size_per_gpu,
        enable_rgb=args.enable_generated_gtm,
    )
    eval_cfg['DATA_CONFIG']['DATA_INFOS'][0]['GAIN'] = float(gain)
    eval_cfg['DATA_CONFIG']['DATA_INFOS'][0]['READOUT_NOISE_VAR'] = float(gaussian_var)
    eval_cfg['DATA_CONFIG']['DATA_INFOS'][0]['EXPOSURE_MS'] = float(args.exposure_ms)
    eval_cfg['DATA_CONFIG']['DATA_INFOS'][0]['RESCALE'] = False
    write_yaml(eval_yaml_path, eval_cfg)

    metrics = run_eval(case_dir, args, eval_yaml_path)
    row = {
        'sweep_type': 'generated',
        'case_name': case_name,
        'gain': float(gain),
        'gaussian_var': float(gaussian_var),
        'exposure_ms': float(args.exposure_ms),
        'effective_readout_std': compute_effective_readout_std(gaussian_var, args.exposure_ms, gain),
        'sample_count': len(entries),
    }
    row.update(metrics)

    if not args.keep_generated:
        shutil.rmtree(case_dir / 'generated', ignore_errors=True)

    return row


def run_generated_case(case_dir, case_name, entries, base_cfg, renderer, gain, gaussian_var, args, case_seed):
    metrics_path = case_dir / 'metrics.json'
    if metrics_path.exists() and not args.force:
        with open(metrics_path, 'r') as handle:
            return json.load(handle)

    if case_dir.exists() and args.force:
        shutil.rmtree(case_dir)
    ensure_dir(case_dir)

    if not should_aggregate_by_experiment(args, entries):
        row = run_generated_case_single(
            case_dir=case_dir,
            case_name=case_name,
            entries=entries,
            base_cfg=base_cfg,
            renderer=renderer,
            gain=gain,
            gaussian_var=gaussian_var,
            args=args,
            case_seed=case_seed,
        )
        row['aggregation_level'] = args.aggregate_level
        row['experiment_count'] = len(group_entries_by_experiment(entries))
        write_json(metrics_path, row)
        return row

    per_experiment_rows = []
    for experiment_index, (experiment_name, experiment_entries) in enumerate(group_entries_by_experiment(entries).items()):
        experiment_case_dir = case_dir / experiment_name
        ensure_dir(experiment_case_dir)
        row = run_generated_case_single(
            case_dir=experiment_case_dir,
            case_name=f'{case_name}_{experiment_name}',
            entries=experiment_entries,
            base_cfg=base_cfg,
            renderer=renderer,
            gain=gain,
            gaussian_var=gaussian_var,
            args=args,
            case_seed=case_seed + experiment_index * 100000,
        )
        row['experiment_name'] = experiment_name
        row['experiment_index'] = experiment_index
        per_experiment_rows.append(row)

    write_rows_json(case_dir / 'per_experiment_metrics.json', per_experiment_rows)
    aggregated_row = build_aggregated_row(
        case_name=case_name,
        gain=float(gain),
        gaussian_var=float(gaussian_var),
        exposure_ms=float(args.exposure_ms),
        effective_readout_std=compute_effective_readout_std(gaussian_var, args.exposure_ms, gain),
        per_experiment_rows=per_experiment_rows,
        args=args,
    )
    write_json(metrics_path, aggregated_row)
    return aggregated_row


def run_gain_sweep(output_root, entries, base_cfg, renderer, args):
    results = []
    sweep_root = output_root / 'gain_sweep'
    ensure_dir(sweep_root)
    for index, gain in enumerate(args.gain_values):
        case_name = (
            f'gain_{slugify_float(gain)}_gvar_{slugify_float(args.fixed_gaussian_var)}'
        )
        print(
            f'[gain-sweep] case {index + 1}/{len(args.gain_values)}: '
            f'gain={gain}, gaussian_var={args.fixed_gaussian_var}',
            flush=True,
        )
        case_dir = sweep_root / case_name
        row = run_generated_case(
            case_dir=case_dir,
            case_name=case_name,
            entries=entries,
            base_cfg=base_cfg,
            renderer=renderer,
            gain=gain,
            gaussian_var=args.fixed_gaussian_var,
            args=args,
            case_seed=args.seed + index * 1000,
        )
        row['sweep_type'] = 'gain'
        results.append(row)
    results = attach_deltas(results)
    return results


def run_gaussian_sweep(output_root, entries, base_cfg, renderer, args):
    results = []
    sweep_root = output_root / 'gaussian_sweep'
    ensure_dir(sweep_root)
    for index, gaussian_var in enumerate(args.gaussian_values):
        case_name = (
            f'gvar_{slugify_float(gaussian_var)}_gain_{slugify_float(args.fixed_gain)}'
        )
        print(
            f'[gaussian-sweep] case {index + 1}/{len(args.gaussian_values)}: '
            f'gain={args.fixed_gain}, gaussian_var={gaussian_var}',
            flush=True,
        )
        case_dir = sweep_root / case_name
        row = run_generated_case(
            case_dir=case_dir,
            case_name=case_name,
            entries=entries,
            base_cfg=base_cfg,
            renderer=renderer,
            gain=args.fixed_gain,
            gaussian_var=gaussian_var,
            args=args,
            case_seed=args.seed + 100000 + index * 1000,
        )
        row['sweep_type'] = 'gaussian'
        results.append(row)
    results = attach_deltas(results)
    return results


def save_results(output_root, reference_row, gain_rows, gaussian_rows, metadata):
    if reference_row is not None:
        write_rows_json(output_root / 'reference_metrics.json', [reference_row])
        write_rows_csv(output_root / 'reference_metrics.csv', [reference_row])
    if gain_rows:
        write_rows_json(output_root / 'gain_sweep_metrics.json', gain_rows)
        write_rows_csv(output_root / 'gain_sweep_metrics.csv', gain_rows)
    if gaussian_rows:
        write_rows_json(output_root / 'gaussian_sweep_metrics.json', gaussian_rows)
        write_rows_csv(output_root / 'gaussian_sweep_metrics.csv', gaussian_rows)

    summary = {
        'reference': reference_row,
        'gain_sweep': gain_rows,
        'gaussian_sweep': gaussian_rows,
        'metadata': metadata,
    }
    write_json(output_root / 'summary.json', summary)
    write_json(output_root / 'metadata.json', metadata)


def build_metadata(args, entries):
    return {
        'cfg_file': str(Path(args.cfg_file).resolve()),
        'pretrained_model': str(Path(args.pretrained_model).resolve()),
        'split_file': str(Path(args.split_file).resolve()),
        'dataset_root': str(Path(args.dataset_root).resolve()),
        'experiment_names': normalize_experiment_names(args.experiment_name),
        'aggregation_level': args.aggregate_level,
        'sample_count': len(entries),
        'exposure_ms': args.exposure_ms,
        'fixed_gaussian_var': args.fixed_gaussian_var,
        'fixed_gain': args.fixed_gain,
        'gain_unit': 'dB',
        'gain_values': args.gain_values,
        'gaussian_values': args.gaussian_values,
        'poisson_scale': args.poisson_scale,
        'radiance_capacity': args.radiance_capacity,
        'disable_radiance_scale': args.disable_radiance_scale,
        'disable_motion_blur': args.disable_motion_blur,
        'motion_angle_deg': args.motion_angle_deg,
        'motion_velocity_scale': args.motion_velocity_scale,
        'motion_min_length': args.motion_min_length,
        'motion_max_length': args.motion_max_length,
        'motion_canvas_size': args.motion_canvas_size,
        'motion_line_sigma': args.motion_line_sigma,
        'motion_edge_softness': args.motion_edge_softness,
        'enable_generated_gtm': args.enable_generated_gtm,
        'workers': args.workers,
        'batch_size_per_gpu': args.batch_size_per_gpu,
        'cuda_visible_devices': args.cuda_visible_devices,
        'seed': args.seed,
        'nbits': args.nbits,
    }


def print_brief_summary(reference_row, gain_rows, gaussian_rows):
    if reference_row is not None:
        print(
            '[reference] '
            + ', '.join(f'{key}={reference_row[key]:.4f}' for key in METRIC_KEYS),
            flush=True,
        )
    if gain_rows:
        best = min(gain_rows, key=lambda row: row['d1_all'])
        worst = max(gain_rows, key=lambda row: row['d1_all'])
        print(
            '[gain-sweep] '
            f"best gain={best['gain']} d1_all={best['d1_all']:.4f}; "
            f"worst gain={worst['gain']} d1_all={worst['d1_all']:.4f}",
            flush=True,
        )
    if gaussian_rows:
        best = min(gaussian_rows, key=lambda row: row['d1_all'])
        worst = max(gaussian_rows, key=lambda row: row['d1_all'])
        print(
            '[gaussian-sweep] '
            f"best gaussian_var={best['gaussian_var']} d1_all={best['d1_all']:.4f}; "
            f"worst gaussian_var={worst['gaussian_var']} d1_all={worst['d1_all']:.4f}",
            flush=True,
        )


def main():
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    ensure_dir(output_root)

    base_cfg = load_yaml(args.cfg_file)
    entries = load_split_entries(
        split_path=args.split_file,
        dataset_root=args.dataset_root,
        experiment_name=args.experiment_name,
        max_samples=args.max_samples,
    )

    metadata = build_metadata(args, entries)
    renderer = make_renderer(args)

    reference_row = None
    if not args.skip_reference:
        print('[reference] evaluating original selected inputs', flush=True)
        reference_dir = output_root / 'reference_original'
        if reference_dir.exists() and args.force:
            shutil.rmtree(reference_dir)
        ensure_dir(reference_dir)
        reference_metrics_path = reference_dir / 'metrics.json'
        if reference_metrics_path.exists() and not args.force:
            with open(reference_metrics_path, 'r') as handle:
                reference_row = json.load(handle)
        else:
            reference_row = create_reference_case(
                case_dir=reference_dir,
                entries=entries,
                base_cfg=base_cfg,
                args=args,
            )
            write_json(reference_metrics_path, reference_row)

    gain_rows = []
    if not args.skip_gain_sweep:
        gain_rows = run_gain_sweep(output_root, entries, base_cfg, renderer, args)

    gaussian_rows = []
    if not args.skip_gaussian_sweep:
        gaussian_rows = run_gaussian_sweep(output_root, entries, base_cfg, renderer, args)

    save_results(output_root, reference_row, gain_rows, gaussian_rows, metadata)
    print_brief_summary(reference_row, gain_rows, gaussian_rows)


if __name__ == '__main__':
    main()