import argparse
import json
import shutil
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.sweep_gain_readout_raftstereo import (  # noqa: E402
    METRIC_KEYS,
    attach_deltas,
    create_reference_case,
    ensure_dir,
    load_split_entries,
    load_yaml,
    make_renderer,
    normalize_experiment_names,
    run_generated_case,
    slugify_float,
    write_json,
    write_rows_csv,
    write_rows_json,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Sweep exposure time with fixed gain/readout noise and motion blur for RAFTStereo on Carla Experiment1.'
    )
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
        default=str(
            REPO_ROOT / 'output' / 'gain_readout_sweeps' /
            'raftstereo_carla_600x800_experiment1_exposure_motionblur'
        ),
        help='Directory used for generated cases and aggregated metrics.')
    parser.add_argument(
        '--exposure-values',
        type=float,
        nargs='+',
        default=[float(value) for value in range(1, 21)],
        help='Exposure times in milliseconds scanned while gain/readout noise are fixed.')
    parser.add_argument(
        '--fixed-gaussian-var',
        type=float,
        default=1.0,
        help='Readout-noise variance kept constant during the exposure sweep.')
    parser.add_argument(
        '--fixed-gain',
        type=float,
        default=1.0,
        help='Gain kept constant during the exposure sweep.')
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
        help='Disable motion blur. By default this sweep keeps the current motion-blur configuration enabled.')
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


def build_metadata(args, entries, renderer):
    return {
        'cfg_file': str(Path(args.cfg_file).resolve()),
        'pretrained_model': str(Path(args.pretrained_model).resolve()),
        'split_file': str(Path(args.split_file).resolve()),
        'dataset_root': str(Path(args.dataset_root).resolve()),
        'experiment_names': normalize_experiment_names(args.experiment_name),
        'aggregation_level': args.aggregate_level,
        'sample_count': len(entries),
        'exposure_values': args.exposure_values,
        'fixed_gaussian_var': args.fixed_gaussian_var,
        'fixed_gain': args.fixed_gain,
        'poisson_scale': args.poisson_scale,
        'radiance_capacity': args.radiance_capacity,
        'disable_radiance_scale': args.disable_radiance_scale,
        'disable_motion_blur': args.disable_motion_blur,
        'motion_blur_enabled': bool(getattr(renderer, 'motion_blur', False)),
        'motion_angle_deg': float(getattr(renderer, 'motion_angle_deg', 0.0)),
        'motion_velocity_scale': args.motion_velocity_scale,
        'motion_min_length': args.motion_min_length,
        'motion_max_length': args.motion_max_length,
        'motion_canvas_size': args.motion_canvas_size,
        'motion_line_sigma': args.motion_line_sigma,
        'motion_edge_softness': args.motion_edge_softness,
        'motion_blur_params': dict(getattr(renderer, 'motion_blur_params', {})),
        'enable_generated_gtm': args.enable_generated_gtm,
        'workers': args.workers,
        'batch_size_per_gpu': args.batch_size_per_gpu,
        'cuda_visible_devices': args.cuda_visible_devices,
        'seed': args.seed,
        'nbits': args.nbits,
    }


def run_exposure_sweep(output_root, entries, base_cfg, renderer, args):
    results = []
    sweep_root = output_root / 'exposure_sweep'
    ensure_dir(sweep_root)
    total_cases = len(args.exposure_values)

    for index, exposure_ms in enumerate(args.exposure_values):
        case_name = (
            f'exposure_{slugify_float(exposure_ms)}ms_'
            f'gain_{slugify_float(args.fixed_gain)}_'
            f'gvar_{slugify_float(args.fixed_gaussian_var)}'
        )
        print(
            f'[exposure-sweep] case {index + 1}/{total_cases}: '
            f'exposure_ms={exposure_ms}, gain={args.fixed_gain}, '
            f'gaussian_var={args.fixed_gaussian_var}',
            flush=True,
        )
        case_dir = sweep_root / case_name
        args.exposure_ms = float(exposure_ms)
        row = run_generated_case(
            case_dir=case_dir,
            case_name=case_name,
            entries=entries,
            base_cfg=base_cfg,
            renderer=renderer,
            gain=args.fixed_gain,
            gaussian_var=args.fixed_gaussian_var,
            args=args,
            case_seed=args.seed + index * 1000,
        )
        row['sweep_type'] = 'exposure'
        results.append(row)
    return attach_deltas(results)


def save_results(output_root, reference_row, exposure_rows, metadata):
    if reference_row is not None:
        write_rows_json(output_root / 'reference_metrics.json', [reference_row])
        write_rows_csv(output_root / 'reference_metrics.csv', [reference_row])
    if exposure_rows:
        write_rows_json(output_root / 'exposure_sweep_metrics.json', exposure_rows)
        write_rows_csv(output_root / 'exposure_sweep_metrics.csv', exposure_rows)

    summary = {
        'reference': reference_row,
        'exposure_sweep': exposure_rows,
        'metadata': metadata,
    }
    write_json(output_root / 'summary.json', summary)
    write_json(output_root / 'metadata.json', metadata)


def print_brief_summary(reference_row, exposure_rows):
    if reference_row is not None:
        print(
            '[reference] '
            + ', '.join(f'{key}={reference_row[key]:.4f}' for key in METRIC_KEYS),
            flush=True,
        )
    if exposure_rows:
        best = min(exposure_rows, key=lambda row: row['d1_all'])
        worst = max(exposure_rows, key=lambda row: row['d1_all'])
        print(
            '[exposure-sweep] '
            f"best exposure_ms={best['exposure_ms']} d1_all={best['d1_all']:.4f}; "
            f"worst exposure_ms={worst['exposure_ms']} d1_all={worst['d1_all']:.4f}",
            flush=True,
        )


def main():
    args = parse_args()
    args.exposure_ms = float(args.exposure_values[0])

    output_root = Path(args.output_root).resolve()
    ensure_dir(output_root)

    base_cfg = load_yaml(args.cfg_file)
    entries = load_split_entries(
        split_path=args.split_file,
        dataset_root=args.dataset_root,
        experiment_name=args.experiment_name,
        max_samples=args.max_samples,
    )
    renderer = make_renderer(args)
    metadata = build_metadata(args, entries, renderer)

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
            reference_row['exposure_ms'] = None
            write_json(reference_metrics_path, reference_row)
        if reference_row is not None:
            reference_row['exposure_ms'] = None

    exposure_rows = run_exposure_sweep(output_root, entries, base_cfg, renderer, args)
    save_results(output_root, reference_row, exposure_rows, metadata)
    print_brief_summary(reference_row, exposure_rows)


if __name__ == '__main__':
    main()