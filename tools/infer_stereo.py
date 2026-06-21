import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.distributed as dist
from easydict import EasyDict

sys.path.insert(0, './')
from cfgs.data_basic import DATA_PATH_DICT
from stereo.datasets import build_dataloader
from stereo.modeling import build_trainer
from stereo.utils import common_utils


DEFAULT_ADEC_ROOT = '/home/lgz/dataset/ADEC'
DEFAULT_METHOD_NAME = 'adpreward'
CARLA_SUBSETS = ('carla_600x800', 'carla_1280x384')


def _cfg_get(cfg, key, default=None):
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _get_logger_iter_interval(cfgs, default_val=10):
    trainer_cfg = cfgs.get('TRAINER', None)
    if trainer_cfg is None:
        return default_val
    return trainer_cfg.get('LOGGER_ITER_INTERVAL', default_val)


def _get_eval_batch_size(cfgs, default_val=1):
    evaluator_cfg = cfgs.get('EVALUATOR', None)
    if evaluator_cfg is None:
        return default_val
    return int(evaluator_cfg.get('BATCH_SIZE_PER_GPU', default_val))


def _populate_data_paths(data_cfg):
    for data_info in data_cfg.DATA_INFOS:
        dataset_name = data_info.DATASET
        if dataset_name == 'KittiDataset':
            eval_split = data_info.DATA_SPLIT.EVALUATING.lower()
            dataset_name = 'KittiDataset15' if 'kitti15' in eval_split else 'KittiDataset12'
        if dataset_name not in DATA_PATH_DICT:
            raise KeyError(
                'Dataset {} is not in DATA_PATH_DICT, please add it in cfgs/data_basic.py'.format(
                    dataset_name
                )
            )
        data_info.DATA_PATH = DATA_PATH_DICT[dataset_name]
        if not os.path.exists(data_info.DATA_PATH):
            raise FileNotFoundError(
                '[Errno 2] No such file or directory: {}, '
                'You must modify the data root path in cfgs/data_basic.py to your own dataset path.'.format(
                    data_info.DATA_PATH
                )
            )


def _promote_single_carla_to_dual(data_cfg, logger=None):
    for data_info in data_cfg.DATA_INFOS:
        if data_info.DATASET == 'CarlaStereoDataset':
            data_info.DATASET = 'CarlaStereoDualDataset'
            if not hasattr(data_info, 'FRAME_STEP'):
                data_info.FRAME_STEP = 1
            if logger is not None:
                logger.warning(
                    'Promote CarlaStereoDataset to CarlaStereoDualDataset for exposure rollout. '
                    'The split must contain consecutive frames.'
                )


def _validate_dual_data_config(data_cfg):
    bad_datasets = [
        data_info.DATASET
        for data_info in data_cfg.DATA_INFOS
        if data_info.DATASET != 'CarlaStereoDualDataset'
    ]
    if bad_datasets:
        raise ValueError(
            'infer_stereo.py needs dual-frame samples. Unsupported DATASET values: {}'.format(
                ', '.join(bad_datasets)
            )
        )


def _get_split_file(data_info, mode='EVALUATING'):
    data_split = data_info.DATA_SPLIT
    if mode in data_split:
        return str(data_split[mode])
    if mode.lower() in data_split:
        return str(data_split[mode.lower()])
    return ''


def _read_first_split_token(split_file):
    if not split_file or not os.path.exists(split_file):
        return ''
    with open(split_file, 'r', encoding='utf-8') as fp:
        for line in fp:
            line = line.strip()
            if line and not line.startswith('#'):
                return line.split()[0]
    return ''


def _infer_carla_subset(cfgs):
    candidates = []
    for data_info in cfgs.DATA_CONFIG.DATA_INFOS:
        split_file = _get_split_file(data_info)
        candidates.append(split_file)
        candidates.append(_read_first_split_token(split_file))
    joined = ' '.join(candidates)
    for subset in CARLA_SUBSETS:
        if subset in joined:
            return subset
    return None


def _resolve_save_root(args, cfgs):
    if args.save_root_dir:
        return Path(args.save_root_dir)

    subset = _infer_carla_subset(cfgs)
    if subset is None:
        raise ValueError(
            'Cannot infer Carla subset from DATA_SPLIT. Please pass --save_root_dir explicitly.'
        )
    return Path(args.data_root) / subset / 'ae_methods' / args.method_name


def _path_to_parts(path_like):
    return [part for part in Path(str(path_like)).parts if part not in ('', os.sep)]


def _find_experiment_name(path_like):
    parts = _path_to_parts(path_like)
    for part in parts:
        if part.lower().startswith('experiment'):
            return part
    path_obj = Path(str(path_like))
    if path_obj.parent.parent.name:
        return path_obj.parent.parent.name
    if path_obj.parent.name:
        return path_obj.parent.name
    return 'default'


def _resolve_output_pair_paths(left_name, right_name, save_root):
    left_path = Path(str(left_name))
    right_path = Path(str(right_name))
    experiment = _find_experiment_name(left_path)
    left_file = left_path.with_suffix('.png').name or 'left.png'
    right_file = right_path.with_suffix('.png').name or left_file
    left_save = save_root / experiment / 'hdr_left' / left_file
    right_save = save_root / experiment / 'hdr_right' / right_file
    return left_save, right_save


def _safe_relative_path(path, root):
    path = Path(path)
    root = Path(root)
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_abs_path(sample_name, dataset_roots):
    sample_path = Path(str(sample_name))
    candidates = [sample_path]
    if not sample_path.is_absolute():
        candidates.extend(Path(root) / sample_path for root in dataset_roots)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return sample_path


def _read_sample_hw(sample_name, dataset_roots):
    sample_path = _resolve_abs_path(sample_name, dataset_roots)
    if not sample_path.exists():
        return None

    ext = sample_path.suffix.lower()
    if ext == '.npy':
        image = np.load(sample_path)
    else:
        image = cv2.imread(str(sample_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return None
    return image.shape[:2]


def _crop_to_sample_size(image, sample_name, dataset_roots):
    sample_hw = _read_sample_hw(sample_name, dataset_roots)
    if sample_hw is None:
        return image

    h, w = sample_hw
    if image.shape[0] < h or image.shape[1] < w:
        return image
    return image[-h:, :w]


def _tensor_image_to_uint8(image_tensor, sample_name, dataset_roots):
    image = image_tensor.detach().float().cpu().numpy()
    if image.ndim != 3:
        raise ValueError('Expected image tensor with shape [C, H, W], got {}'.format(image.shape))
    image = np.transpose(image, (1, 2, 0))
    image = _crop_to_sample_size(image, sample_name, dataset_roots)
    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    if image.shape[2] > 3:
        image = image[:, :, :3]
    return np.clip(image * 255.0, 0.0, 255.0).round().astype(np.uint8)


def _save_rgb_png(path, image_rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(path), image_bgr):
        raise IOError('Failed to write image: {}'.format(path))


def _move_sample_to_device(sample, device):
    for key, value in sample.items():
        sample[key] = value.to(device) if torch.is_tensor(value) else value
    return sample


def _to_name_list(names, batch_n):
    if isinstance(names, (tuple, list)):
        return [str(name) for name in names]
    if names is None:
        return [None] * batch_n
    return [str(names)] * batch_n


def _replace_side_token(left_name):
    side_pairs = (
        ('hdr_left', 'hdr_right'),
        ('ldr_left', 'ldr_right'),
        ('left', 'right'),
        ('image_2', 'image_3'),
    )
    right_name = str(left_name)
    for left_token, right_token in side_pairs:
        if left_token in right_name:
            return right_name.replace(left_token, right_token, 1)
    return right_name


def _build_disp_lookup(eval_dataset):
    lookup = {}
    datasets = getattr(eval_dataset, 'datasets', [eval_dataset])
    for dataset in datasets:
        root = Path(str(getattr(dataset, 'root', '')))
        data_list = getattr(dataset, 'data_list', [])
        get_item_paths = getattr(dataset, '_get_item_paths', None)
        for item in data_list:
            if get_item_paths is not None:
                left_path, right_path, _, _, disp_path, _ = get_item_paths(item)
            elif len(item) >= 3:
                left_path, right_path, disp_path = item[:3]
            else:
                continue

            lookup[str(left_path)] = (str(right_path), str(disp_path))
            if root:
                lookup[str(root / left_path)] = (str(right_path), str(disp_path))
    return lookup


def _resolve_pair_and_disp(left_name, disp_lookup):
    if left_name in disp_lookup:
        return disp_lookup[left_name]
    left_name_str = str(left_name)
    return _replace_side_token(left_name_str), _derive_disp_path(left_name_str)


def _derive_disp_path(left_name):
    path = Path(str(left_name))
    frame_id = path.stem
    parts = list(path.parts)
    for idx, part in enumerate(parts):
        if part in {'hdr_left', 'ldr_left', 'left', 'image_2'}:
            parts[idx] = 'ground_truth_disparity_left'
            return Path(*parts[:idx], parts[idx], 'disparity_map_{}.npy'.format(frame_id)).as_posix()
    return path.with_name('disparity_map_{}.npy'.format(frame_id)).as_posix()


def _get_model_ref(model, dist_mode):
    return model.module if dist_mode else model


def _extract_final_state(model_ref, rollout, batch):
    transitions = rollout.get('transitions', [])
    if transitions:
        return transitions[-1]['next_state'].detach()
    if hasattr(model_ref, 'env') and hasattr(model_ref.env, 'get_initial_state'):
        return model_ref.env.get_initial_state(batch['left_1']).detach()
    raise AttributeError('Model rollout did not provide transitions and model.env is unavailable.')


def _render_final_images(model_ref, batch, rollout, final_state):
    final_images = rollout.get('final_images', None)
    if isinstance(final_images, (tuple, list)) and len(final_images) >= 2:
        return final_images[0].detach(), final_images[1].detach()

    if not hasattr(model_ref, 'env') or not hasattr(model_ref.env, 'render_dual_frames'):
        raise AttributeError('Model must provide rollout final_images or env.render_dual_frames().')
    left_1, right_1, _, _ = model_ref.env.render_dual_frames(
        batch['left_1'],
        batch['right_1'],
        batch['left_2'],
        batch['right_2'],
        final_state,
    )
    return left_1.detach(), right_1.detach()


def _format_float(value):
    if value is None or not np.isfinite(value):
        return 'nan'
    return '{:.8g}'.format(float(value))


def _state_to_exposure_values(state_values):
    values = [float(value) for value in state_values]
    if len(values) >= 6:
        exp_t1, gain1 = values[0], values[1]
        exp_t2_left, gain2_left = values[2], values[3]
        exp_t2_right, gain2_right = values[4], values[5]
    elif len(values) >= 4:
        exp_t1, gain1 = values[0], values[1]
        exp_t2_left, gain2_left = values[2], values[3]
        exp_t2_right, gain2_right = values[2], values[3]
    elif len(values) >= 2:
        exp_t1, gain1 = values[0], values[1]
        exp_t2_left, gain2_left = values[0], values[1]
        exp_t2_right, gain2_right = values[0], values[1]
    else:
        exp_t1 = gain1 = exp_t2_left = gain2_left = exp_t2_right = gain2_right = float('nan')
    return exp_t1, gain1, exp_t2_left, gain2_left, exp_t2_right, gain2_right


def _build_exposure_row(sample_index, left_rel, right_rel, disp_rel, state_values):
    exposure_values = _state_to_exposure_values(state_values)
    state_text = ','.join(_format_float(value) for value in state_values)
    row_values = [
        str(sample_index),
        left_rel,
        right_rel,
        disp_rel,
        *[_format_float(value) for value in exposure_values],
        str(len(state_values)),
        state_text,
    ]
    return ' '.join(row_values)


def _write_text_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    text = '\n'.join(lines)
    if text:
        text += '\n'
    path.write_text(text, encoding='utf-8')


def _validate_batch(data):
    required_keys = ('left_1', 'right_1', 'left_2', 'right_2', 'disp')
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        raise KeyError(
            'Batch is missing dual-frame keys: {}. Use CarlaStereoDualDataset data.'.format(
                ', '.join(missing_keys)
            )
        )


@torch.no_grad()
def run_batch_render(args, cfgs, model, device, logger):
    if cfgs.get('DATA_CONFIG', None) is None:
        raise ValueError('infer_stereo.py needs DATA_CONFIG.')

    batch_size = int(args.batch_size) if args.batch_size is not None else _get_eval_batch_size(cfgs)
    eval_dataset, eval_loader, _ = build_dataloader(
        data_cfg=cfgs.DATA_CONFIG,
        batch_size=batch_size,
        is_dist=args.dist_mode,
        workers=args.workers,
        pin_memory=args.pin_memory,
        mode='evaluating',
    )
    disp_lookup = _build_disp_lookup(eval_dataset)
    dataset_roots = [str(each.DATA_PATH) for each in cfgs.DATA_CONFIG.DATA_INFOS]
    save_root = _resolve_save_root(args, cfgs)
    save_root.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root)
    model_ref = _get_model_ref(model, args.dist_mode)
    if not hasattr(model_ref, 'rollout_episode'):
        raise AttributeError('Model {} does not implement rollout_episode().'.format(type(model_ref).__name__))

    rollout_steps = int(args.rollout_steps) if args.rollout_steps is not None else int(
        _cfg_get(cfgs.MODEL, 'ROLLOUT_STEPS', 1)
    )
    rollout_steps = max(rollout_steps, 1)
    amp_enabled = bool(cfgs.OPTIMIZATION.get('AMP', False)) and device.type == 'cuda'
    logger_iter_interval = _get_logger_iter_interval(cfgs)

    val_lines = []
    exposure_lines = [
        '# sample_index left_path right_path disp_path exp_t1 gain1 '
        'exp_t2_left gain2_left exp_t2_right gain2_right state_dim state_values'
    ]
    total_saved = 0
    max_samples = int(args.max_samples) if args.max_samples is not None else -1
    if max_samples == 0:
        _write_text_lines(save_root / 'val.txt', val_lines)
        _write_text_lines(save_root / 'exposure_params.txt', exposure_lines)
        logger.info('Stereo render skipped because --max_samples 0 was set.')
        logger.info('Saved split file: {}'.format(save_root / 'val.txt'))
        logger.info('Saved exposure file: {}'.format(save_root / 'exposure_params.txt'))
        return

    for iter_idx, data in enumerate(eval_loader):
        _validate_batch(data)
        names = _to_name_list(data.get('name', None), batch_n=len(data['left_1']))
        data = _move_sample_to_device(data, device)

        with torch.cuda.amp.autocast(enabled=amp_enabled):
            rollout = model_ref.rollout_episode(
                batch=data,
                steps=rollout_steps,
                deterministic=not args.stochastic,
            )
            final_state = _extract_final_state(model_ref, rollout, data)
            final_left, final_right = _render_final_images(model_ref, data, rollout, final_state)

        batch_n = final_left.shape[0]
        final_state_np = final_state.detach().cpu().numpy()

        for batch_idx in range(batch_n):
            if max_samples >= 0 and total_saved >= max_samples:
                break

            left_name = names[batch_idx] if batch_idx < len(names) else 'sample_{:06d}.png'.format(total_saved)
            right_name, disp_path = _resolve_pair_and_disp(left_name, disp_lookup)
            left_save, right_save = _resolve_output_pair_paths(left_name, right_name, save_root)

            left_rgb = _tensor_image_to_uint8(final_left[batch_idx], left_name, dataset_roots)
            right_rgb = _tensor_image_to_uint8(final_right[batch_idx], right_name, dataset_roots)
            _save_rgb_png(left_save, left_rgb)
            _save_rgb_png(right_save, right_rgb)

            left_rel = _safe_relative_path(left_save, data_root)
            right_rel = _safe_relative_path(right_save, data_root)
            disp_rel = _safe_relative_path(_resolve_abs_path(disp_path, dataset_roots), data_root)
            val_lines.append('{} {} {}'.format(left_rel, right_rel, disp_rel))

            state_values = final_state_np[batch_idx].reshape(-1).tolist()
            exposure_lines.append(
                _build_exposure_row(total_saved, left_rel, right_rel, disp_rel, state_values)
            )
            total_saved += 1

        if iter_idx % logger_iter_interval == 0:
            logger.info(
                'Stereo render iter:{:>4d}/{} saved:{} -> {}'.format(
                    iter_idx, len(eval_loader), total_saved, save_root
                )
            )
        if max_samples >= 0 and total_saved >= max_samples:
            break

    val_path = save_root / 'val.txt'
    exposure_path = save_root / 'exposure_params.txt'
    _write_text_lines(val_path, val_lines)
    _write_text_lines(exposure_path, exposure_lines)
    logger.info('Stereo render finished, total saved: {}'.format(total_saved))
    logger.info('Saved split file: {}'.format(val_path))
    logger.info('Saved exposure file: {}'.format(exposure_path))


def parse_config():
    parser = argparse.ArgumentParser(description='Render exposure stereo images with OpenStereo models.')
    parser.add_argument('--dist_mode', action='store_true', default=False, help='torchrun ddp multi gpu')
    parser.add_argument('--cfg_file', type=str, required=True, help='model/config yaml')
    parser.add_argument('--eval_data_cfg_file', type=str, default=None, help='optional data config yaml')
    parser.add_argument('--pretrained_model', type=str, default=None, help='checkpoint path')
    parser.add_argument('--workers', type=int, default=0, help='number of dataloader workers')
    parser.add_argument('--pin_memory', action='store_true', default=False, help='data loader pin memory')
    parser.add_argument('--save_root_dir', type=str, default=None, help='output method directory')
    parser.add_argument('--data_root', type=str, default=DEFAULT_ADEC_ROOT, help='ADEC root for split paths')
    parser.add_argument('--method_name', type=str, default=DEFAULT_METHOD_NAME, help='AE method directory name')
    parser.add_argument('--batch_size', type=int, default=None, help='override evaluator batch size')
    parser.add_argument('--max_samples', type=int, default=-1, help='limit rendered samples for smoke tests')
    parser.add_argument('--rollout_steps', type=int, default=None, help='override model rollout steps')
    parser.add_argument('--stochastic', action='store_true', default=False, help='sample stochastic actor actions')

    args = parser.parse_args()
    yaml_config = common_utils.config_loader(args.cfg_file)
    cfgs = EasyDict(yaml_config)

    if args.pretrained_model is not None:
        cfgs.MODEL.PRETRAINED_MODEL = args.pretrained_model

    if args.eval_data_cfg_file:
        eval_data_yaml_config = common_utils.config_loader(args.eval_data_cfg_file)
        eval_data_cfgs = EasyDict(eval_data_yaml_config)
        cfgs.DATA_CONFIG = eval_data_cfgs.DATA_CONFIG
        if eval_data_cfgs.get('EVALUATOR', None) is not None:
            cfgs.EVALUATOR = eval_data_cfgs.EVALUATOR

    _promote_single_carla_to_dual(cfgs.DATA_CONFIG)
    _populate_data_paths(cfgs.DATA_CONFIG)
    _validate_dual_data_config(cfgs.DATA_CONFIG)

    args.exp_group_path = os.path.join(cfgs.DATA_CONFIG.DATA_INFOS[0].DATASET, cfgs.MODEL.NAME)
    args.run_mode = 'infer'
    return args, cfgs


@torch.no_grad()
def main():
    args, cfgs = parse_config()
    if args.dist_mode:
        raise NotImplementedError('Distributed metadata writing is not supported by infer_stereo.py yet.')

    local_rank = 0
    global_rank = 0
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')

    common_utils.set_random_seed(seed=0)
    logger = common_utils.create_logger(log_file=None, rank=local_rank)

    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    common_utils.log_configs(cfgs, logger=logger)

    trainer = build_trainer(args, cfgs, local_rank, global_rank, logger, None)
    model = trainer.model
    model.eval()

    run_batch_render(args, cfgs, model, device, logger)

    if args.dist_mode:
        dist.barrier()


if __name__ == '__main__':
    main()
