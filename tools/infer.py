import sys
import os
import argparse
import numpy as np
import cv2
import torch
import torch.distributed as dist
from easydict import EasyDict
from PIL import Image
from pathlib import Path

sys.path.insert(0, './')
from stereo.utils import common_utils
from stereo.modeling import build_trainer
from stereo.utils.disp_color import disp_to_color
from stereo.datasets.dataset_template import build_transform_by_cfg
from stereo.datasets import build_dataloader
from cfgs.data_basic import DATA_PATH_DICT


def _get_logger_iter_interval(cfgs, default_val=10):
    trainer_cfg = cfgs.get('TRAINER', None)
    if trainer_cfg is None:
        return default_val
    return trainer_cfg.get('LOGGER_ITER_INTERVAL', default_val)


def _get_max_disp_for_vis(cfgs, default_val=192):
    evaluator_cfg = cfgs.get('EVALUATOR', None)
    if evaluator_cfg is not None and evaluator_cfg.get('MAX_DISP', None) is not None:
        return evaluator_cfg.MAX_DISP

    data_infos = cfgs.get('DATA_CONFIG', {}).get('DATA_INFOS', [])
    if len(data_infos) > 0 and data_infos[0].get('MAX_DISP', None) is not None:
        return data_infos[0].MAX_DISP

    return default_val


def _get_sample_group_name(sample_path):
    parent = sample_path.parent
    if parent.name in {'hdr_left', 'hdr_right', 'ldr_left', 'ldr_right', 'left', 'right', 'image_2', 'image_3'}:
        if parent.parent.name:
            return parent.parent.name
    return parent.name if parent.name else sample_path.stem


def _resolve_batch_output_path(sample_name, dataset_roots, output_dir, fallback_idx):
    if sample_name is None:
        rel_path = Path(f'sample_{fallback_idx:06d}.png')
    else:
        sample_path = Path(str(sample_name))
        group_name = _get_sample_group_name(sample_path)
        stem = sample_path.stem
        if stem.isdigit():
            file_name = f'disparity_map_{stem}.png'
        elif stem.startswith('disparity_map_'):
            file_name = f'{stem}.png'
        else:
            file_name = sample_path.with_suffix('.png').name
        if not file_name:
            file_name = f'sample_{fallback_idx:06d}.png'
        rel_path = Path(group_name) / file_name

    rel_path = Path(str(rel_path).lstrip('/\\'))
    if '..' in rel_path.parts:
        rel_path = Path(f'sample_{fallback_idx:06d}.png')

    save_path = Path(output_dir) / rel_path
    save_path.parent.mkdir(parents=True, exist_ok=True)
    return str(save_path)


def _render_disparity_magma(disp):
    disp = np.asarray(disp, dtype=np.float32)
    finite = np.isfinite(disp)
    if not finite.any():
        gray = np.zeros(disp.shape, dtype=np.uint8)
        return cv2.applyColorMap(gray, cv2.COLORMAP_MAGMA)

    valid_disp = disp[finite]
    min_val = float(valid_disp.min())
    max_val = float(valid_disp.max())
    if max_val <= min_val:
        gray = np.zeros(disp.shape, dtype=np.uint8)
    else:
        gray_float = (np.clip(disp, min_val, max_val) - min_val) / (max_val - min_val)
        gray_float[~finite] = 0.0
        gray = (gray_float * 255.0).round().astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_MAGMA)


def _read_sample_hw(sample_name, dataset_roots=None):
    if sample_name is None:
        return None

    sample_path = Path(str(sample_name))
    candidate_paths = [sample_path]
    if not sample_path.is_absolute():
        candidate_paths.extend(Path(root) / sample_path for root in (dataset_roots or []))

    for candidate_path in candidate_paths:
        if not candidate_path.exists():
            continue

        ext = candidate_path.suffix.lower()
        if ext == '.npy':
            image = np.load(candidate_path)
        else:
            image = cv2.imread(str(candidate_path), cv2.IMREAD_UNCHANGED)
        if image is not None:
            return image.shape[:2]

    return None


def _crop_to_sample_size(disp, sample_name, dataset_roots=None):
    sample_hw = _read_sample_hw(sample_name, dataset_roots)
    if sample_hw is None:
        return disp

    h, w = sample_hw
    if disp.shape[0] < h or disp.shape[1] < w:
        return disp
    return disp[-h:, :w]


def _move_sample_to_device(sample, local_rank):
    for k, v in sample.items():
        sample[k] = v.to(local_rank) if torch.is_tensor(v) else v
    return sample


@torch.no_grad()
def run_single_infer(args, cfgs, model, local_rank):
    if args.left_img_path is None or args.right_img_path is None:
        raise ValueError('Single-image infer mode needs --left_img_path and --right_img_path.')
    if args.savename is None:
        raise ValueError('Single-image infer mode needs --savename.')

    transform_config = cfgs.DATA_CONFIG.DATA_TRANSFORM.EVALUATING
    transform = build_transform_by_cfg(transform_config)
    # left_img = np.array(Image.open(args.left_img_path).convert('RGB'), dtype=np.float32)
    # right_img = np.array(Image.open(args.right_img_path).convert('RGB'), dtype=np.float32)
    left_img = np.array(cv2.imread(args.left_img_path, cv2.IMREAD_UNCHANGED), dtype=np.float32)
    right_img = np.array(cv2.imread(args.right_img_path, cv2.IMREAD_UNCHANGED), dtype=np.float32)
    sample = {
        'left': left_img,
        'right': right_img,
    }



    sample = transform(sample)
    sample['left'] = sample['left'].unsqueeze(0)
    sample['right'] = sample['right'].unsqueeze(0)

    sample = _move_sample_to_device(sample, local_rank)
    with torch.cuda.amp.autocast(enabled=cfgs.OPTIMIZATION.AMP):
        model_pred = model(sample)

    disp_pred = model_pred['disp_pred'].squeeze().detach().cpu().numpy()
    max_disp = _get_max_disp_for_vis(cfgs, default_val=192)
    # img_color = disp_to_color(disp_pred, max_disp=max_disp).astype('uint8')
    img_color = _render_disparity_magma(disp_pred)

    save_dir = os.path.dirname(args.savename)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    # Image.fromarray(img_color).save(args.savename)
    cv2.imwrite(args.savename, img_color)


@torch.no_grad()
def run_batch_infer(args, cfgs, model, local_rank, logger):
    if cfgs.get('DATA_CONFIG', None) is None:
        raise ValueError('Batch infer mode needs DATA_CONFIG, please pass --eval_data_cfg_file.')

    batch_size = 1
    if cfgs.get('EVALUATOR', None) is not None:
        batch_size = cfgs.EVALUATOR.get('BATCH_SIZE_PER_GPU', 1)

    _, eval_loader, _ = build_dataloader(
        data_cfg=cfgs.DATA_CONFIG,
        batch_size=batch_size,
        is_dist=args.dist_mode,
        workers=args.workers,
        pin_memory=args.pin_memory,
        mode='evaluating')

    output_dir = args.save_root_dir
    os.makedirs(output_dir, exist_ok=True)

    dataset_roots = [str(each.DATA_PATH) for each in cfgs.DATA_CONFIG.DATA_INFOS]
    logger_iter_interval = _get_logger_iter_interval(cfgs)
    max_disp = _get_max_disp_for_vis(cfgs, default_val=192)

    total_saved = 0
    for i, data in enumerate(eval_loader):
        names = data.get('name', None)
        data = _move_sample_to_device(data, local_rank)

        with torch.cuda.amp.autocast(enabled=cfgs.OPTIMIZATION.AMP):
            model_pred = model(data)

        disp_batch = model_pred['disp_pred'].squeeze(1).detach().cpu().numpy()
        batch_n = disp_batch.shape[0]

        if isinstance(names, (tuple, list)):
            name_list = [str(x) for x in names]
        else:
            name_list = [None] * batch_n

        for bi in range(batch_n):
            sample_name = name_list[bi] if bi < len(name_list) else None
            save_path = _resolve_batch_output_path(sample_name, dataset_roots, output_dir, total_saved + bi)
            # img_color = disp_to_color(disp_batch[bi], max_disp=max_disp).astype('uint8')
            disp_vis = _crop_to_sample_size(disp_batch[bi], sample_name, dataset_roots)
            img_color = _render_disparity_magma(disp_vis)
            cv2.imwrite(save_path, img_color)


        total_saved += batch_n
        if i % logger_iter_interval == 0:
            logger.info('Batch infer iter:{:>4d}/{} saved:{} -> {}'.format(i, len(eval_loader), total_saved, output_dir))

    logger.info('Batch infer finished, total saved: {}. Output dir: {}'.format(total_saved, output_dir))


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dist_mode', action='store_true', default=False, help='torchrun ddp multi gpu')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for eval')
    parser.add_argument('--eval_data_cfg_file', type=str, default=None, help='specify the data config for batch infer')
    # data
    parser.add_argument('--left_img_path', type=str, default=None)
    parser.add_argument('--right_img_path', type=str, default=None)
    parser.add_argument('--pretrained_model', type=str, default=None, help='pretrained_model')
    parser.add_argument('--savename', type=str, default=None)
    parser.add_argument('--workers', type=int, default=0, help='number of workers for dataloader')
    parser.add_argument('--pin_memory', action='store_true', default=False, help='data loader pin memory')
    parser.add_argument('--save_root_dir', type=str, default='./output', help='save root dir for batch infer')

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

        for each in cfgs.DATA_CONFIG.DATA_INFOS:
            dataset_name = each.DATASET
            if dataset_name == 'KittiDataset':
                eval_split = each.DATA_SPLIT.EVALUATING.lower()
                dataset_name = 'KittiDataset15' if 'kitti15' in eval_split else 'KittiDataset12'
            if dataset_name not in DATA_PATH_DICT:
                raise KeyError('Dataset {} is not in DATA_PATH_DICT, please add it in cfgs/data_basic.py'.format(dataset_name))
            each.DATA_PATH = DATA_PATH_DICT[dataset_name]
            assert os.path.exists(each.DATA_PATH), (
                '[Errno 2] No such file or directory: {}, '
                'You must modify the data root path in cfgs/data_basic.py to your own dataset path.'
            ).format(each.DATA_PATH)

        args.exp_group_path = os.path.join(cfgs.DATA_CONFIG.DATA_INFOS[0].DATASET, cfgs.MODEL.NAME)
    else:
        args.exp_group_path = os.path.join('single', cfgs.MODEL.NAME)
    
    args.run_mode = 'infer'
    return args, cfgs



@torch.no_grad()
def main():
    args, cfgs = parse_config()
    if args.dist_mode:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ["LOCAL_RANK"])
        global_rank = int(os.environ["RANK"])
    else:
        local_rank = 0
        global_rank = 0

    # env
    torch.cuda.set_device(local_rank)
    seed = 0 if not args.dist_mode else dist.get_rank()
    common_utils.set_random_seed(seed=seed)

    # log
    logger = common_utils.create_logger(log_file=None, rank=local_rank)

    # log args and cfgs
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    common_utils.log_configs(cfgs, logger=logger)

    # model
    trainer = build_trainer(args, cfgs, local_rank, global_rank, logger, None)
    model = trainer.model

    model.eval()
    if args.eval_data_cfg_file:
        run_batch_infer(args, cfgs, model, local_rank, logger)
    else:
        run_single_infer(args, cfgs, model, local_rank)

    if args.dist_mode:
        dist.barrier()


if __name__ == '__main__':
    main()
