import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    from stereo.datasets.carla_stereo_dataset import (
        _load_disparity,
        _load_stereo_image,
        radiance_scale,
    )
    from stereo.datasets.dataset_template import DatasetTemplate
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from stereo.datasets.carla_stereo_dataset import (
        _load_disparity,
        _load_stereo_image,
        radiance_scale,
    )
    from stereo.datasets.dataset_template import DatasetTemplate


def _replace_last_number(path, step=1):
    path_obj = Path(path)
    stem = path_obj.stem
    matches = list(re.finditer(r'\d+', stem))
    if not matches:
        raise ValueError(f'Cannot infer next frame path from: {path}')

    match = matches[-1]
    number = match.group(0)
    next_number = str(int(number) + step).zfill(len(number))
    next_stem = stem[:match.start()] + next_number + stem[match.end():]
    return str(path_obj.with_name(next_stem + path_obj.suffix))


def _safe_max_normalize(image):
    image = image.astype(np.float32)
    max_val = float(np.max(image))
    if max_val <= 1e-6:
        return np.zeros_like(image, dtype=np.float32)
    return image / max_val


def _validate_exposure_range(exposure_range):
    if exposure_range is None:
        exposure_range = [0.5, 1.5]
    try:
        min_exposure, max_exposure = [float(x) for x in exposure_range]
    except (TypeError, ValueError):
        raise ValueError(
            'NEXT_EXPOSURE_RANGE must contain two positive values: '
            f'{exposure_range}'
        )

    if min_exposure <= 0 or max_exposure <= 0 or min_exposure > max_exposure:
        raise ValueError(
            'NEXT_EXPOSURE_RANGE must satisfy 0 < min <= max, got: '
            f'{exposure_range}'
        )
    return min_exposure, max_exposure


def _apply_exposure_scale(left_img, right_img, exposure_range):
    exposure_scale = np.random.uniform(exposure_range[0], exposure_range[1])
    left_img = np.clip(left_img * exposure_scale, 0.0, 1.0).astype(np.float32)
    right_img = np.clip(right_img * exposure_scale, 0.0, 1.0).astype(np.float32)
    return left_img, right_img


def apply_gtm(img, eps=1e-6, param=0.1):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    Lw_ave = np.exp(np.mean(np.log(eps + img)))
    Lm = (param / Lw_ave) * img
    Lm_max = np.max(Lm)
    out = (Lm * (1 + (Lm / (Lm_max ** 2)))) / (1 + Lm)
    out = np.clip(out, 0, 1.).astype(np.float32)
    return out


class CarlaStereoDualDataset(DatasetTemplate):
    def __init__(self, data_info, data_cfg, mode):
        super().__init__(data_info, data_cfg, mode)
        self.max_disp = getattr(self.data_info, 'MAX_DISP', 192)
        self.rescale = getattr(self.data_info, 'RESCALE', True)
        self.enable_rgb = getattr(self.data_info, 'ENABLE_RGB', False)
        self.frame_step = int(getattr(self.data_info, 'FRAME_STEP', 1))
        self.next_exposure_aug = (
            bool(getattr(self.data_info, 'NEXT_EXPOSURE_AUG', False))
            and self.mode.upper() == 'TRAINING'
        )
        self.next_exposure_range = _validate_exposure_range(
            getattr(self.data_info, 'NEXT_EXPOSURE_RANGE', [0.5, 1.5])
        )
        self.data_list = [item for item in self.data_list if self._has_next_frame(item)]

    def _get_item_paths(self, item):
        if len(item) >= 6:
            left_path, right_path, left_next_path, right_next_path, disp_path, disp_next_path = item[:6]
        elif len(item) >= 5:
            left_path, right_path, left_next_path, right_next_path, disp_path = item[:5]
            disp_next_path = _replace_last_number(disp_path, step=self.frame_step)
        elif len(item) >= 3:
            left_path, right_path, disp_path = item[:3]
            left_next_path = _replace_last_number(left_path, step=self.frame_step)
            right_next_path = _replace_last_number(right_path, step=self.frame_step)
            disp_next_path = _replace_last_number(disp_path, step=self.frame_step)
        else:
            raise ValueError(f'Unsupported split entry with {len(item)} fields: {item}')

        return left_path, right_path, left_next_path, right_next_path, disp_path, disp_next_path

    def _has_next_frame(self, item):
        try:
            _, _, left_next_path, right_next_path, _, disp_next_path = self._get_item_paths(item)
        except ValueError:
            return False

        left_next_abs = os.path.join(self.root, left_next_path)
        right_next_abs = os.path.join(self.root, right_next_path)
        disp_next_abs = os.path.join(self.root, disp_next_path)
        return (
            os.path.exists(left_next_abs)
            and os.path.exists(right_next_abs)
            and os.path.exists(disp_next_abs)
        )

    def __getitem__(self, idx):
        item = self.data_list[idx]
        left_path, right_path, left_next_path, right_next_path, disp_path, disp_next_path = self._get_item_paths(item)

        left_img = _load_stereo_image(os.path.join(self.root, left_path))
        right_img = _load_stereo_image(os.path.join(self.root, right_path))
        left_next_img = _load_stereo_image(os.path.join(self.root, left_next_path))
        right_next_img = _load_stereo_image(os.path.join(self.root, right_next_path))

        left_img = _safe_max_normalize(left_img)
        right_img = _safe_max_normalize(right_img)
        left_next_img = _safe_max_normalize(left_next_img)
        right_next_img = _safe_max_normalize(right_next_img)

        if self.next_exposure_aug:
            left_next_img, right_next_img = _apply_exposure_scale(
                left_next_img,
                right_next_img,
                self.next_exposure_range,
            )

        if self.enable_rgb:
            left_img = apply_gtm(left_img)
            right_img = apply_gtm(right_img)
            left_next_img = apply_gtm(left_next_img)
            right_next_img = apply_gtm(right_next_img)

        if self.rescale:
            left_img = radiance_scale(left_img, capacity=12)
            right_img = radiance_scale(right_img, capacity=12)
            left_next_img = radiance_scale(left_next_img, capacity=12)
            right_next_img = radiance_scale(right_next_img, capacity=12)


        left_disp = _load_disparity(os.path.join(self.root, disp_path))
        left_disp_next = _load_disparity(os.path.join(self.root, disp_next_path))

        sample = {
            'left_1': left_img,
            'right_1': right_img,
            'disp_1': left_disp,
            'occ_mask_1': np.zeros_like(left_disp, dtype=bool),
            'left_2': left_next_img,
            'right_2': right_next_img,
            'disp_2': left_disp_next,
            'occ_mask_2': np.zeros_like(left_disp_next, dtype=bool),
        }
        if self.transform is not None:
            sample = self.transform(sample)

        sample['valid_1'] = (sample['disp_1'] > 0) & (sample['disp_1'] < self.max_disp)
        sample['valid_2'] = (sample['disp_2'] > 0) & (sample['disp_2'] < self.max_disp)
        sample['disp'] = sample['disp_1']
        sample['occ_mask'] = sample['occ_mask_1']
        sample['valid'] = sample['valid_1']
        sample['index'] = idx
        sample['name'] = left_path
        sample['name_next'] = left_next_path
        return sample


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test CarlaStereoDualDataset2')
    parser.add_argument('--data_root', type=str, default='/home/lgz/dataset/ADEC/', help='Root directory of the dataset')
    parser.add_argument(
        '--split_file',
        type=str,
        default='/home/lgz/workspace/OpenStereo/dataset_split/carla_600x800/val.txt',
        help='Path to the split file',
    )
    args = parser.parse_args()

    data_info = SimpleNamespace(
        DATA_PATH=args.data_root,
        DATA_SPLIT={
            'TRAINING': args.split_file,
            'EVALUATING': args.split_file,
            'TESTING': args.split_file,
        },
        MAX_DISP=192,
        MINMAX_NORM=True,
        ENABLE_HDR=True,
        FRAME_STEP=1,
    )
    data_cfg = SimpleNamespace(
        DATA_TRANSFORM={
            'TRAINING': [],
            'EVALUATING': [],
            'TESTING': [],
        }
    )

    dataset = CarlaStereoDualDataset2(data_info=data_info, data_cfg=data_cfg, mode='training')
    print(f'Dataset length: {len(dataset)}')
    sample = dataset[0]
    print('Sample keys:', sample.keys())
    print('Left_1 image shape:', sample['left_1'].shape)
    print('Right_1 image shape:', sample['right_1'].shape)
    print('Disp_1 shape:', sample['disp_1'].shape)
    print('Left_2 image shape:', sample['left_2'].shape)
    print('Right_2 image shape:', sample['right_2'].shape)
    print('Disp_2 shape:', sample['disp_2'].shape)
