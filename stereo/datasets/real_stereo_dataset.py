import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

try:
    from stereo.datasets.dataset_template import DatasetTemplate
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from stereo.datasets.dataset_template import DatasetTemplate


def radiance_scale(radiance, capacity=12):
    mean_val = np.mean(radiance)
    scale = capacity / (mean_val + 1e-8)
    return radiance * scale


def apply_gtm(img, eps=1e-6, param=0.1):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    Lw_ave = np.exp(np.mean(np.log(eps + img)))
    Lm = (param / Lw_ave) * img
    Lm_max = np.max(Lm)
    out = (Lm * (1 + (Lm / (Lm_max ** 2)))) / (1 + Lm)
    out = np.clip(out, 0, 1.).astype(np.float32)
    return out


def _load_image(path):
    path = Path(path)
    ext = path.suffix.lower()
    if ext == '.npy':
        image = np.load(str(path), allow_pickle=False).astype(np.float32)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=2)
        if image.ndim != 3:
            raise ValueError('image must be HxWxC, got shape: ' + str(image.shape) + ', path: ' + str(path))
        if image.shape[2] == 1:
            image = np.repeat(image, 3, axis=2)
        if image.shape[2] > 3:
            image = image[..., :3]
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError('failed to read image: ' + str(path))
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)
    if image.ndim != 3:
        raise ValueError('image must be HxWxC, got shape: ' + str(image.shape) + ', path: ' + str(path))
    if image.shape[2] > 3:
        image = image[..., :3]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return image


def _load_disp(path):
    path = Path(path)
    if path.suffix.lower() != '.npy':
        raise NotImplementedError('Only .npy disparity maps are supported: ' + str(path))

    disp = np.load(str(path), allow_pickle=False).astype(np.float32)
    if disp.ndim == 3 and disp.shape[2] == 1:
        disp = disp[..., 0]
    if disp.ndim != 2:
        raise ValueError('disparity must be HxW, got shape: ' + str(disp.shape) + ', path: ' + str(path))
    return disp


def _resolve_image_path(root, rel_path, mode, side):
    rel_path = Path(rel_path)
    raw_path = Path(root) / rel_path

    mode_upper = mode.upper()
    if mode_upper == 'TRAINING':
        hdr_path = raw_path.with_name(side + '_hdr.npy')
        if hdr_path.exists():
            return hdr_path
    elif mode_upper in ('EVALUATING', 'TESTING'):
        rectified_path = raw_path.with_name(side + '_rectified.npy')
        if rectified_path.exists():
            return rectified_path

    if raw_path.exists():
        return raw_path
    raise FileNotFoundError('image file not found: ' + str(raw_path))


def _resolve_disp_path(root, rel_path):
    disp_path = Path(root) / Path(rel_path)
    if not disp_path.exists():
        raise FileNotFoundError('disparity file not found: ' + str(disp_path))
    return disp_path


class RealStereoDataset(DatasetTemplate):
    def __init__(self, data_info, data_cfg, mode):
        super().__init__(data_info, data_cfg, mode)
        self.max_disp = getattr(self.data_info, 'MAX_DISP', 192)
        self.enable_rgb = getattr(self.data_info, 'ENABLE_RGB', False)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        if len(item) < 3:
            raise ValueError('RealStereoDataset expects split rows: left right disp')

        left_path = _resolve_image_path(self.root, item[0], self.mode, 'left')
        right_path = _resolve_image_path(self.root, item[1], self.mode, 'right')
        disp_path = _resolve_disp_path(self.root, item[2])

        left_img = _load_image(left_path)
        right_img = _load_image(right_path)
        
        if self.enable_rgb:
            left_img = apply_gtm(left_img)
            right_img = apply_gtm(right_img)
        
        
        left_disp = _load_disp(disp_path)

        if left_img.shape[:2] != right_img.shape[:2]:
            raise ValueError(
                'left/right image size mismatch: '
                + str(left_img.shape[:2]) + ' vs ' + str(right_img.shape[:2])
                + ', left: ' + str(left_path) + ', right: ' + str(right_path)
            )
        if left_img.shape[:2] != left_disp.shape[:2]:
            raise ValueError(
                'image/disparity size mismatch: '
                + str(left_img.shape[:2]) + ' vs ' + str(left_disp.shape[:2])
                + ', image: ' + str(left_path) + ', disp: ' + str(disp_path)
            )

        sample = {
            'left': left_img,
            'right': right_img,
            'disp': left_disp,
            'occ_mask': np.zeros_like(left_disp, dtype=bool),
        }
        if self.transform is not None:
            sample = self.transform(sample)
        sample['valid'] = (sample['disp'] > 0) & (sample['disp'] < self.max_disp)
        sample['index'] = idx
        sample['name'] = str(left_path)
        return sample


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test RealStereoDataset')
    parser.add_argument('--data_root', type=str, default='/home/lgz/dataset/ADEC/real')
    parser.add_argument('--split_file', type=str, default='/home/lgz/workspace/OpenStereo/dataset_split/real_stereo/val.txt')
    parser.add_argument('--mode', type=str, default='testing', choices=['training', 'evaluating', 'testing'])
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()

    mode_key = args.mode.upper()
    data_info = SimpleNamespace(
        DATA_PATH=args.data_root,
        DATA_SPLIT={mode_key: args.split_file},
        MAX_DISP=192,
    )
    data_cfg = SimpleNamespace(
        DATA_TRANSFORM={
            mode_key: [],
        }
    )

    dataset = RealStereoDataset(data_info=data_info, data_cfg=data_cfg, mode=args.mode)
    sample = dataset[args.index]
    print('Dataset length:', len(dataset))
    print('Sample keys:', sample.keys())
    print('Left image:', sample['left'].shape, sample['left'].dtype, sample['left'].min(), sample['left'].max())
    print('Right image:', sample['right'].shape, sample['right'].dtype, sample['right'].min(), sample['right'].max())
    print('Disparity:', sample['disp'].shape, sample['disp'].dtype, sample['disp'].min(), sample['disp'].max())
    print('Valid:', sample['valid'].shape, sample['valid'].dtype, sample['valid'].sum())
    # ValueError: Cannot load file containing pickled data when allow_pickle=False
    # train/Scene13/18_10_14_267/left_hdr.npy train/Scene13/18_10_14_267/right_hdr.npy train/Scene13/18_10_14_267/disp.npy