import torch
import os
import sys
import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from types import SimpleNamespace


try:
    from stereo.datasets.dataset_template import DatasetTemplate
except ModuleNotFoundError:
    # Allow running this file directly: python stereo/datasets/carla_stereo_dataset.py
    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from stereo.datasets.dataset_template import DatasetTemplate



def apply_gtm(img, eps=1e-6, param=0.1):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    Lw_ave = np.exp(np.mean(np.log(eps + img)))
    Lm = (param / Lw_ave) * img
    Lm_max = np.max(Lm)
    out = (Lm * (1 + (Lm / (Lm_max ** 2)))) / (1 + Lm)
    out = np.clip(out, 0, 1.).astype(np.float32)
    return out


def _safe_minmax_normalize(arr):
    arr = arr.astype(np.float32)
    min_val = float(arr.min())
    max_val = float(arr.max())
    scale = max(max_val - min_val, 1e-6)
    return (arr - min_val) / scale


def _safe_max_normalize(arr):
    arr = arr.astype(np.float32)
    max_val = float(arr.max())
    if max_val <= 1e-6:
        return arr
    return arr / max_val



def inverse_tmo(ldr_image, mu:float=3000.0, l_max: float = 3000.0)->np.ndarray:
    ldr = np.clip(ldr_image, 0, 1)
    hdr = l_max * (((1.0 + mu) ** ldr - 1.0) / mu)
    return hdr.astype(np.float32)


def radiance_scale(radiance, capacity=12):
    mean_val = np.mean(radiance)
    scale = capacity / (mean_val + 1e-8)
    return radiance * scale


def _load_stereo_image(path):
    ext = Path(path).suffix.lower()
    if ext == '.hdr':
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError('Failed to read HDR image: ' + path)
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        if img.shape[2] >= 3:
            img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)
        # img = _safe_minmax_normalize(img) 
        return img.astype(np.float32)
    if ext == '.npy':
        img = np.load(path)
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        if img.ndim == 3 and img.shape[2] > 3:
            img = img[..., :3]
        # img = _safe_minmax_normalize(img)
        return img.astype(np.float32)
    # img = Image.open(path).convert('RGB')
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return np.array(img, dtype=np.float32)



def _load_disparity(path):
    ext = Path(path).suffix.lower()
    if ext == '.npy':
        disp = np.load(path).astype(np.float32)
        if disp.ndim == 3:
            disp = disp[..., 0]
        return disp
    if ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp']:
        return np.array(Image.open(path), dtype=np.float32)
    raise NotImplementedError('Unsupported disparity format: ' + ext)


class CarlaStereoDataset(DatasetTemplate):
    def __init__(self, data_info, data_cfg, mode):
        super().__init__(data_info, data_cfg, mode)
        self.max_disp = getattr(self.data_info, 'MAX_DISP', 192)
        self.rescale = getattr(self.data_info, 'RESCALE', False)
        self.enable_rgb = getattr(self.data_info, 'ENABLE_RGB', False)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        full_paths = [os.path.join(self.root, x) for x in item]
        left_path, right_path, disp_path = full_paths    
        # print(f"left path: {left_path}, right path: {right_path}, disp path: {disp_path}") 
        left_img = _load_stereo_image(left_path)
        right_img = _load_stereo_image(right_path)

        left_img = _safe_max_normalize(left_img)
        right_img = _safe_max_normalize(right_img)
        if self.rescale:
            left_img = radiance_scale(left_img, capacity=12)
            right_img = radiance_scale(right_img, capacity=12)
        
        if self.enable_rgb:
            left_img = apply_gtm(left_img)
            right_img = apply_gtm(right_img)

        left_disp = _load_disparity(disp_path)
        occ_mask = np.zeros_like(left_disp, dtype=bool)
        sample = {
            'left': left_img,
            'right': right_img,
            'disp': left_disp,
            'occ_mask': occ_mask
        }
        if self.transform is not None:
            sample = self.transform(sample)
        sample['valid'] = (sample['disp'] > 0) & (sample['disp'] < self.max_disp)
        sample['index'] = idx
        sample['name'] = left_path
        return sample



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Test CarlaStereoDataset')
    parser.add_argument('--data_root', type=str, default="/home/lgz/dataset/ADEC/" ,help='Root directory of the dataset')
    # parser.add_argument('--split_file', type=str, default="/home/lgz/workspace/OpenStereo/dataset_split/carla_stereo/val.txt", help='Path to the split file')
    parser.add_argument('--split_file', type=str, default="/home/lgz/workspace/OpenStereo/dataset_split/carla_width/val.txt", help='Path to the split file')

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
    )
    data_cfg = SimpleNamespace(
        DATA_TRANSFORM={
            'TRAINING': [],
            'EVALUATING': [],
            'TESTING': [],
        }
    )

    dataset = CarlaStereoDataset(data_info=data_info, data_cfg=data_cfg, mode='training')
    
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[1]
    print('Sample keys:', sample.keys())
    print('Left image shape:', sample['left'].shape, sample['left'].min(), sample['left'].max(), sample['left'].mean() )
    print('Right image shape:', sample['right'].shape, sample['right'].min(), sample['right'].max(), sample['right'].mean() )
    print('Disparity shape:', sample['disp'].shape, sample['disp'].min(), sample['disp'].max(), sample['disp'].mean() )
    print('Occ mask shape:', sample['occ_mask'].shape, sample['occ_mask'].min(), sample['occ_mask'].max(), sample['occ_mask'].mean() )


    left_img = sample['left']
    right_img = sample['right']


    # left_img = apply_gtm(left_img)
    right_img = apply_gtm(right_img)

    left_img = _safe_minmax_normalize(left_img)
    right_img = _safe_minmax_normalize(right_img)
    left_img = np.clip(left_img*255, 0, 255).astype(np.uint8)
    right_img = np.clip(right_img*255, 0, 255).astype(np.uint8)
    disp_img = np.clip(_safe_minmax_normalize(sample['disp'])  * 255, 0, 255).astype(np.uint8)
    cv2.imwrite('test_disp.png', cv2.applyColorMap(disp_img, cv2.COLORMAP_MAGMA))
    cv2.imwrite('test_left_hdr.png', cv2.cvtColor(left_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite('test_right_hdr.png', cv2.cvtColor(right_img, cv2.COLOR_RGB2BGR))
    print("Saved test_left_hdr.png and test_right_hdr.png for visual inspection.")






