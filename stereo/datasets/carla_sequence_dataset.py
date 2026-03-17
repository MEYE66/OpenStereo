import glob
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



def apply_gtm(img, eps=1e-6, param=0.18):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    Lw_ave = np.exp(np.mean(np.log(eps + img)))
    Lm = (param / Lw_ave) * img
    Lm_max = np.max(Lm)
    out = (Lm * (1 + (Lm / (Lm_max ** 2)))) / (1 + Lm)
    # out = np.clip(out, 0, 1.).astype(np.float32)
    return out


def _safe_minmax_normalize(arr):
    arr = arr.astype(np.float32)
    min_val = float(arr.min())
    max_val = float(arr.max())
    scale = max(max_val - min_val, 1e-6)
    return (arr - min_val) / scale




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
        img = _safe_minmax_normalize(img) 
        return img.astype(np.float32)

    if ext == '.npy':
        img = np.load(path)
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=2)
        if img.ndim == 3 and img.shape[2] > 3:
            img = img[..., :3]
        img = _safe_minmax_normalize(img)
        return img.astype(np.float32)

    img = Image.open(path).convert('RGB')
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




def paired_paths_from_folder(folder):
    folder_paths = list(glob.glob(os.path.join(folder, 'Experiment*')))
    # print(folder_paths)
    # exit(234)
    left_name = "/hdr_left/*.hdr"
    right_name = "/hdr_right/*.hdr"
    depth_name = "/ground_truth_disparity_left/*.npy"

    stereo_paths = []
    disp_paths = []
    for folder in folder_paths:
        left_path = sorted(list(glob.glob(folder + left_name)), key=sort_key_func)
        right_path = sorted(list(glob.glob(folder + right_name)), key=sort_key_func)
        depth_path = sorted(list(glob.glob(folder + depth_name)), key=sort_key_func)


        for idx in range(0, len(left_path)-1, 2):
            if idx + 1 < len(left_path):
                stereo_paths.append((left_path[idx], right_path[idx]))
                # load next frame as the second stereo pair for temporal training
                stereo_paths.append((left_path[idx + 1], right_path[idx + 1]))
                disp_paths.append(depth_path[idx])
    return stereo_paths, disp_paths



class CarlaSequenceDataset(DatasetTemplate):
    def __init__(self, data_info, data_cfg, mode):
        super().__init__(data_info, data_cfg, mode)
        self.max_disp = getattr(self.data_info, 'MAX_DISP', 192)

    
    def __len__(self):
        # Each sample pairs frame idx with frame idx+1, so the last frame has no pair.
        return max(0, len(self.data_list) - 1)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        item_next = self.data_list[idx + 1]

        # Guard against cross-experiment pairing: compare parent-of-parent of left image.
        if Path(item[0]).parent.parent != Path(item_next[0]).parent.parent:
            item_next = item

        left_path, right_path, disp_path = [os.path.join(self.root, x) for x in item]
        left_path2, right_path2, disp_path2 = [os.path.join(self.root, x) for x in item_next]

        left_disp = _load_disparity(disp_path)
        left_disp2 = _load_disparity(disp_path2)
        sample = {
            'left_1': _load_stereo_image(left_path),
            'right_1': _load_stereo_image(right_path),
            'disp_1': left_disp,
            'occ_mask_1': np.zeros_like(left_disp, dtype=bool),
            'left_2': _load_stereo_image(left_path2),
            'right_2': _load_stereo_image(right_path2),
            'disp_2': left_disp2,
            'occ_mask_2': np.zeros_like(left_disp2, dtype=bool),
        }
        if self.transform is not None:
            sample = self.transform(sample)
        sample['valid_1'] = (sample['disp_1'] > 0) & (sample['disp_1'] < self.max_disp)
        sample['valid_2'] = (sample['disp_2'] > 0) & (sample['disp_2'] < self.max_disp)
        sample['index'] = idx
        sample['name'] = left_path
        return sample


if __name__ == '__main__':
    import argparse
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description='Test CarlaSequenceDataset')
    parser.add_argument('--data_root', type=str, default="/home/lgz/dataset/ADEC/carla/" ,help='Root directory of the dataset')
    parser.add_argument('--split_file', type=str, default="/home/lgz/workspace/OpenStereo/dataset_split/carla_stereo/val.txt", help='Path to the split file')
    args = parser.parse_args()

    data_info = SimpleNamespace(
        DATA_PATH=args.data_root,
        DATA_SPLIT={
            'TRAINING': args.split_file,
            'EVALUATING': args.split_file,
            'TESTING': args.split_file,
        },
        MAX_DISP=512,
    )
    data_cfg = SimpleNamespace(
        DATA_TRANSFORM={
            'TRAINING': [],
            'EVALUATING': [],
            'TESTING': [],
        }
    )
    dataset = CarlaSequenceDataset(data_info=data_info, data_cfg=data_cfg, mode='training')
    
    print(f"Dataset length: {len(dataset)}")
    sample = dataset[0]
    print('Sample keys:', sample.keys())
    print('Left image shape:', sample['left_1'].shape)
    print('Right image shape:', sample['right_1'].shape)
    print('Disparity shape:', sample['disp_1'].shape)
    print('Left2 image shape:', sample['left_2'].shape)
    print('Right2 image shape:', sample['right_2'].shape)
    print('Disparity2 shape:', sample['disp_2'].shape)
    print('Occ mask shape:', sample['occ_mask_1'].shape)
    print('Occ mask2 shape:', sample['occ_mask_2'].shape)

    # dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    # for batch in dataloader:
    #     print('Batch keys:', batch.keys())
    #     print('Left image shape:', batch['left_1'].shape)
    #     print('Right image shape:', batch['right_1'].shape)
    #     print('Disparity shape:', batch['disp_1'].shape)
    #     print('Occ mask shape:', batch['occ_mask_1'].shape)
    #     break




