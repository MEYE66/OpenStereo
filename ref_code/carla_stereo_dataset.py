import os
import re
import glob
import cv2
import random
import numpy as np

from os import path as osp
import torch
from torch.utils import data as data
from torchvision.transforms.functional import normalize

# from basicsr.data.data_util import paired_paths_from_folder
# from basicsr.data.transforms import augment, paired_random_crop
# from basicsr.utils import  img2tensor, rgb2ycbcr, scandir, tensor2img
# from basicsr.utils.registry import DATASET_REGISTRY



def paired_stereo_crop(image_left, image_right, depth, crop_size):
    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(image_left) else 'Numpy'

    if input_type == 'Tensor':
        h, w = image_left.size()[-2:]
    else:
        h, w = image_left.shape[0:2]

    # randomly choose top and left coordinates for crop
    top = random.randint(0, h - crop_size)
    left = random.randint(0, w - crop_size)
    # crop patches
    if input_type == 'Tensor':
        image_left = image_left[:, :, top:top + crop_size, left:left + crop_size]
        image_right = image_right[:, :, top:top + crop_size, left:left + crop_size]
        depth = depth[:, :, top:top + crop_size, left:left + crop_size]
    else:
        image_left = image_left[top:top + crop_size, left:left + crop_size, ...]
        image_right = image_right[top:top + crop_size, left:left + crop_size, ...]
        depth = depth[top:top + crop_size, left:left + crop_size, ...]

    return image_left, image_right, depth






def padding_division(image, divisor=32):
    h, w = image.shape[0:2]
    pad_h = (divisor - h % divisor) % divisor
    pad_w = (divisor - w % divisor) % divisor

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if image.ndim == 3:
        padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)
    else:
        padded_image = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
    return padded_image



def sort_key_func(file):
    numbers = re.findall(r'\d+', os.path.basename(file))
    return int(numbers[0]) if numbers else 0



def paired_paths_from_folder(folder):
    # folder_path = list(scandir(folder, full_path=True, recursive=True))
    folder_paths = list(glob.glob(os.path.join(folder, 'Experiment*')))
    left_name = "/hdr_left/*.hdr"
    right_name = "/hdr_right/*.hdr"
    depth_name = "/ground_truth_disparity_left/*.npy"

    stereo_paths = []
    disp_paths = []
    for folder in folder_paths:
        left_path = sorted(list(glob.glob(folder + left_name)), key=sort_key_func)
        right_path = sorted(list(glob.glob(folder + right_name)), key=sort_key_func)
        depth_path = sorted(list(glob.glob(folder + depth_name)), key=sort_key_func)

        if not (len(left_path) == len(right_path) == len(depth_path)):
            raise ValueError(
                f"Mismatched file counts in {folder}: "
                f"left={len(left_path)}, right={len(right_path)}, disp={len(depth_path)}"
            )

        for left_file, right_file, disp_file in zip(left_path, right_path, depth_path):
            stereo_paths.append((left_file, right_file))
            disp_paths.append(disp_file)
    return stereo_paths, disp_paths



def paired_stereo_crop(image_left, image_right, depth, crop_size):
    # determine input type: Numpy array or Tensor
    input_type = 'Tensor' if torch.is_tensor(image_left) else 'Numpy'

    if input_type == 'Tensor':
        h, w = image_left.size()[-2:]
    else:
        h, w = image_left.shape[0:2]

    # randomly choose top and left coordinates for crop
    top = random.randint(0, h - crop_size)
    left = random.randint(0, w - crop_size)

    # crop patches
    if input_type == 'Tensor':
        image_left = image_left[:, :, top:top + crop_size, left:left + crop_size]
        image_right = image_right[:, :, top:top + crop_size, left:left + crop_size]
        depth = depth[:, :, top:top + crop_size, left:left + crop_size]
    else:
        image_left = image_left[top:top + crop_size, left:left + crop_size, ...]
        image_right = image_right[top:top + crop_size, left:left + crop_size, ...]
        depth = depth[top:top + crop_size, left:left + crop_size, ...]

    return image_left, image_right, depth


def data_loader(file_name):
    ext = os.path.splitext(file_name)[-1]

    if ext == '.hdr':
        img = cv2.imread(file_name, cv2.IMREAD_UNCHANGED)
        # img = cv2.imread(file_name, cv2.IMREAD_ANYDEPTH)
        # print(f"Loaded .hdr file: {file_name}, shape: {img.shape}, dtype: {img.dtype}, min: {img.min()}, max: {img.max()}")
        min_val = np.min(img)
        max_val = np.max(img)
        normalized_img = (img - min_val) / (max_val - min_val)
        return normalized_img

    elif ext == '.npy':
        img = np.load(file_name)
        # print(f"Loaded .npy file: {file_name}, shape: {img.shape}, dtype: {img.dtype}, min: {img.min()}, max: {img.max()}")
        # read disparity file
        if len(img.shape)==2:
            return img
        # read image file
        else:
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            min_val = np.min(img)
            max_val = np.max(img)
            normalized_img = (img - min_val) / (max_val - min_val)
            return normalized_img
    return []


# @DATASET_REGISTRY.register()
class CarlaStereoDataset(data.Dataset):
    """Read sequence image based pn carla dataset."""
    def __init__(self, opt):
        super(CarlaStereoDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.use_disparity = opt['use_disparity'] if 'use_disparity' in opt else True
        self.max_disp = opt['max_disp'] if 'max_disp' in opt else 192.0

        # self.root_path = opt['data_root']
        self.stereo_paths, self.disp_paths = paired_paths_from_folder(opt['dataset_root'])


    def __getitem__(self, index):
        # print(f"Loading sample {index}...")
        # disp_index = (index * 2 % len(self.stereo_paths)) // 2
        # load the sceme radiance images and depth map
        img1_left = data_loader(self.stereo_paths[index][0])
        img1_right = data_loader(self.stereo_paths[index][1])
        img_disp = data_loader(self.disp_paths[index])

        # augmentation for training
        if self.opt['phase'] == 'train':
        #     # random crop
            img1_left, img1_right, img_disp = paired_stereo_crop(img1_left, img1_right, img_disp, self.opt['crop_size'])
        # padding to stereo model
        img1_left = padding_division(img1_left, divisor=32)
        img1_right = padding_division(img1_right, divisor=32)
        img_disp = padding_division(img_disp, divisor=32)



        img1_left = img2tensor(img1_left, bgr2rgb=True, float32=True, minmax_norm=True)
        img1_right = img2tensor(img1_right, bgr2rgb=True, float32=True, minmax_norm=True)
        # img2_left = img2tensor(img2_left, bgr2rgb=True, float32=True, minmax_norm=True)
        # img2_right = img2tensor(img2_right, bgr2rgb=True, float32=True, minmax_norm=True)
        img_disp = img2tensor(img_disp, bgr2rgb=False, float32=True, minmax_norm=False)
        img_disp = torch.clamp(img_disp, min=0.0, max=512.0)  # clip disparity to a reasonable range
        mask = (img_disp > 0) & (img_disp < self.max_disp)
        # print(f"img disp range: min {img_disp.min().item()}, max {img_disp.max().item()}")
        # flow = np.stack([-img_disp, np.zeros_like(img_disp)], axis=-1)
        # flow = img2tensor(flow, bgr2rgb=False, float32=True, minmax_norm=False)
        # flow = torch.from_numpy(flow).permute(2, 0, 1).float()  # H x W x 2 -> 2 x H x W
        # mask = (flow[0].abs() < 512) & (flow[1].abs() < 512)
        # flow = flow[:1]
        # print(f"flow shape:{flow.shape}, flow min: {flow.min().item()}, flow max: {flow.max().item()}")
        # return {'img1_left': img1_left, 'img1_right': img1_right, 'img2_left': img2_left, 'img2_right': img2_right, 'img_disp': img_disp, 'flow': flow, 'mask': mask, 'img_disp_path':self.disp_paths[index]}
        return {'img1_left': img1_left, 'img1_right': img1_right, 'img_disp': img_disp, 'mask':mask,'img_disp_path':self.disp_paths[index]}

    def __len__(self):
        return len(self.stereo_paths)




def sequence_loss(flow_preds, flow_gt, valid, loss_gamma=0.9, max_flow=700):
    """ Loss function defined over sequence of flow predictions """
    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0

    mag = torch.sum(flow_gt**2, dim=1).sqrt()

    valid = ((valid >= 0.5) & (mag < max_flow)).unsqueeze(1)
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    for i in range(n_predictions):
        assert not torch.isnan(flow_preds[i]).any() and not torch.isinf(flow_preds[i]).any()
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        assert i_loss.shape == valid.shape, [i_loss.shape, valid.shape, flow_gt.shape, flow_preds[i].shape]
        flow_loss += i_weight * i_loss[valid.bool()].mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }
    return flow_loss, metrics




def apply_gtm(image):
    """Apply Reinhard tone mapping to an image."""
    # Convert to float if needed
    # Global Reinhard tone mapping
    lw_avg = np.exp(np.mean(np.log(image + 1e-5)))
    l_out = image / (1.0 + image / (2.0 * lw_avg + 1e-5))
    l_out = (l_out - l_out.min()) / (l_out.max() - l_out.min() + 1e-5)  # Normalize to [0, 1]
    l_out = np.clip(l_out * 255.0, 0, 255).astype(np.uint8)
    return l_out



def dataset_split(data_root):
    folder_list = sorted(glob.glob(os.path.join(data_root, 'Experiment*')), key=sort_key_func)
    train_list = []
    val_list = []
    for folder in folder_list:
        folder_name = os.path.basename(folder)
        match = re.search(r'Experiment(\d+)', folder_name)
        if not match:
            continue
        exp_id = int(match.group(1))
        if exp_id >= 300:
            val_list.append(folder)
        else:
            train_list.append(folder)

    return train_list, val_list


def write_path_to_txt(path_input, target_txt):
    """Write stereo/disparity paths to txt.

    Args:
        path_input: dataset root path (str) or a list of Experiment folder paths.
        target_txt: output txt file path.
    """
    stereo_paths = []
    disp_paths = []

    if isinstance(path_input, str):
        stereo_paths, disp_paths = paired_paths_from_folder(path_input)
        rel_root = path_input
    else:
        folder_list = sorted(list(path_input), key=sort_key_func)
        left_name = '/hdr_left/*.hdr'
        right_name = '/hdr_right/*.hdr'
        depth_name = '/ground_truth_disparity_left/*.npy'

        common_root = os.path.commonpath(folder_list) if folder_list else ''
        rel_root = os.path.dirname(common_root) if common_root else ''

        for folder in folder_list:
            left_path = sorted(list(glob.glob(folder + left_name)), key=sort_key_func)
            right_path = sorted(list(glob.glob(folder + right_name)), key=sort_key_func)
            depth_path = sorted(list(glob.glob(folder + depth_name)), key=sort_key_func)

            if not (len(left_path) == len(right_path) == len(depth_path)):
                raise ValueError(
                    f"Mismatched file counts in {folder}: "
                    f"left={len(left_path)}, right={len(right_path)}, disp={len(depth_path)}"
                )

            for left_file, right_file, disp_file in zip(left_path, right_path, depth_path):
                stereo_paths.append((left_file, right_file))
                disp_paths.append(disp_file)

    os.makedirs(os.path.dirname(target_txt) or '.', exist_ok=True)
    with open(target_txt, 'w', encoding='utf-8') as f:
        for (left_path, right_path), disp_path in zip(stereo_paths, disp_paths):
            if rel_root:
                left_rel = os.path.relpath(left_path, rel_root).replace('\\', '/')
                right_rel = os.path.relpath(right_path, rel_root).replace('\\', '/')
                disp_rel = os.path.relpath(disp_path, rel_root).replace('\\', '/')
            else:
                left_rel = left_path.replace('\\', '/')
                right_rel = right_path.replace('\\', '/')
                disp_rel = disp_path.replace('\\', '/')
            # Two-line format per sample:
            # /ExperimentX/hdr_left/*.hdr   /ExperimentX/hdr_right/*.hdr
            # /ExperimentX/ground_truth_disparity_left/*.npy
            f.write(f"{left_rel} {right_rel} {disp_rel}\n")
            # f.write(f"{disp_rel}\n")
    return len(disp_paths)


if __name__ == '__main__':
    import glob
    dataset_opt = {
    'phase': 'train',
    'dataset_root': "/home/lgz/dataset/ADEC/calra/val/",
    'crop_size': 256,
    'use_hflip': True,
    'use_rot': True,
        'io_backend': dict(type='disk'),
        'fcos': 1000.0,
        'baseline': 0.1,
    }
    # dataset = CarlaStereoDataset(dataset_opt)

    data_path = "/home/lgz/dataset/ADEC/carla/dataset/"
    train_list, val_list = dataset_split(data_path)
    print(f"Total training samples: {len(train_list)}, Total validation samples: {len(val_list)}")
    train_nums = write_path_to_txt(train_list, "/home/lgz/workspace/OpenStereo/dataset_split/carla_stereo/train.txt")
    val_nums = write_path_to_txt(val_list, "/home/lgz/workspace/OpenStereo/dataset_split/carla_stereo/val.txt")
    print(f"train samples written: {train_nums}, val samples written: {val_nums}")
    exit(234)
  

