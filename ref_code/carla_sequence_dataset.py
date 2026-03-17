import os
import re
import glob
import cv2
import random
import numpy as np
from scipy.ndimage import distance_transform_edt, distance_transform_bf, distance_transform_cdt

from os import path as osp
import torch
from torch.utils import data as data
from torchvision.transforms.functional import normalize

# from basicsr.data.data_util import paired_paths_from_folder
from basicsr.data.transforms import augment, paired_random_crop
from basicsr.utils import  img2tensor, rgb2ycbcr, scandir, tensor2img
from basicsr.utils.registry import DATASET_REGISTRY




def crop_division(image, divisor=32):
    """
    将图像中心裁剪到指定除数的整数倍
    :param image: 输入图像 (np.ndarray)
    :param divisor: 除数，默认为 32
    :return: 裁剪后的图像
    """
    h, w = image.shape[0:2]

    # 计算需要减去的总像素
    # 例如：h=100, divisor=32 -> 100 % 32 = 4 (需要裁掉 4 像素)
    crop_h = h % divisor
    crop_w = w % divisor

    # 计算四周分别需要裁剪的像素 (中心裁剪逻辑)
    top = crop_h // 2
    bottom = h - (crop_h - top)
    left = crop_w // 2
    right = w - (crop_w - left)

    # 执行裁剪
    if image.ndim == 3:
        cropped_image = image[top:bottom, left:right, :]
    else:
        cropped_image = image[top:bottom, left:right]

    return cropped_image


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

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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


@DATASET_REGISTRY.register()
class CarlaSequenceDataset(data.Dataset):
    """Read sequence image based pn carla dataset."""
    def __init__(self, opt):
        super(CarlaSequenceDataset, self).__init__()
        self.opt = opt
        # file client (io backend)
        self.io_backend_opt = opt['io_backend']
        self.mean = opt['mean'] if 'mean' in opt else None
        self.std = opt['std'] if 'std' in opt else None
        self.use_disparity = opt['use_disparity'] if 'use_disparity' in opt else True
        self.max_disp = opt['max_disp'] if 'max_disp' in opt else 192

        # self.root_path = opt['data_root']
        self.stereo_paths, self.disp_paths = paired_paths_from_folder(opt['dataset_root'])


    def __getitem__(self, index):
        # print(f"Loading sample {index}...")
        # disp_index = (index * 2 % len(self.stereo_paths)) // 2


        # load the sceme radiance images and depth map
        img1_left = data_loader(self.stereo_paths[index][0])
        img1_right = data_loader(self.stereo_paths[index][1])

        img2_left = data_loader(self.stereo_paths[index+1][0])
        img2_right = data_loader(self.stereo_paths[index+1][1])
        img_disp = data_loader(self.disp_paths[index])

        # print(f"Loaded sample {index}: left image shape {img1_left.shape}, right image shape {img1_right.shape}, gt_disp shape {gt_disp.shape}")

        img1_left = img2tensor(img1_left, bgr2rgb=True, float32=True, minmax_norm=True)
        img1_right = img2tensor(img1_right, bgr2rgb=True, float32=True, minmax_norm=True)
        img2_left = img2tensor(img2_left, bgr2rgb=True, float32=True, minmax_norm=True)
        img2_right = img2tensor(img2_right, bgr2rgb=True, float32=True, minmax_norm=True)

        img_disp = img2tensor(img_disp, bgr2rgb=False, float32=True, minmax_norm=False)

        # flow = np.stack([-img_disp, np.zeros_like(img_disp)], axis=-1)
        # flow = img2tensor(flow, bgr2rgb=False, float32=True, minmax_norm=False)
        # flow = torch.from_numpy(flow).permute(2, 0, 1).float()  # H x W x 2 -> 2 x H x W
        # mask = (flow[0].abs() < 512) & (flow[1].abs() < 512)
        # flow = flow[:1]
        # print(f"flow shape:{flow.shape}, flow min: {flow.min().item()}, flow max: {flow.max().item()}")
        # return {'img1_left': img1_left, 'img1_right': img1_right, 'img2_left': img2_left, 'img2_right': img2_right, 'img_disp': img_disp, 'flow': flow, 'mask': mask, 'img_disp_path':self.disp_paths[index]}

        return {'img1_left': img1_left, 'img1_right': img1_right, 'img2_left': img2_left, 'img2_right': img2_right,
                'img_disp': img_disp, 'img_disp_path':self.disp_paths[index]}

    def __len__(self):
        return len(self.stereo_paths)




def apply_gtm(image):
    """Apply Reinhard tone mapping to an image."""
    # Convert to float if needed
    # Global Reinhard tone mapping
    lw_avg = np.exp(np.mean(np.log(image + 1e-5)))
    l_out = image / (1.0 + image / (2.0 * lw_avg + 1e-5))
    l_out = (l_out - l_out.min()) / (l_out.max() - l_out.min() + 1e-5)  # Normalize to [0, 1]
    l_out = np.clip(l_out * 255.0, 0, 255).astype(np.uint8)
    return l_out

if __name__ == '__main__':
    import glob
    dataset_opt = {
    'phase': 'val',
    'dataset_root': "/home/ligongzhe/data/ADEC/calra/train/",
    'crop_size': 256,
    'use_hflip': True,
    'use_rot': True,
        'io_backend': dict(type='disk'),
        'fcos': 1000.0,
        'baseline': 0.1,
    }
    # dataset = CarlaStereoDataset(dataset_opt)

    data_path = "/home/ligongzhe/data/ADEC/calra/train/"
    # path_list = list(glob.glob(path))
    # data_path = paired_paths_from_folder(data_path)
    # # print((data_path))
    # index = index * 2 % len(data_path)
    # disp_index = index // 2
    # # disp = self.disparity_reader(self.disparity_list[disp_index])
    # sample = data_path[index]
    # left_sample = sample['left_path']
    # right_sample = sample['right_path']
    # depth_sample = sample['depth_path']
    # print(len(left_sample))
    dataset = CarlaSequenceDataset(dataset_opt)
    # print(f"disp length: {len(dataset.disp_paths)},   setreo paths length: {len(dataset.stereo_paths)}")



    for i in range(5):
        sample = dataset[i]
        img_left = sample['img1_left']
        # print(f"Sample {i}: left image shape {sample['img1_left'].shape}, right image shape {sample['img1_right'].shape}, gt_disp shape {sample['img_disp'].shape}, disp path: {sample['img_disp_path']}")

        img_left = img_left.squeeze().numpy().transpose(1, 2, 0)  # C x H x W -> H x W x C
        img_left = apply_gtm(img_left)
        img_left = cv2.cvtColor(img_left, cv2.COLOR_RGB2BGR)
        cv2.imwrite(f"sample_{i}_left.png", img_left)

    # sample = dataset[0]
    # img1_left = tensor2img(sample['img1_left'])
    # img1_right = tensor2img(sample['img1_right'])
    # img_disp = sample['img_disp'].squeeze().numpy()

    # img1_left = apply_gtm(img1_left)
    # img1_right = apply_gtm(img1_right)

    # flow = tensor2img(sample['flow'])
    # mask = tensor2img(sample['mask'])

    # print(flow.shape, flow.min(), flow.max())
    # print(mask.shape, mask.min(), mask.max())





    # root_path = "/home/ligongzhe/data/ADEC/calra/train/"
    # hdr_left_path = "Experiment1/hdr_left/1.hdr"
    # hdr_right_path = "Experiment1/hdr_right/1.hdr"
    # disp_left_path = "Experiment1/ground_truth_disparity_left/disparity_map_0.npy"

    # img1_left = data_loader(os.path.join(root_path, hdr_left_path))
    # img2_left = data_loader(os.path.join(root_path, hdr_right_path))
    # disp_left = data_loader(os.path.join(root_path, disp_left_path))

    # print(img1_left.shape, img1_left.min(), img1_left.max(), "\n", img2_left.shape, img2_left.min(), img2_left.max())

    # img1_left = (img1_left ).astype(np.uint8)
    # img1_right = (img1_right).astype(np.uint8)
    # # img_disp = ((img_disp - img_disp.min()) / (img_disp.max() - img_disp.min()) ).astype(np.uint8)
    # img_disp = np.clip(img_disp / img_disp.max() * 255, 0, 255)
    # img_disp = img_disp.astype(np.uint8)

    # cv2.imwrite("img1_left_tm.png", img1_left.astype(np.uint8))
    # cv2.imwrite("img1_right_tm.png", img1_right.astype(np.uint8))
    # cv2.imwrite("img_disp.png", img_disp.astype(np.uint8))



