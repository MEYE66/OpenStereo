# Created: 2026-03-22  
# Author: Gongzhe Li
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plt_img(img):
    img = minmax_norm(img)
    plt.figure()
    plt.imshow(img)
    plt.show()


class ImageFormationModel:
    """
    使用 NumPy 实现的图像成像模型：
    I = quant(clip(g * (Phi * t + n_pre) + n_post))
    """

    def __init__(self, nbits=8, ):
        self.nbits = nbits
        self.gaussian_var = 3e-5
        self.poisson_scale = 3.3e-4
        # 推荐使用新的 Generator 接口
        self.rng = np.random.default_rng()

    def forward(self, radiance, t_pred, g_pred):
        """
        参数假设：
        - radiance: (N, C, H, W) 或 (H, W) 的 np.ndarray
        - t_pred, g_pred: 标量或形状为 (N,) 的数组
        """
        # 确保输入是 numpy 数组
        radiance = np.asanyarray(radiance)
        # 处理维度，使其支持广播机制 (Broadcasting)
        # 假设输入 radiance 为 (N, C, H, W)
        if isinstance(t_pred, np.ndarray):
            t_view = t_pred.reshape(-1, 1, 1, 1)
            g_view = g_pred.reshape(-1, 1, 1, 1)
        else:
            t_view, g_view = t_pred, g_pred

        # 计算噪声标准差和尺度
        # gauss_std = np.sqrt(self.gaussian_var) * (1 / t_view)
        # self.poisson_scale = self.poisson_scale * (1 / t_view)
        
        gauss_std = np.sqrt(self.gaussian_var) * (t_view)
        self.poisson_scale = self.poisson_scale * ( t_view)

        # 1. 计算信号部分
        radiance = radiance * t_view

        # 2. 散粒噪声 (Shot Noise) - 泊松分布
        # 注意：np.random.poisson 不接受负数，且在大 lambda 时可能溢出，需注意缩放
        # lam = np.maximum(signal / (p_scale + 1e-8), 0)
        # shot_noise = self.rng.poisson(lam) * p_scale * g_view
        # shot_noise = self.radiance_scale(radiance) * t_view
        shot_noise = self.rng.poisson(radiance / self.poisson_scale) * self.poisson_scale * g_view

        # 3. 读取噪声 (Readout Noise)
        readout_noise = gauss_std * self.rng.standard_normal(radiance.shape) * g_view

        # 4. ADC 噪声
        adc_noise = gauss_std * self.rng.standard_normal(radiance.shape)

        # 合成最终光强
        noise_radiance = shot_noise + readout_noise + adc_noise
        # 加上原始信号并裁剪
        img = np.clip(radiance+noise_radiance, 0.0, None)
        img = np.round(img)
        img = np.clip(img, 0.0, (2 ** self.nbits - 1))
        return img


def load_hdr_image(path):
    path = Path(path)
    if path.suffix.lower() == '.hdr':
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError('Failed to read HDR image: ' + str(path))
    elif path.suffix.lower() == '.npy':
        img = np.load(str(path))
    else:
        raise ValueError(f'Unsupported file format: {path.suffix}')

    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=2)
    if img.shape[2] >= 3:
        img = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB)
    img = minmax_norm(img)
    return img.astype(np.float32)


def minmax_norm(image):
    min_val = np.min(image)
    max_val = np.max(image)
    if max_val - min_val < 1e-6:
        return np.zeros_like(image)
    return (image - min_val) / (max_val - min_val)


def apply_gtm(img, eps=1e-6, param=0.18):
    Lw_ave = np.exp(np.mean(np.log(eps + img)))
    Lm = (param / Lw_ave) * img
    Lm_max = np.max(Lm)
    out = (Lm * (1 + (Lm / (Lm_max ** 2)))) / (1 + Lm)
    out = minmax_norm(out)
    return out.astype(np.float32)


def inverse_mu_law(img, mu=5000.0, eps=1e-8):
    # inverse μ-law
    # img = np.clip(img, 0, 1)
    # x = (np.power(1 + mu, img) - 1.0) / (mu + eps)
    x = np.expm1(img * np.log1p(mu)) / mu
    return x



def cal_dynamic_range(image):
    luminance = 0.2126 * image[:, :, 0] + 0.7152 * image[:, :, 1] + 0.0722 * image[:, :, 2]
    luminance_flat = luminance.flatten()
    luminance_flat = luminance_flat[luminance_flat > 0]  # Remove zero values
    percentile_5 = np.percentile(luminance_flat, 5)
    percentile_95 = np.percentile(luminance_flat, 95)
    d_r = 20 * np.log10(percentile_95 / (percentile_5 + 1e-8))
    return d_r




def radiance_scale(radiance, capacity):
    radiance = minmax_norm(radiance)
    mean_val = np.mean(radiance)
    scale = capacity / (mean_val + 1e-8)
    return radiance * scale




def img_vis_test():
    
    root_path = "/home/lgz/dataset/ADEC/real/val/Test1"
    # 测试图像可视化
    img = np.random.rand(100, 100, 3) * 10  # 随机生成一个高动态范围图像
    plt_img(img)





if __name__ == '__main__':
    image_formation_model = ImageFormationModel()

    # root_path = "//mnt/data1/ADEC/carla/"
    root_path = "/home/lgz/dataset/ADEC/carla/dataset"

    # folder = "Experiment168"
    folder = "Experiment1"
    id = 2
    left_img_path = Path(root_path) / folder / "hdr_left" / f"{id}.hdr"
    right_img_path = Path(root_path) / folder / "hdr_right" / f"{id}.hdr"
    disp_path = Path(root_path) / folder / "ground_truth_disparity_left" / f"disparity_map_{id}.npy"

    left_img = load_hdr_image(left_img_path) 
    right_img = load_hdr_image(right_img_path)
    disparity = np.load(disp_path).astype(np.float32)
    
    # left_img = left_img ** (1.2)
    # right_img = right_img ** (1.2)
    
    
    # left_img = np.clip(left_img*255, 0, 255).astype(np.uint8)
    # right_img = np.clip(right_img*255, 0, 255).astype(np.uint8)
    # cv2.imwrite("./left_img.png", left_img)
    # cv2.imwrite("./right_img.png", right_img)
    # exit(234)
    # left_img = inverse_mu_law(left_img, mu=3000.0, eps=1e-8)
    # right_img = inverse_mu_law(right_img, mu=3000.0, eps=1e-8)
    # left_img = apply_gtm(left_img)

    left_img = inverse_mu_law(left_img, mu=100.0, eps=1e-8)
    right_img = inverse_mu_law(right_img, mu=500.0, eps=1e-8)
    left_img = radiance_scale(left_img, 1.0)
    right_img = radiance_scale(right_img, 1.0)

    print(f"image range:{left_img.mean()} {left_img.min()}, {left_img.max()},{right_img.mean()}  {right_img.min()}, {right_img.max()}")
    print(f"dynamic range` :{cal_dynamic_range(left_img)}, {cal_dynamic_range(right_img)}")


    # print(128/0.18)

    exp, gain = 10, 15
    left_img = image_formation_model.forward(left_img, exp, gain)
    right_img = image_formation_model.forward(right_img, exp, gain)

    print(f"left range :{left_img.mean()} {left_img.min()} {left_img.max()}")
    print(f"right range :{right_img.mean()} {right_img.min()} {right_img.max()}")

    plt_img(left_img)
    plt_img(right_img)

    # exit(234)
    # left_img = minmax_norm(left_img) * 255
    # right_img = minmax_norm(right_img) * 255

    left_img = np.clip(left_img, 0, 255).astype(np.uint8)
    right_img = np.clip(right_img, 0, 255).astype(np.uint8)
    # left_img = radiance_scale(left_img, 1e2)
    # right_img = radiance_scale(right_img, 1e2)
    # plt_img(left_img)
    # plt_img(right_img)
    left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(f"./left_img.png", left_img)
    cv2.imwrite(f"./right_img.png", right_img)