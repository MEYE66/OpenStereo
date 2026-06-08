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




class QuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, n=12):
        # LDR image max value
        max_val = 2**n - 1
        # Quantize
        # x_scaled = input * max_val
        # x_clamped = torch.clamp(x_scaled, 0, max_val)
        # x_clamped = torch.round(x_clamped)
        output = torch.clamp(torch.floor(input + 0.5), min=0, max=max_val)

        # Normalized to 0~1
        output = output / max_val
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None



class ImageFormationModel(nn.Module):
    def __init__(self, nbits=8):
        super(ImageFormationModel, self).__init__()
        self.nbits = nbits

        # self.gaussian_var = torch.tensor(1.0e-3, requires_grad=True)
        # self.poisson_scale = torch.tensor(3.4e-4, requires_grad=True)
        self.gaussian_var = torch.tensor(5., requires_grad=True)
        self.poisson_scale = torch.tensor(1., requires_grad=True)


    def forward(self, radiance, t_pred, g_pred):
        # add noise based on exposure value
        # adjust exposure
        # print(radiance.shape, t_pred.shape, g_pred.shape)
        t_pred = t_pred.view(-1, 1, 1, 1)
        g_pred = g_pred.view(-1, 1, 1, 1)

        gauss_std = torch.sqrt(self.gaussian_var) * t_pred
        poisson_scale = self.poisson_scale * t_pred

        radiance = radiance * t_pred
        # Shot noise
        shot_noise = torch.poisson(radiance / poisson_scale) * poisson_scale * g_pred
        # Readout noise
        readout_noise = gauss_std * torch.randn_like(radiance) * g_pred
        # ADC noise
        adc_noise = gauss_std * torch.randn_like(radiance)

        noise_radiance = shot_noise + readout_noise + adc_noise
        noise_radiance = torch.clamp(radiance + noise_radiance, 0.0, None)

        noise_radiance = QuantizeSTE.apply(noise_radiance, self.nbits)
        return noise_radiance



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
    # radiance = minmax_norm(radiance)
    radiance = (radiance - radiance.min()) / (radiance.max() - radiance.min())
    if radiance.dtype == np.float32:
        # radiance = radiance.astype(np.float64)
        mean_val = np.mean(radiance)
    elif radiance.dtype == torch.float32:
        mean_val = torch.mean(radiance)
    scale = capacity / (mean_val + 1e-8)
    return radiance * scale





class KinoshitaITMO(nn.Module):
    """
    基于 Kinoshita 等人的无参数 Reinhard 逆映射算子。
    极低延迟，适合计算资源受限的边缘设备推理。
    """
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        # 论文一使用的 RGB 到亮度的转换权重
        self.register_buffer('weights', torch.tensor([0.27, 0.67, 0.06]).view(1, 3, 1, 1))

    def forward(self, ldr):
        # 假设输入 ldr 张量范围为 [0, 1]，形状为 (B, 3, H, W)
        # 1. 计算显示亮度 L_d
        L_d = torch.sum(ldr * self.weights, dim=1, keepdim=True)
        
        # 为了防止除以零或由于饱和导致负数，限制 L_d 的最大值
        L_d = torch.clamp(L_d, 0.0, 1.0 - self.eps)

        # 2. 计算伪世界亮度 L_w' (设定 A=1, G=1)
        L_w_prime = L_d / (1.0 - L_d)

        # 3. 恢复 HDR 色彩 C' = (L_w' / L_d) * C
        # 提取比率，避免 0/0 导致 NaN
        ratio = L_w_prime / (L_d + self.eps)
        hdr = ratio * ldr
        
        return hdr


class BanterleInversePhotographic(nn.Module):
    """
    基于 Banterle 等人的框架中提出的逆 Photographic 算子。
    允许通过 L_max_prime 和 L_white 进行参数化控制。
    """
    def __init__(self, l_max_prime=100.0, l_white=10.0, eps=1e-6):
        super().__init__()
        self.l_max_prime = l_max_prime
        self.l_white = l_white
        self.eps = eps
        # 论文二使用的亮度转换权重 (Standard Rec. 709 / sRGB)
        self.register_buffer('weights', torch.tensor([0.213, 0.715, 0.072]).view(1, 3, 1, 1))

    def forward(self, ldr):
        # ldr 形状 (B, 3, H, W)，范围 [0, 1]
        L_d = torch.sum(ldr * self.weights, dim=1, keepdim=True)
        L_d = torch.clamp(L_d, 0.0, 1.0 - self.eps)

        # 近似原图的对数平均亮度 (Geometric Mean)
        # 计算 exp(mean(log(L_d + delta)))
        log_L_d = torch.log(L_d + self.eps)
        L_w_bar = torch.exp(torch.mean(log_L_d, dim=[2, 3], keepdim=True))

        # 计算 alpha (Key value)
        alpha = (self.l_white * L_w_bar) / self.l_max_prime

        # 构造二次方程 a*(L_w)^2 + b*(L_w) + c = 0
        # a = alpha^2 / (L_white^2 * L_w_bar^2)
        a = (alpha**2) / ((self.l_white**2) * (L_w_bar**2) + self.eps)
        
        # b = (alpha / L_w_bar) * (1 - L_d)
        b = (alpha / (L_w_bar + self.eps)) * (1.0 - L_d)
        
        # c = -L_d
        c = -L_d

        # 求解二次方程，取最大的正根
        # det = b^2 - 4ac (因为 c 是负数，所以判别式必然为正)
        det = torch.sqrt(b**2 - 4 * a * c + self.eps)
        L_w = (-b + det) / (2 * a + self.eps)

        # 色彩恢复
        ratio = L_w / (L_d + self.eps)
        hdr_initial = ratio * ldr
        
        return hdr_initial




if __name__ == '__main__':

    import os

    import multiprocessing

    root_path = "/home/lgz/dataset/ADEC/carla_600x800/ae_methods/smoke_pid_gradient-t/ae_pid_pid_gradient/val/Experiment1/ldr_left"
    image_path = os.path.join(root_path, "1.png")
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    print(img.shape, img.dtype, img.min(), img.max())
    print(np.iinfo(np.uint16))

    print(multiprocessing.cpu_count())
    



    exit(234)
    file_path = '/home/lgz/dataset/ADEC/carla_ae/gradient/val copy.txt'  # 请替换为你的文件名
    # input_file = "input.txt"
    output_file = "/home/lgz/dataset/ADEC/carla_ae/gradient/val.txt"

    with open(file_path, "r", encoding="utf-8") as f_in, open(output_file, "w", encoding="utf-8") as f_out:
        for line in f_in:
            parts = line.strip().split()
            if len(parts) >= 4:
                f_out.write(" ".join(parts[:-1]) + "\n")
            else:
                f_out.write(line)

    exit(234)
    image_formation_model = ImageFormationModel(nbits=8)

    # root_path = "//mnt/data1/ADEC/carla/"
    root_path = "/home/lgz/dataset/ADEC/carla/dataset"

    # folder = "Experiment168"
    folder = "Experiment12"
    id = 3
    left_img_path = Path(root_path) / folder / "hdr_left" / f"{id}.hdr"
    right_img_path = Path(root_path) / folder / "hdr_right" / f"{id}.hdr"
    disp_path = Path(root_path) / folder / "ground_truth_disparity_left" / f"disparity_map_{id}.npy"

    left_img = load_hdr_image(left_img_path) 
    right_img = load_hdr_image(right_img_path)
    disparity = np.load(disp_path).astype(np.float32)
    
    # left_img = np.clip(left_img*255, 0, 255).astype(np.uint8)
    # right_img = np.clip(right_img*255, 0, 255).astype(np.uint8)
    # cv2.imwrite("./left_img.png", left_img)
    # cv2.imwrite("./right_img.png", right_img)
    # exit(234)
    # left_img = inverse_mu_law(left_img, mu=3000.0, eps=1e-8)
    # right_img = inverse_mu_law(right_img, mu=3000.0, eps=1e-8)
    # left_img = apply_gtm(left_img)

    # left_img = inverse_mu_law(left_img, mu=100.0, eps=1e-8)
    # right_img = inverse_mu_law(right_img, mu=500.0, eps=1e-8)
    left_img = radiance_scale(left_img, 1.0)
    right_img = radiance_scale(right_img, 1.0)

    print(f"image range:{left_img.mean()} {left_img.min()}, {left_img.max()},{right_img.mean()}  {right_img.min()}, {right_img.max()}")
    print(f"dynamic range` :{cal_dynamic_range(left_img)}, {cal_dynamic_range(right_img)}")
    left_img = torch.from_numpy(left_img).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
    right_img = torch.from_numpy(right_img).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)   

    exp, gain = torch.tensor(1.), torch.tensor(1.)
    left_img = image_formation_model.forward(left_img, exp, gain)
    right_img = image_formation_model.forward(right_img, exp, gain)

    print(f"left range :{left_img.mean()} {left_img.min()} {left_img.max()}")
    print(f"right range :{right_img.mean()} {right_img.min()} {right_img.max()}")


    left_img = left_img.permute(0, 2, 3, 1).squeeze(0).detach().numpy()  # (H, W, C)
    right_img = right_img.permute(0, 2, 3, 1).squeeze(0).detach().numpy()  # (H, W, C)

    left_img = minmax_norm(left_img)
    right_img = minmax_norm(right_img)

    left_img = np.clip(left_img*255, 0, 255).astype(np.uint8)
    right_img = np.clip(right_img*255, 0, 255).astype(np.uint8)
    # left_img = radiance_scale(left_img, 1e2)
    # right_img = radiance_scale(right_img, 1e2)
    # plt_img(left_img)
    # plt_img(right_img)
    left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
    right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)

    cv2.imwrite(f"./left_img.png", left_img)
    cv2.imwrite(f"./right_img.png", right_img)