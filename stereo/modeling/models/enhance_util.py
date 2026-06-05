import math
import cv2
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.color import rgb_to_hsv, hsv_to_rgb
from typing import Optional, Tuple



class DCT2D(nn.Module):
    """
    基于正交变换矩阵的 2D 离散余弦变换 (DCT-II) 和逆变换 (IDCT-II)。
    利用矩阵乘法实现 D = C * X * C^T。
    """
    def __init__(self):
        super(DCT2D, self).__init__()

    def _get_dct_matrix(self, N, dtype, device):
        n = torch.arange(N, dtype=dtype, device=device)
        k = n.view(-1, 1)
        C = torch.cos(math.pi * k * (2 * n + 1) / (2 * N))
        C[0, :] /= math.sqrt(2.0)
        C *= math.sqrt(2.0 / N)
        return C

    def forward(self, x):
        # x: (..., H, W)
        H, W = x.shape[-2], x.shape[-1]
        C_H = self._get_dct_matrix(H, x.dtype, x.device)
        C_W = self._get_dct_matrix(W, x.dtype, x.device)
        return C_H @ x @ C_W.t()

    def inverse(self, x_dct):
        # x_dct: (..., H, W)
        H, W = x_dct.shape[-2], x_dct.shape[-1]
        C_H = self._get_dct_matrix(H, x_dct.dtype, x_dct.device)
        C_W = self._get_dct_matrix(W, x_dct.dtype, x_dct.device)
        return C_H.t() @ x_dct @ C_W


def apply_local_contrast(image: torch.Tensor, alpha: float = 1.5, dct2d: Optional[DCT2D] = None):
    """
    使用 DCT 频域加权的局部对比度增强。
    支持输入形状:
      - (H, W)
      - (C, H, W)
      - (B, C, H, W)
    """
    B, C, H, W = image.shape
    dtype, device = image.dtype, image.device

    dct2d = dct2d if dct2d is not None else DCT2D()

    # 1) DCT
    d_coeff = dct2d(image)

    # 2) 构建频域权重矩阵 w(k,l)
    k = torch.arange(H, dtype=dtype, device=device).view(-1, 1)
    l = torch.arange(W, dtype=dtype, device=device).view(1, -1)

    w_k = 1.0 + ((alpha - 1.0) / max(H - 1, 1)) * k
    w_l = 1.0 + ((alpha - 1.0) / max(W - 1, 1)) * l
    w_matrix = (w_k * w_l).view(1, 1, H, W)

    # 3) 加权并 IDCT
    d_hat = d_coeff * w_matrix
    out = dct2d.inverse(d_hat)

    # 4) 裁剪到 [0, 1]
    out = torch.clamp(out, 0.0, 1.0)

    return out



def apply_local_contrast(image: torch.Tensor, alpha: float =3.2, dct2d: Optional[DCT2D] = None):
    """
    使用 DCT 频域加权的局部对比度增强。
    支持输入形状:
      - (H, W)
      - (C, H, W)
      - (B, C, H, W)
    """
    B, C, H, W = image.shape
    dtype, device = image.dtype, image.device

    dct2d = dct2d if dct2d is not None else DCT2D()

    # 1) DCT
    d_coeff = dct2d(image)

    # 2) 构建频域权重矩阵 w(k,l)
    k = torch.arange(H, dtype=dtype, device=device).view(-1, 1)
    l = torch.arange(W, dtype=dtype, device=device).view(1, -1)

    w_k = 1.0 + ((alpha - 1.0) / max(H - 1, 1)) * k
    w_l = 1.0 + ((alpha - 1.0) / max(W - 1, 1)) * l
    w_matrix = (w_k * w_l).view(1, 1, H, W)

    # 3) 加权并 IDCT
    d_hat = d_coeff * w_matrix
    out = dct2d.inverse(d_hat)

    # 4) 裁剪到 [0, 1]
    out = torch.clamp(out, 0.0, 1.0)

    return out





class SECE(nn.Module):
    """
    基于空间熵的全局图像对比度增强算法 (SECE)
    """
    def __init__(self, y_d=0.0, y_u=255.0):
        super(SECE, self).__init__()
        self.y_d = y_d
        self.y_u = y_u

    def forward(self, img):
        """
        前向传播
        :param img: (B, 1, H, W) 单通道张量 (如 HSV 中的 V 通道), 值域 [0, 1]
        :return: 对比度增强后的图像张量, 形状和值域同上
        """
        device = img.device
        B, C, H, W = img.shape
        assert C == 1, "SECE 必须应用于单通道图像 (如灰度图或 HSV 的亮度通道)"

        out = torch.zeros_like(img)

        for b in range(B):
            img_b = img[b, 0]
            # 缩放至 [0, 255] 整数空间进行灰度级统计
            img_int = (img_b * 255).round().long()

            # 提取图像中实际存在的独特灰度级 X = {x_1, ..., x_K}
            unique_vals, inverse_indices = torch.unique(img_int, return_inverse=True)
            K = len(unique_vals)

            if K <= 1:
                out[b, 0] = img_b
                continue

            # 1. 动态确定空间网格大小 M, N
            r = H / W
            N = max(1, int((K / r) ** 0.5))
            M = max(1, int((K * r) ** 0.5))

            # 为每个像素计算其所属的网格坐标 (m, n)
            i_idx = torch.arange(H, device=device).unsqueeze(1).expand(H, W)
            j_idx = torch.arange(W, device=device).unsqueeze(0).expand(H, W)
            
            m_idx = torch.clamp((i_idx * M) // H, 0, M - 1)
            n_idx = torch.clamp((j_idx * N) // W, 0, N - 1)

            # 2. 计算每个灰度级在各个网格中的出现次数 h_k(m,n)
            # 将 (k, m, n) 压缩为 1D 索引，利用 bincount 进行极速直方图统计
            linear_idx = inverse_indices * (M * N) + m_idx * N + n_idx
            hist = torch.bincount(linear_idx.flatten(), minlength=K * M * N)
            hist = hist.view(K, M, N).float()

            # 3. 计算空间熵 S_k
            # S_k = - sum(h * log2(h))
            mask = hist > 0
            hist_log = torch.zeros_like(hist)
            hist_log[mask] = hist[mask] * torch.log2(hist[mask])
            S_k = -torch.sum(hist_log, dim=(1, 2))

            # 4. 计算分布函数 f_k
            S_total = torch.sum(S_k)
            # 避免除 0，并利用 clamp 保证数值稳定性
            denom = torch.clamp(S_total - S_k, min=1e-7)
            f_k = S_k / denom
            
            # 归一化并计算 CDF
            f_k = f_k / torch.sum(f_k)
            F_k = torch.cumsum(f_k, dim=0)
            
            # 5. 映射函数 y_k = floor(F_k * (y_u - y_d) + y_d)
            y_k = torch.floor(F_k * (self.y_u - self.y_d) + self.y_d)
            
            # 使用高级索引将新灰度值映射回原图
            enhanced_int = y_k[inverse_indices]
            
            # 还原到 [0, 1] 区间
            out[b, 0] = enhanced_int / 255.0

        return out



def apply_sece_to_color(rgb_img_tensor):
    """
    将 SECE 安全地应用于彩色 RGB 图像。
    rgb_img_tensor: (B, 3, H, W) 张量, 值域 [0, 1]
    """
    # 转换到 HSV 色彩空间
    hsv_img = rgb_to_hsv(rgb_img_tensor)
    
    # 提取 V (亮度) 通道
    h, s, v = hsv_img[:, 0:1], hsv_img[:, 1:2], hsv_img[:, 2:3]
    
    # 仅对 V 通道使用 SECE 进行增强
    sece_module = SECE()
    v_enhanced = sece_module(v)
    
    # 合并通道并转回 RGB
    hsv_enhanced = torch.cat([h, s, v_enhanced], dim=1)
    rgb_enhanced = hsv_to_rgb(hsv_enhanced)
    
    return rgb_enhanced


