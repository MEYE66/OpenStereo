import os
import yaml
import cv2
import numpy as np
from scipy.interpolate import UnivariateSpline

import torch
import torch.nn as nn


class DifferentiableCRF(nn.Module):
    def __init__(self, str_crf_filepath, table_size=65536):
        """
        PyTorch 兼容的 CRF 模型，支持张量的端到端反向传播。
        """
        super().__init__()

        # 1. 生成原始的 Numpy 查找表
        g_lookup = self._generate_crf_lookup(str_crf_filepath, table_size)

        # 2. 预计算均匀分布的反向查找表 (Irradiance -> Image)
        irr_min = float(g_lookup[0])
        irr_max = float(g_lookup[-1])
        irr_uniform = np.linspace(irr_min, irr_max, table_size)

        # 使用 numpy 的插值生成反向映射表
        int_x_n = np.linspace(0, 1, table_size)
        inv_lookup = np.interp(irr_uniform, g_lookup, int_x_n)

        # 3. 将查找表注册为 PyTorch 的 Buffer (不参与模型参数更新，但保存在 device 上)
        self.register_buffer('forward_table', torch.from_numpy(g_lookup).float())
        self.register_buffer('inverse_table', torch.from_numpy(inv_lookup).float())

        self.irr_min = irr_min
        self.irr_max = irr_max

    def _generate_crf_lookup(self, str_crf_filepath, table_size):
        with open(str_crf_filepath, 'r') as f:
            data = yaml.safe_load(f.read())
            g_func_y = np.array(data["g_func"], np.float64)

        g_func_x = np.arange(0, 256, 1) / 255.0
        g_func = UnivariateSpline(g_func_x[:-1], g_func_y[:-1], s=0.001, k=5)
        int_x_n = np.linspace(0, 1, table_size, dtype=np.float64)
        return g_func(int_x_n)

    def _differentiable_interp1d(self, x, y_table, x_min=0.0, x_max=1.0):
        """
        实现 1D 的可微线性插值
        """
        # 将输入归一化到 [0, 1] 范围
        x_norm = (x - x_min) / (x_max - x_min)
        x_norm = torch.clamp(x_norm, 0.0, 1.0)

        # 映射到表格索引
        N = len(y_table) - 1
        x_scaled = x_norm * N

        # 获取相邻的两个整数索引
        idx_low = x_scaled.floor().long()
        idx_high = torch.clamp(idx_low + 1, max=N)

        # 计算插值权重
        weight = x_scaled - idx_low.float()

        # 获取对应的值
        y_low = y_table[idx_low]
        y_high = y_table[idx_high]

        # 线性插值 (torch.lerp 支持自动求导)
        return torch.lerp(y_low, y_high, weight)

    def img2irr(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """
        将图像 Tensor 转换为辐照度 (Irradiance)。
        注意：期望输入的 img_tensor 是在 [0, 1] 范围内的 float tensor (代表原 0~65535)。
        """
        return self._differentiable_interp1d(img_tensor, self.forward_table, x_min=0.0, x_max=1.0)

    def irr2img(self, irr_tensor: torch.Tensor) -> torch.Tensor:
        """
        将辐照度 (Irradiance) 转换为图像 Tensor ([0, 1] 范围)。
        """
        return self._differentiable_interp1d(irr_tensor, self.inverse_table, x_min=self.irr_min, x_max=self.irr_max)