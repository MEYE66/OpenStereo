import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


class AttExposureController(nn.Module):
    """
    ATT-ISP style parameter predictor adapted for scalar exposure control.

    Parameter Prediction Path:
    - 5 encoder stages with Conv(3x3)+ReLU+MaxPool.
    - Last feature map is projected to a local score map g(R).

    Multi-Attention Path:
    - 1-channel projection from each stage feature (multi-level features).
    - Zoom-in / zoom-out alignment to the third stage spatial size.
    - Two extra downsampling steps + conv heads to generate c(R).

    Weighted pooling:
    - exposure score = sum_j softmax(c(R_j)) * g(R_j)
    - mapped to multiplicative exposure factor within [1/max_exposure, max_exposure].
    """

    def __init__(
        self,
        in_channels=3,
        base_channels=16,
        max_exposure=10.0,
        sigmoid_scale=3.0,
    ):
        super().__init__()
        self.max_exposure = float(max_exposure)
        self.sigmoid_scale = float(sigmoid_scale)

        channels = [base_channels * (2 ** i) for i in range(5)]
        self.encoders = nn.ModuleList()
        self.level_projections = nn.ModuleList()

        prev_channels = in_channels
        for out_channels in channels:
            self.encoders.append(_EncoderBlock(prev_channels, out_channels))
            self.level_projections.append(nn.Conv2d(out_channels, 1, kernel_size=1))
            prev_channels = out_channels

        self.score_head = nn.Conv2d(channels[-1], 1, kernel_size=1)

        self.attention_path = nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 1, kernel_size=1),
        )

    @staticmethod
    def _resize_to(x, target_hw):
        return F.interpolate(x, size=target_hw, mode='bilinear', align_corners=False)

    @staticmethod
    def _weighted_pool(score_map, attention_map):
        score_flat = score_map.flatten(start_dim=2)
        attn_flat = attention_map.flatten(start_dim=2)
        weights = torch.softmax(attn_flat, dim=-1)
        pooled = (score_flat * weights).sum(dim=-1)
        return pooled.squeeze(1)

    def _build_multi_attention(self, level_features, target_hw):
        # Align all multi-level features to level-3 scale before fusion.
        ref_hw = level_features[2].shape[-2:]
        aligned = [self._resize_to(feat, ref_hw) for feat in level_features]
        fused = torch.cat(aligned, dim=1)
        attention = self.attention_path(fused)
        if attention.shape[-2:] != target_hw:
            attention = self._resize_to(attention, target_hw)
        return attention

    def forward(self, img):
        x = img
        level_features = []

        for encoder, level_proj in zip(self.encoders, self.level_projections):
            x = encoder(x)
            level_features.append(level_proj(x))

        score_map = self.score_head(x)
        attention_map = self._build_multi_attention(level_features, target_hw=score_map.shape[-2:])
        pooled_score = self._weighted_pool(score_map, attention_map)

        log_range = math.log(self.max_exposure)
        scaled = torch.tanh(self.sigmoid_scale * pooled_score) * log_range
        exposure_ratio = torch.exp(scaled)
        return exposure_ratio
