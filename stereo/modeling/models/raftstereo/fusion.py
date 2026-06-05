import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearExposureFusion(nn.Module):
    """Fuse per-exposure feature maps into a single feature map."""

    def __init__(self, dim, num_exposures=2, num_heads=4, tau=1.0):
        super().__init__()
        self.dim = dim
        self.num_exposures = num_exposures
        self.tau = tau

        self.exp_embed = nn.Parameter(torch.zeros(1, num_exposures, dim, 1, 1))
        self.agent_embed = nn.Parameter(torch.zeros(1, num_exposures, dim))

        self.agent_norm = nn.LayerNorm(dim)
        self.token_norm = nn.LayerNorm(dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.agent_ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        self.local_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.agent_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1),
        )

    def forward(self, feats, exposure_confidence=None):
        """
        Args:
            feats: [B, N, C, H, W]
            exposure_confidence: optional [B, N, H, W]

        Returns:
            fused: [B, C, H, W]
            alpha: [B, N, H, W]
        """
        batch_size, num_exposures, channels, height, width = feats.shape
        if num_exposures != self.num_exposures:
            raise ValueError(
                f'Expected {self.num_exposures} exposures, got {num_exposures}'
            )
        if channels != self.dim:
            raise ValueError(f'Expected channel dim {self.dim}, got {channels}')

        x = feats + self.exp_embed

        tokens = x.permute(0, 1, 3, 4, 2).reshape(batch_size, num_exposures * height * width, channels)
        tokens = self.token_norm(tokens)

        agents = x.mean(dim=(-1, -2)) + self.agent_embed
        agents_q = self.agent_norm(agents)

        attn_out, _ = self.cross_attn(
            query=agents_q,
            key=tokens,
            value=tokens,
            need_weights=False,
        )

        agents = agents + attn_out
        agents = agents + self.agent_ffn(agents)

        local_feat = self.local_proj(x.reshape(batch_size * num_exposures, channels, height, width))
        local_feat = local_feat.reshape(batch_size, num_exposures, channels, height, width)
        agent_feat = self.agent_proj(agents).reshape(batch_size, num_exposures, channels, 1, 1)

        scores = (local_feat * agent_feat).sum(dim=2) / math.sqrt(channels)
        if exposure_confidence is not None:
            scores = scores + torch.log(exposure_confidence.clamp_min(1e-6))

        alpha = F.softmax(scores / self.tau, dim=1)
        fused = (alpha.unsqueeze(2) * feats).sum(dim=1)
        fused = fused + self.out_proj(fused)
        return fused, alpha