import math
import torch
import torch.nn as nn
import torch.nn.functional as F




class LinearExposureFusion(nn.Module):
    """
    Feature-space exposure fusion.

    Input:
        feats: [B, N, C, H, W], N=4 exposures

    Output:
        fused: [B, C, H, W]
        alpha: [B, N, H, W]
    """
    def __init__(self, dim, num_exposures=4, num_heads=4, tau=1.0):
        super().__init__()
        self.dim = dim
        self.num_exposures = num_exposures
        self.tau = tau

        # Exposure identity embedding: tells the module which branch is dark/mid/bright.
        self.exp_embed = nn.Parameter(torch.zeros(1, num_exposures, dim, 1, 1))

        # Exposure-agent token initialized by pooled feature + learnable bias.
        self.agent_embed = nn.Parameter(torch.zeros(1, num_exposures, dim))

        self.agent_norm = nn.LayerNorm(dim)
        self.token_norm = nn.LayerNorm(dim)

        # CrossViT-style: small number of exposure agents query all exposure feature tokens.
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.agent_ffn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )

        # Local feature projection and agent projection for exposure score prediction.
        self.local_proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.agent_proj = nn.Linear(dim, dim)

        # Optional post-fusion refinement.
        self.out_proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=1)
        )

    def forward(self, feats, exposure_confidence=None):
        """
         Args:
            feats: [B, N, C, H, W]
            exposure_confidence: optional [B, N, H, W],
                e.g. reliability prior from saturation/noise mask.

        Returns:
            fused: [B, C, H, W]
            alpha: [B, N, H, W]
        """
        B, N, C, H, W = feats.shape
        assert N == self.num_exposures
        assert C == self.dim

          # Add exposure identity embedding.
        x = feats + self.exp_embed

        # Flatten all exposure feature maps as Key/Value tokens.
        # tokens: [B, N*H*W, C]
        tokens = x.permute(0, 1, 3, 4, 2).reshape(B, N * H * W, C)
        tokens = self.token_norm(tokens)

        # Exposure agents from global average pooled exposure features.
        # agents: [B, N, C]
        agents = x.mean(dim=(-1, -2)) + self.agent_embed
        agents_q = self.agent_norm(agents)

        # Linear-complexity cross-attention:
        # Query length = N, Key/Value length = N*H*W.
        attn_out, _ = self.cross_attn(
            query=agents_q,
            key=tokens,
            value=tokens,
            need_weights=False
        )

        agents = agents + attn_out
        agents = agents + self.agent_ffn(agents)

        # Predict local exposure weights.
        # local_feat: [B, N, C, H, W]
        local_feat = x.reshape(B * N, C, H, W)
        local_feat = self.local_proj(local_feat)
        local_feat = local_feat.reshape(B, N, C, H, W)

        # agent_feat: [B, N, C, 1, 1]
        agent_feat = self.agent_proj(agents).reshape(B, N, C, 1, 1)

        # Dot-product score between local feature and corresponding exposure agent.
        # scores: [B, N, H, W]
        scores = (local_feat * agent_feat).sum(dim=2) / math.sqrt(C)

        # Optional reliability prior, useful for exposure fusion.
        # For example, suppress saturated or extremely dark/noisy regions.
        if exposure_confidence is not None:
            eps = 1e-6
            scores = scores + torch.log(exposure_confidence.clamp_min(eps))
        alpha = F.softmax(scores / self.tau, dim=1)

        # Weighted sum over exposure dimension.
        fused = (alpha.unsqueeze(2) * feats).sum(dim=1)

        # Post-fusion refinement with residual.
        fused = fused + self.out_proj(fused)

        return fused


if __name__ == "__main__":
    # Test the module with dummy data.
    B, N, C, H, W = 2, 4, 64, 32, 32
    dummy_feats = torch.randn(B, N, C, H, W)
    model = LinearExposureFusion(dim=C)
    fused = model(dummy_feats)
    print("Fused shape:", fused.shape)  # Expected: [B, C, H, W]



