import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from types import SimpleNamespace
from collections.abc import Mapping


repo_root = Path(__file__).resolve().parents[4]
# print(f"Adding repo root to sys.path: {repo_root}")
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
try:
    # Prefer absolute imports so this file can be run directly.
    from stereo.modeling.models.aegwcnet.ae_util import ImageFormationModel
    from stereo.modeling.models.gwcnet.gwcnet import GwcNet as BaseGwcNet
except ModuleNotFoundError:
    # Fallback for environments where package root is preconfigured.
    from .ae_util import ImageFormationModel
    from ..gwcnet.gwcnet import GwcNet as BaseGwcNet



class AverageBasedAutoExposure(nn.Module):
    """Simple average-based auto-exposure.
    - forward(image) -> gamma (float tensor)
    - Optionally return adjusted image when return_image=True.
    Gamma logic: gamma = 0.5 * Mwhite / Imean (keeps same spirit as the provided code).
    """

    def __init__(self, Mwhite: float = 255.0, eps: float = 1e-6):
        """
        Args:
            Mwhite: target white level (e.g., 255 for 8-bit images).
            eps: small value to avoid division by zero.
        """
        super().__init__()
        self.Mwhite = float(Mwhite)
        self.eps = float(eps)

    def compute_mean(self, image: torch.Tensor) -> torch.Tensor:
        """Compute mean pixel value across (B, C, H, W) -> returns (B,)"""
        # Use float for numeric stability
        return image.detach().float().mean(dim=(1, 2, 3))

    def get_gamma(self, image: torch.Tensor) -> torch.Tensor:
        """Return gamma per-batch: shape (B,)"""
        Imean = self.compute_mean(image)
        gamma = 0.5 * (self.Mwhite / (Imean + self.eps))
        return gamma

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            image: input tensor (B x C x H x W)
            return_image: if True, also return the adjusted image (clamped).
        Returns:
            gamma: tensor shape (B,) -- multiplicative exposure factor to apply.
            adjusted_image (optional): image * gamma.unsqueeze(-1,-1) (clamped).
        """
        if image.ndim != 4:
            raise ValueError("image must be (B, C, H, W)")

        gamma = self.get_gamma(image)  # (B,)
        # Expand to multiply over H,W and channels when returning adjusted image
        return gamma


def exposure_value_equation(ev_vaules, time_limits, gain_limits):
    gain = torch.clamp(ev_vaules / time_limits[1], 1., None)
    expo = ev_vaules / gain
    gain = torch.clamp(gain, gain_limits[0], gain_limits[1])
    expo = torch.clamp(expo, time_limits[0], time_limits[1])
    return expo, gain


class AverageAEGwcNet(nn.Module):
    def __init__(self, cfgs, time_limits=[1., 20.], gain_limits=[1., 14.]):
        super(AverageAEGwcNet, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # cfgs = self._normalize_gwc_cfg(cfgs)

        self.image_formation_model = ImageFormationModel().to(self.device)
        self.exposure_controller = AverageBasedAutoExposure().to(self.device)
        self.disp_model = BaseGwcNet(cfgs=cfgs).to(self.device)

        self.time_limits = torch.tensor(time_limits, requires_grad=True, device=self.device)
        self.gain_limits = torch.tensor(gain_limits, requires_grad=True, device=self.device)

        self.init_exp = torch.tensor((time_limits[0] + time_limits[1]) / 2, requires_grad=True, device=self.device)
        self.init_gain = torch.tensor((gain_limits[0] + gain_limits[1]) / 2, requires_grad=True, device=self.device)

   

    def forward(self,radiance_left, radiance_right ):
        # 1.simulation
        # print(radiance_left.shape, radiance_right.shape)
        # print(f"value range:{radiance_left.min().item()} ~ {radiance_left.max().item()}, {radiance_right.min().item()} ~ {radiance_right.max().item()}")
        img_left = self.image_formation_model(radiance_left, self.init_exp, self.init_gain)
        img_right = self.image_formation_model(radiance_right, self.init_exp, self.init_gain)

        # 2.auto exposure control
        expo_values = self.exposure_controller((img_left+img_right)/2)
        expo_update, gain_update = exposure_value_equation(expo_values, self.time_limits, self.gain_limits)

        # 3.update simulation
        img_left_updated = self.image_formation_model(radiance_left, expo_update, gain_update)
        img_right_updated = self.image_formation_model(radiance_right, expo_update, gain_update)

        # 4.disparity estimation
        disp_pred = self.disp_model({'left': img_left_updated, 'right': img_right_updated})
        return disp_pred



def unit_case_run_model():
    """Minimal smoke test to verify the model can be constructed and run once."""
    # cfgs = {
    #     'max_disp': 192,
    #     'use_concat_volume': False,
    #     'concat_channels': 8,
    #     'downsample': 4,
    #     'num_groups': 8,
    # }
    

    cfgs = SimpleNamespace(
            MAX_DISP=int(192),
            USE_CONCAT_VOLUME=bool(False),
            CONCAT_CHANNELS=int(8),
            DOWNSAMPLE=int(4),
            NUM_GROUPS=int(8),
        )
    

    model = AverageAEStereoNet(cfgs=cfgs)
    model.eval()

    device = model.device
    radiance_left = torch.rand(1, 3, 256, 512, device=device)
    radiance_right = torch.rand(1, 3, 256, 512, device=device)

    with torch.no_grad():
        disp_pred = model(radiance_left, radiance_right)
    
    print(disp_pred)

    if isinstance(disp_pred, dict):
        assert 'disp_pred' in disp_pred, f"Unexpected output keys: {list(disp_pred.keys())}"
        out_shape = tuple(disp_pred['disp_pred'].shape)
    else:
        out_shape = tuple(disp_pred.shape)

    print(f"Unit case passed, output shape: {out_shape}")



if __name__ == '__main__':
    unit_case_run_model()