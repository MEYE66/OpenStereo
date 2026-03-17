import sys
from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


repo_root = Path(__file__).resolve().parents[4]
# print(f"Adding repo root to sys.path: {repo_root}")
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from stereo.modeling.models.aegwcnet.ae_util import ImageFormationModel, exposure_value_equation
from stereo.modeling.models.gwcnet.gwcnet import GwcNet as BaseGwcNet





def _histogram_subimage_green(
    image: torch.Tensor,
    grid_size: int,
    bins: int = 4096,
    value_min: float = 0.0,
    value_max: float = 4095.0,
) -> torch.Tensor:
    batch_size, _, height, width = image.shape
    grid_h, grid_w = height // grid_size, width // grid_size

    out = []
    for b in range(batch_size):
        cur = []
        for i in range(grid_size):
            for j in range(grid_size):
                sub = image[b, 1, i * grid_h:(i + 1) * grid_h, j * grid_w:(j + 1) * grid_w]
                sub = torch.clamp(sub, value_min, value_max)
                cur.append(torch.histc(sub, bins=bins, min=value_min, max=value_max))
        cur = torch.stack(cur, dim=0).mean(dim=0)
        out.append(cur)
    return torch.stack(out, dim=0)


def _multi_scale_histogram(
        image: torch.Tensor,
        bins: int = 4096,
        value_min: float = 0.0,
        value_max: float = 4095.0,
) -> torch.Tensor:
    scales = [1, 3, 7]
    hists = [_histogram_subimage_green(image, s, bins=bins, value_min=value_min, value_max=value_max) for s in scales]
    return torch.stack(hists, dim=0).mean(dim=0)


def _batch_skewness(batch_hist: torch.Tensor, fixed_mean: Optional[float] = None, eps: float = 1e-6) -> torch.Tensor:
    device = batch_hist.device
    pixel_values = torch.arange(batch_hist.shape[1], device=device, dtype=torch.float32)
    if fixed_mean is None:
        fixed_mean = (batch_hist.shape[1] - 1) / 2.0

    diff = pixel_values.view(1, -1) - fixed_mean
    total = batch_hist.sum(dim=1) + eps
    numerator = (batch_hist * (diff ** 3)).sum(dim=1)
    variance = (batch_hist * (diff ** 2)).sum(dim=1) / total
    std = torch.sqrt(variance + eps)
    skewness = numerator / (total * (std ** 3 + eps))
    return skewness


def _clamping_ratio(batch_hist: torch.Tensor, low_threshold: float = 0.05, high_threshold: float = 0.95,
                    eps: float = 1e-6) -> torch.Tensor:
    bins = batch_hist.shape[1]
    low_idx = int(low_threshold * (bins - 1))
    high_idx = int(high_threshold * (bins - 1))

    total = batch_hist.sum(dim=1) + eps
    low_ratio = batch_hist[:, :low_idx].sum(dim=1) / total
    high_ratio = batch_hist[:, high_idx:].sum(dim=1) / total
    return torch.stack([low_ratio, high_ratio], dim=1)


class StereoExposureController(nn.Module):
    def __init__(self, min_exposure=1., max_exposure=20.0, low_threshold=0.05, high_threshold=0.95,
                 hdr_ratio_threshold=0.05, alpha_skew=0.1, exp_gap_threshold=2.0,
                 hist_bins=4096, hist_value_min=0.0, hist_value_max=4095.0):
        super().__init__()
        self.min_exposure = min_exposure
        self.max_exposure = max_exposure
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.hdr_ratio_threshold = hdr_ratio_threshold
        self.alpha_skew = alpha_skew
        self.exp_gap_threshold = exp_gap_threshold
        self.hist_bins = hist_bins
        self.hist_value_min = hist_value_min
        self.hist_value_max = hist_value_max
        self.fixed_mean = (hist_bins - 1) / 2.0

    def forward(self,
                left_image: torch.Tensor,
                right_image: torch.Tensor,
                current_exp_left: torch.Tensor,
                current_exp_right: torch.Tensor,
                alpha_left: torch.Tensor,
                alpha_right: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hist_left = _multi_scale_histogram(
            left_image,
            bins=self.hist_bins,
            value_min=self.hist_value_min,
            value_max=self.hist_value_max,
        )
        hist_right = _multi_scale_histogram(
            right_image,
            bins=self.hist_bins,
            value_min=self.hist_value_min,
            value_max=self.hist_value_max,
        )

        skew_left = _batch_skewness(hist_left, fixed_mean=self.fixed_mean)
        skew_right = _batch_skewness(hist_right, fixed_mean=self.fixed_mean)

        ratio_left = _clamping_ratio(hist_left, self.low_threshold, self.high_threshold)
        ratio_right = _clamping_ratio(hist_right, self.low_threshold, self.high_threshold)

        hdr_left = (ratio_left[:, 0] > self.hdr_ratio_threshold) & (ratio_left[:, 1] > self.hdr_ratio_threshold)
        hdr_right = (ratio_right[:, 0] > self.hdr_ratio_threshold) & (ratio_right[:, 1] > self.hdr_ratio_threshold)
        hdr_scene = hdr_left | hdr_right

        exp_diff = torch.abs(current_exp_left - current_exp_right)
        widen_mask = hdr_scene & (exp_diff < self.exp_gap_threshold)

        left_higher = current_exp_left > current_exp_right
        new_left_hdr = torch.where(
            left_higher,
            current_exp_left + alpha_left * ratio_left[:, 0],
            current_exp_left - alpha_left * ratio_left[:, 1],
        )
        new_right_hdr = torch.where(
            left_higher,
            current_exp_right - alpha_right * ratio_right[:, 1],
            current_exp_right + alpha_right * ratio_right[:, 0],
        )

        new_left_ldr = current_exp_left - self.alpha_skew * skew_left
        new_right_ldr = current_exp_right - self.alpha_skew * skew_right

        new_left = torch.where(widen_mask, new_left_hdr, new_left_ldr)
        new_right = torch.where(widen_mask, new_right_hdr, new_right_ldr)

        # new_left = torch.clamp(new_left, self.min_exposure, self.max_exposure)
        # new_right = torch.clamp(new_right, self.min_exposure, self.max_exposure)
        return new_left, new_right






class StereoAEGwcNet(BaseGwcNet):
    def __init__(self, cfgs, time_limits=[1., 15.], gain_limits=[1., 10.], iters=3):
        super(StereoAEGwcNet, self).__init__(cfgs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.iters = iters

        self.max_ev = time_limits[1] * gain_limits[1]
        self.image_formation_model = ImageFormationModel().to(self.device)
        self.exposure_controller = StereoExposureController().to(self.device)

        self.time_limits = torch.tensor(time_limits, requires_grad=True, device=self.device)
        self.gain_limits = torch.tensor(gain_limits, requires_grad=True, device=self.device)

        self.init_exp = torch.tensor((time_limits[0] + time_limits[1]) / 2, requires_grad=True, device=self.device)
        self.init_gain = torch.tensor((gain_limits[0] + gain_limits[1]) / 2, requires_grad=True, device=self.device)

    def forward(self, inputs):
        radiance_left, radiance_right = inputs['left'], inputs['right']

        img_left_updated = self.image_formation_model(radiance_left, self.init_exp, self.init_gain) * 4095.0
        img_right_updated = self.image_formation_model(radiance_right, self.init_exp, self.init_gain) * 4095.0

        batch_size = img_left_updated.shape[0]
        # current_exp_left = self.init_exp.expand(batch_size)
        # current_exp_right = self.init_exp.expand(batch_size)
        alpha_left = torch.full((batch_size,), 0.1, dtype=img_left_updated.dtype, device=img_left_updated.device)
        alpha_right = torch.full((batch_size,), 0.1, dtype=img_right_updated.dtype, device=img_right_updated.device)

        expo_update_left = self.init_exp.expand(batch_size)
        expo_update_right = self.init_exp.expand(batch_size)
        for _ in range(self.iters):
            # values = self.exposure_controller((img_left_updated + img_right_updated) / 2)
            # # expo_update, gain_update = self.update_function(expo_values, self.time_limits, self.gain_limits)
            # expo_update, gain_update = self.update_function(expo_update, values) # exp_time[B, 1], gain [B, 1]
            # img_left_updated = self.image_formation_model(radiance_left, expo_update, gain_update)
            # img_right_updated = self.image_formation_model(radiance_right, expo_update, gain_update)
        
            expo_left, expo_right = self.exposure_controller(
                img_left_updated,
                img_right_updated,
                expo_update_left,
                expo_update_right,
                alpha_left,
                alpha_right,
            )
            
            expo_update_left = torch.clamp(expo_left, self.time_limits[0], self.time_limits[1])
            expo_update_right = torch.clamp(expo_right, self.time_limits[0], self.time_limits[1])
            gain_update_left = torch.clamp(self.max_ev / expo_update_left, self.gain_limits[0], self.gain_limits[1])
            gain_update_right = torch.clamp(self.max_ev / expo_update_right, self.gain_limits[0], self.gain_limits[1])
            # expo_update_left, gain_update_left = exposure_value_equation(expo_left, self.time_limits, self.gain_limits)
            # expo_update_right, gain_update_right = exposure_value_equation(expo_right, self.time_limits, self.gain_limits)
            
            # print(f"Updated exposures: Left {expo_left.mean().item():.2f}, Right {expo_right.mean().item():.2f}")
            # print(f"Updated gains: Left {gain_update_left.mean().item():.2f}, Right {gain_update_right.mean().item():.2f}")

            img_left_updated = self.image_formation_model(radiance_left, expo_update_left, gain_update_left) * 4095.0
            img_right_updated = self.image_formation_model(radiance_right, expo_update_right, gain_update_right) * 4095.0

        img_left_updated = (img_left_updated - img_left_updated.min()) / (img_left_updated.max() - img_left_updated.min() + 1e-6) 
        img_right_updated = (img_right_updated - img_right_updated.min()) / (img_right_updated.max() - img_right_updated.min() + 1e-6) 
        disp_pred = super(StereoAEGwcNet, self).forward({'left': img_left_updated, 'right': img_right_updated})
        return disp_pred




if __name__ == "__main__":
    # simple test
    # controller = StereoExposureController()
    # dummy_left = torch.rand(2, 3, 4, 4) * 255.0
    # dummy_right = torch.rand(2, 3, 4, 4) * 255.0
    # new_exp_left, new_exp_right = controller(dummy_left, dummy_right)
    # print("New exposures:", new_exp_left, new_exp_right)
    from types import SimpleNamespace

    cfgs = SimpleNamespace(
        MAX_DISP=int(192),
        USE_CONCAT_VOLUME=bool(False),
        CONCAT_CHANNELS=int(8),
        DOWNSAMPLE=int(4),
        NUM_GROUPS=int(8),
    )
    model = StereoAEGwcNet(cfgs=cfgs)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    radiance_left = torch.rand(1, 3, 256, 512, device=device)
    radiance_right = torch.rand(1, 3, 256, 512, device=device)

    with torch.no_grad():
        disp_pred = model({'left': radiance_left, 'right': radiance_right})
    print(disp_pred)
