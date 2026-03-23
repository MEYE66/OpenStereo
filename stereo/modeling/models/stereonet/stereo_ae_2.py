import sys
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn


repo_root = Path(__file__).resolve().parents[4]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from stereo.modeling.models.ae_util import _cfg_get, ExposureControlMixin
    from stereo.modeling.models.gwcnet.gwcnet import GwcNet as BaseGwcNet
except ModuleNotFoundError:
    from ..ae_util import _cfg_get, ExposureControlMixin
    from ..gwcnet.gwcnet import GwcNet as BaseGwcNet


def _brightness_channel(image: torch.Tensor) -> torch.Tensor:
    # ITU-R BT.601 luma weights, expects image in [0,1].
    return 0.299 * image[:, 0:1] + 0.587 * image[:, 1:2] + 0.114 * image[:, 2:3]


def _histogram_subimage_brightness(
    image: torch.Tensor,
    grid_size: int,
    bins: int = 256,
    value_min: float = 0.0,
    value_max: float = 1.0,
) -> torch.Tensor:
    bright = _brightness_channel(image)
    batch_size, _, height, width = bright.shape
    grid_h, grid_w = height // grid_size, width // grid_size

    out = []
    for b in range(batch_size):
        cur = []
        for i in range(grid_size):
            for j in range(grid_size):
                sub = bright[b, 0, i * grid_h:(i + 1) * grid_h, j * grid_w:(j + 1) * grid_w]
                sub = torch.clamp(sub, value_min, value_max)
                cur.append(torch.histc(sub, bins=bins, min=value_min, max=value_max))
        cur = torch.stack(cur, dim=0).mean(dim=0)
        out.append(cur)
    return torch.stack(out, dim=0)


def _multi_scale_histogram(
    image: torch.Tensor,
    bins: int = 256,
    value_min: float = 0.0,
    value_max: float = 1.0,
) -> torch.Tensor:
    scales = [1, 3, 7]
    hists = [_histogram_subimage_brightness(image, s, bins=bins, value_min=value_min, value_max=value_max) for s in scales]
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


def _clamping_ratio(
    batch_hist: torch.Tensor,
    low_threshold: float = 0.05,
    high_threshold: float = 0.95,
    eps: float = 1e-6,
) -> torch.Tensor:
    bins = batch_hist.shape[1]
    low_idx = int(low_threshold * (bins - 1))
    high_idx = int(high_threshold * (bins - 1))

    total = batch_hist.sum(dim=1) + eps
    low_ratio = batch_hist[:, :low_idx].sum(dim=1) / total
    high_ratio = batch_hist[:, high_idx:].sum(dim=1) / total
    return torch.stack([low_ratio, high_ratio], dim=1)


class StereoExposureController(nn.Module):
    def __init__(
        self,
        low_threshold=0.05,
        high_threshold=0.95,
        hdr_ratio_threshold=0.05,
        alpha_skew=0.1,
        exp_gap_threshold=2.0,
        hist_bins=256,
        hist_value_min=0.0,
        hist_value_max=1.0,
        alpha=0.1,
    ):
        super().__init__()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.hdr_ratio_threshold = hdr_ratio_threshold
        self.alpha_skew = alpha_skew
        self.exp_gap_threshold = exp_gap_threshold
        self.hist_bins = hist_bins
        self.hist_value_min = hist_value_min
        self.hist_value_max = hist_value_max
        self.alpha = alpha
        self.fixed_mean = (hist_bins - 1) / 2.0

    def forward(
        self,
        left_image: torch.Tensor,
        right_image: torch.Tensor,
        current_exp_left: torch.Tensor,
        current_exp_right: torch.Tensor,
        alpha_left: torch.Tensor,
        alpha_right: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        left_image = torch.clamp(left_image.float(), self.hist_value_min, self.hist_value_max)
        right_image = torch.clamp(right_image.float(), self.hist_value_min, self.hist_value_max)

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
        return new_left, new_right


class StereoHistogramExposureControlMixin(ExposureControlMixin):
    def _build_exposure_controller(self, cfgs, time_limits, gain_limits):
        return StereoExposureController(
            low_threshold=float(_cfg_get(cfgs, 'AE_LOW_THRESHOLD', 0.05)),
            high_threshold=float(_cfg_get(cfgs, 'AE_HIGH_THRESHOLD', 0.95)),
            hdr_ratio_threshold=float(_cfg_get(cfgs, 'AE_HDR_RATIO_THRESHOLD', 0.05)),
            alpha_skew=float(_cfg_get(cfgs, 'AE_ALPHA_SKEW', 0.1)),
            exp_gap_threshold=float(_cfg_get(cfgs, 'AE_EXP_GAP_THRESHOLD', 2.0)),
            hist_bins=int(_cfg_get(cfgs, 'AE_HIST_BINS', 256)),
            hist_value_min=float(_cfg_get(cfgs, 'AE_HIST_VALUE_MIN', 0.0)),
            hist_value_max=float(_cfg_get(cfgs, 'AE_HIST_VALUE_MAX', 1.0)),
            alpha=float(_cfg_get(cfgs, 'AE_ALPHA', 0.1)),
        )


class StereoAEGwcNet(StereoHistogramExposureControlMixin, BaseGwcNet):
    def __init__(self, cfgs, time_limits=(5.0, 20.0), gain_limits=(1.0, 20.0), iters=3):
        super(StereoAEGwcNet, self).__init__(cfgs)
        self._init_exposure_control(cfgs, default_time_limits=time_limits, default_gain_limits=gain_limits)
        self.iters = int(_cfg_get(cfgs, 'AE_ITERS', iters))
        self.max_ev = self.time_limits[1] * self.gain_limits[1]

    def _forward_stereo(self, left_img, right_img):
        return super(StereoAEGwcNet, self).forward({'left': left_img, 'right': right_img})

    def forward(self, inputs):
        radiance_left, radiance_right = inputs['left'], inputs['right']

        img_left_updated, img_right_updated = self._build_seed_images(radiance_left, radiance_right)
        img_left_updated = torch.clamp(img_left_updated.float(), 0.0, 1.0)
        img_right_updated = torch.clamp(img_right_updated.float(), 0.0, 1.0)

        batch_size = img_left_updated.shape[0]
        alpha = float(self.exposure_controller.alpha)
        alpha_left = torch.full((batch_size,), alpha, dtype=img_left_updated.dtype, device=img_left_updated.device)
        alpha_right = torch.full((batch_size,), alpha, dtype=img_right_updated.dtype, device=img_right_updated.device)

        expo_update_left = self._expand_scalar_buffer(self.init_exp, batch_size)
        expo_update_right = self._expand_scalar_buffer(self.init_exp, batch_size)

        for _ in range(self.iters):
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

            img_left_updated = self.image_formation_model(radiance_left, expo_update_left, gain_update_left) 
            img_right_updated = self.image_formation_model(radiance_right, expo_update_right, gain_update_right) 

        return self._forward_stereo(img_left_updated, img_right_updated)


if __name__ == '__main__':
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
