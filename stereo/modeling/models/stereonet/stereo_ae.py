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






class StereoExposureController(nn.Module):
    """
    Stereo exposure controller (vectorized, PyTorch).
    - Computes per-sample histograms from input images
    - Uses clamping ratio to detect HDR scenes and widen exposure gap
    - Uses skewness to nudge LDR scenes toward symmetric histograms
    - Randomly initialize current exposure values if not provided (per-batch)
    """
    def __init__(
        self,
        num_bins: int = 256,
        img_max_value: float = 4095.0,
        low_threshold: float = 0.02,
        high_threshold: float = 0.98,
        hdr_ratio_threshold: float = 0.05,
        alpha_skew: float = 0.1,
        exp_gap_threshold: float = 2.0,
        init_exp_range: Tuple[float, float] = (0.5, 2.0),
        eps: float = 1e-6,
    ):
        """
        Args:
            num_bins: number of histogram bins (default 256)
            img_max_value: maximum pixel value (e.g., 4095.0)
            low_threshold: fraction for low-end clamping threshold
            high_threshold: fraction for high-end clamping threshold
            hdr_ratio_threshold: threshold of low+high ratio to declare HDR clamping
            alpha_skew: gain used for skewness correction (LDR branch)
            exp_gap_threshold: minimum exposure gap required for not widening gap
            init_exp_range: (min, max) random initial exposure multiplier when missing
            eps: small number for numerical stability
        """
        super().__init__()
        self.num_bins = int(num_bins)
        self.img_max = float(img_max_value)
        self.low_threshold = float(low_threshold)
        self.high_threshold = float(high_threshold)
        self.hdr_ratio_threshold = float(hdr_ratio_threshold)
        self.alpha_skew = float(alpha_skew)
        self.exp_gap_threshold = float(exp_gap_threshold)
        self.init_exp_range = tuple(init_exp_range)
        self.eps = float(eps)

        # Precompute bin centers (0..num_bins-1)
        self.register_buffer("bin_centers", torch.arange(self.num_bins, dtype=torch.float32))

    # -------------------------
    # public forward interface
    # -------------------------
    def forward(
        self,
        left_image: torch.Tensor,
        right_image: torch.Tensor,

    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute new exposure multipliers for left and right images.

        Args:
            left_image, right_image: (B, C, H, W) float tensors with values in [0, img_max].
            current_exp_f1, current_exp_f2: (B,) tensors of current exposures. If None, they are randomly initialized.
            alpha1, alpha2: (B,) or scalar α gains for HDR gap adjustments. If None, defaults to 0.1 per sample.

        Returns:
            (new_exp_f1, new_exp_f2): each (B,) tensor
        """
        if left_image.ndim != 4 or right_image.ndim != 4:
            raise ValueError("left_image and right_image must be (B,C,H,W) tensors")

        device = left_image.device
        B = left_image.shape[0]

        current_exp_f1 = self._random_exposures(B, device)
        current_exp_f1 = current_exp_f1.to(device).float().view(-1)

        current_exp_f2 = self._random_exposures(B, device)
        current_exp_f2 = current_exp_f2.to(device).float().view(-1)

        alpha1 = torch.full((B,), 0.1, dtype=torch.float32, device=device)
        alpha1 = alpha1.to(device).float().view(-1)

        alpha2 = torch.full((B,), 0.1, dtype=torch.float32, device=device)
        alpha2 = alpha2.to(device).float().view(-1)

        hist_f1 = self._batched_soft_histogram(left_image, num_bins=self.num_bins, img_max=self.img_max)
        hist_f2 = self._batched_soft_histogram(right_image, num_bins=self.num_bins, img_max=self.img_max)

        skew_f1 = self._batch_skewness_from_hist(hist_f1)
        skew_f2 = self._batch_skewness_from_hist(hist_f2)

        clamp_ratio_f1 = self._clamping_ratio_from_hist(hist_f1)
        clamp_ratio_f2 = self._clamping_ratio_from_hist(hist_f2)

        hdr_flag1 = (clamp_ratio_f1[:, 0] > self.hdr_ratio_threshold) & (clamp_ratio_f1[:, 1] > self.hdr_ratio_threshold)
        hdr_flag2 = (clamp_ratio_f2[:, 0] > self.hdr_ratio_threshold) & (clamp_ratio_f2[:, 1] > self.hdr_ratio_threshold)
        hdr_scene = hdr_flag1 | hdr_flag2

        exp_diff = torch.abs(current_exp_f1 - current_exp_f2)
        widen_mask = hdr_scene & (exp_diff < self.exp_gap_threshold)

        new1_case1 = current_exp_f1 + alpha1 * clamp_ratio_f1[:, 0]
        new2_case1 = current_exp_f2 - alpha2 * clamp_ratio_f2[:, 1]

        new1_case2 = current_exp_f1 - alpha1 * clamp_ratio_f1[:, 1]
        new2_case2 = current_exp_f2 + alpha2 * clamp_ratio_f2[:, 0]

        exp1_gt = (current_exp_f1 > current_exp_f2)
        new_exp1_hdr = torch.where(exp1_gt, new1_case1, new1_case2)
        new_exp2_hdr = torch.where(exp1_gt, new2_case1, new2_case2)

        new_exp1_ldr = current_exp_f1 - self.alpha_skew * skew_f1
        new_exp2_ldr = current_exp_f2 - self.alpha_skew * skew_f2

        new_exp1 = torch.where(widen_mask, new_exp1_hdr, new_exp1_ldr)
        new_exp2 = torch.where(widen_mask, new_exp2_hdr, new_exp2_ldr)

        return new_exp1, new_exp2

    # -------------------------
    # helper utilities
    # -------------------------
    def _random_exposures(self, B: int, device: torch.device) -> torch.Tensor:
        """Randomly initialize exposure multipliers in self.init_exp_range, per-sample."""
        lo, hi = float(self.init_exp_range[0]), float(self.init_exp_range[1])
        return (lo + (hi - lo) * torch.rand(B, device=device)).float()

    def _batched_soft_histogram(
        self,
        images: torch.Tensor,
        num_bins: int = 256,
        img_max: float = 255.0,
        sigma: float = 3.0,
    ) -> torch.Tensor:
        """
        Differentiable soft histogram using Gaussian kernel density.

        Args:
            images: (B, C, H, W)
            num_bins: number of bins
            img_max: max pixel value
            sigma: Gaussian smoothing factor (in pixel-value domain)

        Returns:
            hist: (B, num_bins)
        """
        B, C, H, W = images.shape
        device = images.device

        # 1️⃣ convert to grayscale
        gray = images.float().mean(dim=1)  # (B, H, W)

        # 2️⃣ flatten
        gray = gray.view(B, -1)  # (B, N)
        N = gray.shape[1]

        # 3️⃣ create bin centers
        bin_centers = torch.linspace(
            0.0, img_max, num_bins, device=device
        )  # (bins,)

        # 4️⃣ compute Gaussian weights
        # shape broadcasting:
        # gray: (B, N, 1)
        # centers: (1, 1, bins)
        diff = gray.unsqueeze(-1) - bin_centers.view(1, 1, -1)

        weights = torch.exp(-(diff ** 2) / (2 * sigma ** 2))

        # 5️⃣ sum over pixels
        hist = weights.sum(dim=1)  # (B, bins)

        return hist

    def _clamping_ratio_from_hist(self, hist: torch.Tensor) -> torch.Tensor:
        """
        hist: (B, num_bins)
        returns: (B, 2) tensor with (low_ratio, high_ratio)
        """
        B = hist.shape[0]
        device = hist.device

        low_idx = int(self.low_threshold * (self.num_bins - 1))
        high_idx = int(self.high_threshold * (self.num_bins - 1))

        total = hist.sum(dim=1) + self.eps
        low = hist[:, :low_idx].sum(dim=1) / total
        high = hist[:, high_idx:].sum(dim=1) / total

        return torch.stack([low, high], dim=1)

    def _batch_skewness_from_hist(self, hist: torch.Tensor) -> torch.Tensor:
        """
        Compute Pearson's moment coefficient of skewness per sample from histogram.

        hist: (B, num_bins)
        returns: skewness (B,)
        """
        device = hist.device
        B = hist.shape[0]

        counts = hist  # (B, bins)
        total = counts.sum(dim=1) + self.eps  # (B,)

        # pixel values corresponding to bins (0..num_bins-1)
        vals = self.bin_centers.to(device)  # (bins,)

        # use fixed mean at mid-gray (as original code used fixed_mean=128)
        fixed_mean = (self.num_bins - 1) / 2.0

        diff = vals - fixed_mean  # (bins,)

        # compute numerator = sum f * (x - mean)^3
        num = (counts * (diff.view(1, -1) ** 3)).sum(dim=1)  # (B,)

        # variance = sum f * (x-mean)^2 / total
        var = (counts * (diff.view(1, -1) ** 2)).sum(dim=1) / total  # (B,)
        std = torch.sqrt(var + self.eps)

        skewness = num / (total * (std ** 3 + self.eps))

        return skewness




class StereoAEGwcNet(BaseGwcNet):
    def __init__(self, cfgs, time_limits=[1., 20.], gain_limits=[1., 14.], iters=3):
        super(StereoAEGwcNet, self).__init__(cfgs)
        self.image_formation_model = ImageFormationModel().cuda()
        self.exposure_controller = StereoExposureController().cuda()

        self.time_limits = torch.tensor(time_limits, requires_grad=True).cuda()
        self.gain_limits = torch.tensor(gain_limits, requires_grad=True).cuda()

        self.init_exp = torch.tensor((time_limits[0] + time_limits[1]) / 2, requires_grad=True).cuda()
        self.init_gain = torch.tensor((gain_limits[0] + gain_limits[1]) / 2, requires_grad=True).cuda()

    def forward(self, inputs):
        radiance_left = inputs['left']
        radiance_right = inputs['right']

        img_left = self.image_formation_model(radiance_left, self.init_exp, self.init_gain)
        img_right = self.image_formation_model(radiance_right, self.init_exp, self.init_gain)

        expo_left, expo_right = self.exposure_controller(img_left, img_right)
        
        
        
        expo_update_left, gain_update_left = exposure_value_equation(expo_left, self.time_limits, self.gain_limits)
        expo_update_right, gain_update_right = exposure_value_equation(expo_right, self.time_limits, self.gain_limits)

        img_left_updated = self.image_formation_model(radiance_left, expo_update_left, gain_update_left)
        img_right_updated = self.image_formation_model(radiance_right, expo_update_right, gain_update_right)

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
