import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class QuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, n=8):
        # LDR image max value
        max_val = 2**n - 1

        # Quantize
        x_scaled = input * max_val
        x_clamped = torch.clamp(x_scaled, 0, max_val)
        x_quantized = torch.round(x_clamped).to(torch.uint8)

        # Normalized to 0~1
        x_dequantized = x_quantized / max_val
        return x_dequantized

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class ImageFormationModel(nn.Module):
    def __init__(self, expo_lb=1.0, expo_ub=20.0, nbits=8):
        super(ImageFormationModel, self).__init__()
        self.expo_lb = expo_lb
        self.expo_ub = expo_ub
        self.nbits = nbits

        self.gaussian_var = torch.tensor(1.0e-6, requires_grad=True)
        self.poisson_scale = torch.tensor(3.0e-4, requires_grad=True)


    def forward(self, radiance, t_pred, g_pred):
        # add noise based on exposure value
        # adjust exposure
        # print(radiance.shape, t_pred.shape, g_pred.shape)
        t_pred = t_pred.view(-1, 1, 1, 1)
        g_pred = g_pred.view(-1, 1, 1, 1)

        gauss_std = torch.sqrt(self.gaussian_var) * (1 / t_pred)
        poisson_scale = self.poisson_scale * (1 / t_pred)
        # print(f"device check:{radiance.device}, {t_pred.device}, {g_pred.device}, {gauss_std.device}, {poisson_scale.device}")

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










class GradientExposureControl(nn.Module):
    """Gradient-based auto-exposure control.

    - Estimates an exposure multiplier using gradient strength across gamma anchors.
    - forward(image, current_exp=2.0, grad_update='linear') -> new_exp (tensor)

    Implementation notes / differences from the original:
    - Compute grayscale by channel-mean then avg-pool to reduce spatial resolution for efficiency.
    - Aggregate gradient information across batch and spatial dims to produce a single score curve over gamma anchors.
      (This mirrors the original code's reduction to a scalar per anchor.)
    - Use torch.searchsorted for spline interval search and torch.linalg.solve for tridiagonal solve.
    """

    def __init__(
        self,
        scale: float = 1.0,
        Lambda: float = 10.0,
        delta: float = 0.01,
        n_points: int = 61,
        default_K_p: float = 0.2,
        gamma_anchors: Optional[torch.Tensor] = None,
        downsample_kernel: int = 4,
        device: Optional[torch.device] = None,
    ):
        """
        Args:
            scale: pre-scaling applied to image before gamma transform.
            Lambda: factor inside log for gradient measure.
            delta: threshold inside relu for gradient magnitude.
            n_points: number of gamma anchors to evaluate.
            default_K_p: proportional gain used in exposure_update.
            gamma_anchors: optional 1D tensor of anchors (if None, uniform in [0.1,1.9]).
            downsample_kernel: kernel size for avg pooling prior to gradient computation.
            device: optional device; if None, will use image.device at runtime.
        """
        super().__init__()
        self.scale = float(scale)
        self.Lambda = float(Lambda)
        self.delta = float(delta)
        self.n_points = int(n_points)
        self.K_p = float(default_K_p)
        self.downsample_kernel = int(downsample_kernel)
        if gamma_anchors is None:
            self.gamma_anchors = torch.linspace(0.1, 1.9, steps=self.n_points)
        else:
            if gamma_anchors.numel() != self.n_points:
                raise ValueError("gamma_anchors length must equal n_points")
            self.gamma_anchors = gamma_anchors.clone().detach()
        self.device = device  # can be None; forward will pick up image.device

        # Prebuild sobel kernels (as float tensors; move to device in forward)
        sobel_h = torch.tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]], dtype=torch.float32)
        sobel_v = torch.tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]], dtype=torch.float32)
        # conv2d expects NCHW with kernel [out_ch, in_ch, kH, kW]; we use in_ch=1, out_ch=1
        self.register_buffer("_sobel_h", sobel_h.view(1, 1, 3, 3))
        self.register_buffer("_sobel_v", sobel_v.view(1, 1, 3, 3))

    def forward(self, image: torch.Tensor, current_exp: float = 2.0, grad_update: str = "linear") -> torch.Tensor:
        """
        Compute a new exposure multiplier based on gradient maxima.

        Args:
            image: (B x C x H x W) input images, values expected in [0, M] range (float).
            current_exp: current exposure multiplier (scalar or tensor broadcastable to (B,)).
            grad_update: 'linear' or other (controls r computation).

        Returns:
            new_exp: tensor shape (B,) with updated exposure multipliers.
        """
        if image.ndim != 4:
            raise ValueError("image must be 4D (B,C,H,W)")

        device = self.device if self.device is not None else image.device
        anchors = self.gamma_anchors.to(device)

        # 1) compute gradient info per anchor (aggregated over batch & space -> scalar per anchor).
        grad_curve = self.make_gradient_info(image.to(device), anchors)  # shape: (n_points,)
        # 2) spline interpolation across anchors (we interpolate to same anchor grid for simplicity)
        gamma_interp = anchors  # keeps same domain; we could define denser grid if needed
        gi_interp = self.make_spline(anchors, grad_curve, gamma_interp)  # (n_points,)

        # 3) find gamma that maximizes the interpolated gradient curve
        i_max = torch.argmax(gi_interp)  # scalar index
        gamma_hat = gamma_interp[i_max]   # scalar

        # 4) compute exposure update
        # make current_exp a tensor (B,) so we compute per-batch new_exp; here we treat gamma_hat as scalar
        B = image.shape[0]
        if not torch.is_tensor(current_exp):
            current_exp_t = torch.full((B,), float(current_exp), device=device)
        else:
            current_exp_t = current_exp.to(device).float().view(-1)

        new_exp = self.exposure_update(gamma_hat, current_exp=current_exp_t, grad_update=grad_update, K_p=self.K_p)
        return new_exp

    def exposure_update(self, gamma_hat: torch.Tensor, current_exp: torch.Tensor, grad_update: str = "linear", K_p: float = 0.4) -> torch.Tensor:
        """Compute a new exposure multiplier from gamma_hat.

        Args:
            gamma_hat: scalar tensor
            current_exp: (B,) tensor
            grad_update: 'linear' or other
            K_p: proportional gain

        Returns:
            new_exp: (B,) tensor
        """
        # Ensure scalar gamma value
        if torch.numel(gamma_hat) != 1:
            gamma_hat = gamma_hat.reshape(())

        d_NL_update = 0.5
        if grad_update == "linear":
            r = 1.0 - gamma_hat
        else:
            # non-linear mapping
            r = d_NL_update * torch.tan((1.0 - gamma_hat) * torch.atan(1.0 / d_NL_update))
        r = r.float()

        # If gamma_hat >= 1, use the 0.5*K_p factor
        # current_exp may be (B,)
        mask = (gamma_hat >= 1.0)
        if mask:
            # scalar True branch
            new_exp = (1.0 + 0.5 * K_p * r) * current_exp
        else:
            new_exp = (1.0 + K_p * r) * current_exp
        return new_exp

    def make_gradient_info(self, img_input: torch.Tensor, gamma_anchors: torch.Tensor) -> torch.Tensor:
        """
        Compute a single aggregated gradient score per gamma anchor.

        Args:
            img_input: (B, C, H, W) float tensor
            gamma_anchors: (n_points,) tensor on same device

        Returns:
            grad_info_curve: (n_points,) tensor aggregated over batch & space
        """
        device = img_input.device
        B = img_input.shape[0]

        # Convert to grayscale by channel mean and downsample for efficiency
        img_gray = img_input.float().mean(dim=1, keepdim=True)  # (B,1,H,W)
        if self.downsample_kernel > 1:
            img_ds = F.avg_pool2d(img_gray, kernel_size=self.downsample_kernel, stride=self.downsample_kernel, ceil_mode=False)
        else:
            img_ds = img_gray

        # N normalization factor (scalar)
        N = torch.log(torch.tensor(self.Lambda * (1.0 - self.delta) + 1.0, device=device)).clamp(min=1e-12)

        grad_info_list = []
        # Move sobel kernels to device
        sob_h = self._sobel_h.to(device)
        sob_v = self._sobel_v.to(device)

        for gamma in gamma_anchors:
            # gamma is scalar tensor possibly on device
            g = float(gamma.item()) if torch.is_tensor(gamma) else float(gamma)
            # Apply gamma transform (clamp to avoid nan)
            transformed = torch.pow(torch.clamp(img_ds * self.scale, min=0.0), g) * 0.25  # (B,1,h,w)
            # pad then conv
            padded = F.pad(transformed, (1, 1, 1, 1), mode="reflect")  # small padding for sobel 3x3
            gx = F.conv2d(padded, sob_h)
            gy = F.conv2d(padded, sob_v)
            grad_norm = torch.sqrt(gx * gx + gy * gy)
            # Remove padding effect by cropping to original ds size
            # grad_norm currently shape (B,1,h,w) with same spatial size as padded minus 2 -> original ds size
            # compute per-batch spatial sum of the transformed gradient metric
            relu_term = F.relu(grad_norm - self.delta)
            grad_info = torch.log(torch.tensor(self.Lambda, device=device) * relu_term + 1.0) / N
            # Aggregate across batch and space -> scalar
            agg = grad_info.sum()
            grad_info_list.append(agg)

        # stack -> (n_points,)
        grad_curve = torch.stack(grad_info_list).to(device)
        # Optionally normalize or smooth; keep raw curve for spline
        return grad_curve

    def make_spline(self, x: torch.Tensor, y: torch.Tensor, qx: torch.Tensor) -> torch.Tensor:
        """
        Natural cubic spline interpolation (vectorized tridiagonal solver).
        Args:
            x: (n,) strictly increasing
            y: (n,)
            qx: (m,) query points

        Returns:
            (m,) interpolated values
        """

        device = x.device
        x = x.float()
        y = y.float()
        qx = qx.float()

        n = x.numel()
        if n < 3:
            return self._linear_interp(x, y, qx)

        # ---- Step 1: build tridiagonal system ----

        h = x[1:] - x[:-1]  # (n-1)

        # lower, diag, upper (natural spline)
        lower = torch.zeros(n-1, device=device)
        diag = torch.ones(n, device=device)
        upper = torch.zeros(n-1, device=device)

        diag[1:-1] = 2 * (h[:-1] + h[1:])
        lower[:-1] = h[:-1]
        upper[1:] = h[1:]

        # RHS
        rhs = torch.zeros(n, device=device)
        rhs[1:-1] = 3 * (
            (y[2:] - y[1:-1]) / h[1:] -
            (y[1:-1] - y[:-2]) / h[:-1]
        )

        # ---- Step 2: Thomas algorithm (vectorized O(n)) ----

        # Forward elimination
        c_prime = torch.zeros(n-1, device=device)
        d_prime = torch.zeros(n, device=device)

        c_prime[0] = upper[0] / diag[0]
        d_prime[0] = rhs[0] / diag[0]

        for i in range(1, n-1):
            denom = diag[i] - lower[i-1] * c_prime[i-1]
            c_prime[i] = upper[i] / denom
            d_prime[i] = (rhs[i] - lower[i-1] * d_prime[i-1]) / denom

        d_prime[n-1] = (
            rhs[n-1] - lower[n-2] * d_prime[n-2]
        ) / (diag[n-1] - lower[n-2] * c_prime[n-2])

        # Back substitution
        c = torch.zeros(n, device=device)
        c[n-1] = d_prime[n-1]

        for i in reversed(range(n-1)):
            c[i] = d_prime[i] - c_prime[i] * c[i+1]

        # ---- Step 3: compute spline coefficients ----

        b = (y[1:] - y[:-1]) / h - h * (2*c[:-1] + c[1:]) / 3
        d = (c[1:] - c[:-1]) / (3*h)

        # ---- Step 4: evaluate spline ----

        idx = torch.searchsorted(x, qx) - 1
        idx = torch.clamp(idx, 0, n-2)

        dx = qx - x[idx]

        result = (
            y[idx]
            + b[idx] * dx
            + c[idx] * dx**2
            + d[idx] * dx**3
        )

        return result

    def _linear_interp(self, x: torch.Tensor, y: torch.Tensor, qx: torch.Tensor) -> torch.Tensor:
        """Simple linear interpolation fallback."""
        # (works for 1D x,y)
        idx = torch.searchsorted(x, qx) - 1
        idx = torch.clamp(idx, 0, x.numel() - 2)
        x0 = x[idx]
        x1 = x[idx + 1]
        y0 = y[idx]
        y1 = y[idx + 1]
        t = (qx - x0) / (x1 - x0 + 1e-12)
        return y0 + t * (y1 - y0)



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
        img_max_value: float = 255.0,
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
            img_max_value: maximum pixel value (e.g., 255.0)
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
        current_exp_f1: Optional[torch.Tensor] = None,
        current_exp_f2: Optional[torch.Tensor] = None,
        alpha1: Optional[torch.Tensor] = None,
        alpha2: Optional[torch.Tensor] = None,
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

        # ensure alphas and current_exposures are tensors of shape (B,)
        if current_exp_f1 is None:
            current_exp_f1 = self._random_exposures(B, device)
        else:
            current_exp_f1 = current_exp_f1.to(device).float().view(-1)

        if current_exp_f2 is None:
            current_exp_f2 = self._random_exposures(B, device)
        else:
            current_exp_f2 = current_exp_f2.to(device).float().view(-1)

        if alpha1 is None:
            alpha1 = torch.full((B,), 0.1, dtype=torch.float32, device=device)
        else:
            alpha1 = alpha1.to(device).float().view(-1)

        if alpha2 is None:
            alpha2 = torch.full((B,), 0.1, dtype=torch.float32, device=device)
        else:
            alpha2 = alpha2.to(device).float().view(-1)

        # 1) compute per-sample histograms for left and right (B, num_bins)
        hist_f1 = self._batched_soft_histogram(left_image, num_bins=self.num_bins, img_max=self.img_max)  # (B, bins)
        hist_f2 = self._batched_soft_histogram(right_image, num_bins=self.num_bins, img_max=self.img_max)  # (B, bins)

        # 2) compute statistics
        skew_f1 = self._batch_skewness_from_hist(hist_f1)   # (B,)
        skew_f2 = self._batch_skewness_from_hist(hist_f2)   # (B,)

        clamp_ratio_f1 = self._clamping_ratio_from_hist(hist_f1)  # (B, 2) low, high
        clamp_ratio_f2 = self._clamping_ratio_from_hist(hist_f2)  # (B, 2)

        hdr_flag1 = (clamp_ratio_f1[:, 0] > self.hdr_ratio_threshold) & (clamp_ratio_f1[:, 1] > self.hdr_ratio_threshold)
        hdr_flag2 = (clamp_ratio_f2[:, 0] > self.hdr_ratio_threshold) & (clamp_ratio_f2[:, 1] > self.hdr_ratio_threshold)
        hdr_scene = hdr_flag1 | hdr_flag2   # (B,) boolean

        # 3) determine exposure difference and masks
        exp_diff = torch.abs(current_exp_f1 - current_exp_f2)
        widen_mask = hdr_scene & (exp_diff < self.exp_gap_threshold)  # apply gap widening only if hdr_scene and small gap

        # 4) compute HDR branch candidate exposures (vectorized)
        new1_case1 = current_exp_f1 + alpha1 * clamp_ratio_f1[:, 0]   # exp1 > exp2 case
        new2_case1 = current_exp_f2 - alpha2 * clamp_ratio_f2[:, 1]

        new1_case2 = current_exp_f1 - alpha1 * clamp_ratio_f1[:, 1]   # exp1 <= exp2 case
        new2_case2 = current_exp_f2 + alpha2 * clamp_ratio_f2[:, 0]

        exp1_gt = (current_exp_f1 > current_exp_f2)
        new_exp1_hdr = torch.where(exp1_gt, new1_case1, new1_case2)
        new_exp2_hdr = torch.where(exp1_gt, new2_case1, new2_case2)

        # 5) compute LDR branch candidate exposures (skewness correction)
        new_exp1_ldr = current_exp_f1 - self.alpha_skew * skew_f1
        new_exp2_ldr = current_exp_f2 - self.alpha_skew * skew_f2

        # 6) combine branches
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