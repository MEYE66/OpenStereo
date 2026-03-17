import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class QuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, n=12):
        # LDR image max value
        max_val = 2**n - 1
        # Quantize
        # x_scaled = input * max_val
        # x_clamped = torch.clamp(x_scaled, 0, max_val)
        # x_clamped = torch.round(x_clamped)
        output = torch.clamp(torch.floor(input + 0.5), min=0, max=max_val)

        # Normalized to 0~1
        output = output / max_val
        return output
        # return x_clamped

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None



def electron_to_digital(electrons,  well_capacity=4e4, n_bits=12):
    digital_number = torch.clamp(electrons, 0, well_capacity)  # Clip to the range of 12-bit digital number
    # digital_number = digital_number / well_capacity * (2**n_bits - 1)  # Scale to the range of digital number
    return digital_number


class ImageFormationModel(nn.Module):
    def __init__(self, nbits=12):
        super(ImageFormationModel, self).__init__()
        self.nbits = nbits

        # self.gaussian_var = torch.tensor(1.0e-3, requires_grad=True)
        # self.poisson_scale = torch.tensor(3.4e-4, requires_grad=True)
        self.gaussian_var = torch.tensor(5., requires_grad=True)
        self.poisson_scale = torch.tensor(1., requires_grad=True)


    def forward(self, radiance, t_pred, g_pred):
        # add noise based on exposure value
        # adjust exposure
        # print(radiance.shape, t_pred.shape, g_pred.shape)
        t_pred = t_pred.view(-1, 1, 1, 1)
        g_pred = g_pred.view(-1, 1, 1, 1)

        gauss_std = torch.sqrt(self.gaussian_var) * t_pred
        poisson_scale = self.poisson_scale * t_pred

        radiance = radiance * t_pred
        # radiance = electron_to_digital(radiance, well_capacity=4e4) * g_pred
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


def exposure_value_equation(ev_vaules, time_limits, gain_limits):
    gain = torch.clamp(ev_vaules / time_limits[1], 1., None)
    expo = ev_vaules / gain
    gain = torch.clamp(gain, gain_limits[0], gain_limits[1])
    expo = torch.clamp(expo, time_limits[0], time_limits[1])
    return expo, gain




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









def radiance_scale(radiance,  capacity=1e2):
    mean_val = radiance.mean()
    scale = capacity / mean_val
    radiance = radiance * scale # scale to capacity
    return radiance


if __name__ == "__main__":
    # simulator
    image_formation = ImageFormationModel(expo_lb=1.0, expo_ub=10.0, nbits=12)

    # Example usage
    dataset_root = "/home/lgz/dataset/ADEC/carla/dataset/Experiment101/"
    left_path = dataset_root + "hdr_left/1.hdr"
    right_path = dataset_root + "hdr_right/1.hdr"
    print(left_path, right_path)

    left_hdr = cv2.imread(left_path, cv2.IMREAD_UNCHANGED) 
    left_hdr = radiance_scale(left_hdr, capacity=1e2)
    right_hdr = cv2.imread(right_path, cv2.IMREAD_UNCHANGED) 
    right_hdr = radiance_scale(right_hdr, capacity=1e2)

    print(f"input radiance: {left_hdr.min()} to {left_hdr.max()}, {right_hdr.min()} to {right_hdr.max()}")
    
    print(f"radiance 90% percentile: {np.percentile(left_hdr, 90)}, {np.percentile(right_hdr, 90)}")
    # exit(0)

    # Convert to PyTorch tensors
    left_tensor = torch.from_numpy(left_hdr).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    right_tensor = torch.from_numpy(right_hdr).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    
    # Example exposure values
    t_pred = torch.tensor([10.0])  # exposure multiplier
    g_pred = torch.tensor([1.0])  # gain multiplier 
    # Simulate noisy, quantized images
    left_noisy = image_formation(left_tensor, t_pred, g_pred)
    right_noisy = image_formation(right_tensor, t_pred, g_pred)
    print("Left noisy image :", left_noisy.shape, left_noisy.min().item(), left_noisy.max().item())
    print("Right noisy image :", right_noisy.shape, right_noisy.min().item(), right_noisy.max().item())
    

    left_ldr = left_noisy.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    right_ldr = right_noisy.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    left_ldr = (left_ldr - left_ldr.min()) / (left_ldr.max() - left_ldr.min() )* 255.0
    right_ldr = (right_ldr - right_ldr.min()) / (right_ldr.max() - right_ldr.min()) * 255.0
    print(f"output LDR: {left_ldr.min()} to {left_ldr.max()}, {right_ldr.min()} to {right_ldr.max()}")
    cv2.imwrite("left_ldr.png", left_ldr.astype(np.uint8))
    cv2.imwrite("right_ldr.png", right_ldr.astype(np.uint8)) 