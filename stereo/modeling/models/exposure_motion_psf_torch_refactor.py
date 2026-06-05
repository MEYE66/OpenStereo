"""Exposure-aware motion PSF for batched image tensors.

Default tensor layout:
    - image: [B, C, H, W]
    - exposure_time: [B]
    - angle: [B]
    - psf: [B, 1, K, K]
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F


def _make_odd_int(value: float, mode: str = "ceil") -> int:
    """Convert a scalar to an odd integer."""
    value = math.ceil(float(value)) if mode == "ceil" else math.floor(float(value))
    value = max(1, value)
    if value % 2 == 0:
        value = value + 1 if mode == "ceil" else value - 1
    return max(1, value)


def _normalize_psf(kernel: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return kernel / kernel.sum(dim=(-2, -1), keepdim=True).clamp_min(eps)


def exposure_time_to_kernel_size(
    exposure_time_ms: torch.Tensor,
    velocity_scale: float = 1.5,
    min_length: int = 1,
    max_length: int = 31,
) -> torch.Tensor:
    """Map batched exposure times [B] to odd kernel lengths [B]."""
    exposure_time_ms = exposure_time_ms.reshape(-1)
    min_length = _make_odd_int(min_length, mode="ceil")
    max_length = _make_odd_int(max(max_length, min_length), mode="floor")
    if max_length < min_length:
        max_length = min_length

    length = torch.ceil(exposure_time_ms * float(velocity_scale)).to(torch.int64)
    length = torch.clamp(length, min=min_length, max=max_length)
    length = length + (length % 2 == 0).to(torch.int64)
    length = torch.where(length > max_length, length - 2, length)
    return torch.clamp(length, min=min_length, max=max_length)


def generate_motion_blur_kernel(
    exposure_time: torch.Tensor,
    angle: torch.Tensor,
    velocity_scale: float = 0.8,
    min_length: int = 1,
    max_length: int = 13,
    canvas_size: int = 31,
    line_sigma: float = 0.55,
    edge_softness: float = 0.75,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate batched motion blur PSFs with shape [B, 1, K, K]."""
    length = exposure_time_to_kernel_size(
        exposure_time,
        velocity_scale=velocity_scale,
        min_length=min_length,
        max_length=max_length,
    )
    angle = angle.reshape(-1)
    batch = length.numel()

    canvas_size = _make_odd_int(max(canvas_size, max_length), mode="ceil")

    device = length.device
    dtype = angle.dtype
    radius = canvas_size // 2
    coord = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(coord, coord, indexing="ij")
    xx = xx.unsqueeze(0)
    yy = yy.unsqueeze(0)

    theta = angle * torch.pi / 180.0
    cos_t = torch.cos(theta).view(batch, 1, 1)
    sin_t = torch.sin(theta).view(batch, 1, 1)

    x_parallel = xx * cos_t + yy * sin_t
    y_perp = -xx * sin_t + yy * cos_t

    half_len = 0.5 * length.to(dtype=dtype).view(batch, 1, 1).clamp_min(1.0)

    line_profile = torch.exp(-0.5 * (y_perp / float(line_sigma)) ** 2)
    endpoint_profile = torch.sigmoid((half_len - torch.abs(x_parallel)) / float(edge_softness))
    kernel = line_profile * endpoint_profile
    return _normalize_psf(kernel.unsqueeze(1), eps=eps), length


def apply_psf_to_image(
    image: torch.Tensor,
    psf: torch.Tensor,
    padding_mode: str = "reflect",
) -> torch.Tensor:
    """Apply batched PSFs [B, 1, K, K] to images [B, C, H, W]."""
    b, c, _, _ = image.shape
    k = psf.shape[-1]
    pad = k // 2
    image_pad = F.pad(image, (pad, pad, pad, pad), mode=padding_mode)

    x = image_pad.reshape(1, b * c, image_pad.shape[-2], image_pad.shape[-1])
    weight = psf[:, None, :, :, :].expand(b, c, 1, k, k).reshape(b * c, 1, k, k)
    y = F.conv2d(x, weight, groups=b * c)
    return y.reshape(b, c, image.shape[-2], image.shape[-1])


def apply_exposure_motion_psf(
    image: torch.Tensor,
    exposure_time: torch.Tensor,
    angle: torch.Tensor,
    velocity_scale: float = 0.8,
    min_length: int = 1,
    max_length: int = 13,
    canvas_size: int = 31,
    line_sigma: float = 0.55,
    edge_softness: float = 0.75,
    eps: float = 1e-8,
    padding_mode: str = "reflect",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Generate batched PSFs and apply them to [B, C, H, W] images."""
    psf, length = generate_motion_blur_kernel(
        exposure_time=exposure_time,
        angle=angle,
        velocity_scale=velocity_scale,
        min_length=min_length,
        max_length=max_length,
        canvas_size=canvas_size,
        line_sigma=line_sigma,
        edge_softness=edge_softness,
        eps=eps,
    )
    blurred = apply_psf_to_image(image, psf, padding_mode=padding_mode)
    return blurred, psf, length


if __name__ == "__main__":
    # Minimal sanity test.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = torch.rand(4, 3, 128, 128, device=device)
    exp = torch.tensor([1.0, 3.0, 10.0, 20.0], device=device)
    ang = torch.tensor([0.0, 30.0, 45.0, 90.0], device=device)

    out, psf, length = apply_exposure_motion_psf(
        img,
        exp,
        ang,
        velocity_scale=1.5,
        min_length=1,
        max_length=31,
        canvas_size=31,
    )
    print("input:", tuple(img.shape))
    print("output:", tuple(out.shape))
    print("psf:", tuple(psf.shape))
    print("length:", length)
    print("psf sum:", psf.sum(dim=(-2, -1)).flatten())
