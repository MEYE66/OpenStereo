from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F


ArrayLikeFloat = Union[float, int, torch.Tensor]
DEFAULT_MOTION_ANGLE_DEG = 35.0


@dataclass
class PSFConfig:
    exposure_min_ms: float = 5.0
    exposure_max_ms: float = 20.0
    point_threshold_ms: float = 7.0
    max_motion_length_px: float = 13.0
    motion_gamma: float = 1.25
    kernel_size: int = 17
    default_angle_deg: float = DEFAULT_MOTION_ANGLE_DEG
    short_sigma_major_min: float = 0.55
    short_sigma_major_max: float = 1.05
    short_sigma_minor_min: float = 0.45
    short_sigma_minor_max: float = 0.70
    long_sigma_perp: float = 0.85
    endpoint_softness: float = 0.80

    def __post_init__(self) -> None:
        if self.kernel_size <= 0:
            raise ValueError('kernel_size must be positive.')
        if self.kernel_size % 2 == 0:
            raise ValueError('kernel_size must be odd.')
        if self.max_motion_length_px >= self.kernel_size:
            raise ValueError('max_motion_length_px must be smaller than kernel_size.')


def _as_batch_tensor(x: ArrayLikeFloat, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_tensor(x):
        tensor = x.to(device=device, dtype=dtype)
    else:
        tensor = torch.as_tensor(x, device=device, dtype=dtype)

    if tensor.ndim == 0:
        tensor = tensor.unsqueeze(0)

    return tensor.reshape(-1)


def _smoothstep(edge0: float, edge1: float, x: torch.Tensor) -> torch.Tensor:
    u = torch.clamp((x - edge0) / max(edge1 - edge0, 1e-12), 0.0, 1.0)
    return u * u * (3.0 - 2.0 * u)


def _exposure_to_motion_length(exposure_time_ms: torch.Tensor, cfg: PSFConfig) -> torch.Tensor:
    t = torch.clamp(exposure_time_ms, cfg.exposure_min_ms, cfg.exposure_max_ms)

    u_short = (t - cfg.exposure_min_ms) / max(cfg.point_threshold_ms - cfg.exposure_min_ms, 1e-12)
    short_length = 1.0 + 0.8 * torch.clamp(u_short, 0.0, 1.0)

    u_long = (t - cfg.point_threshold_ms) / max(cfg.exposure_max_ms - cfg.point_threshold_ms, 1e-12)
    long_length = 1.0 + torch.clamp(u_long, 0.0, 1.0).pow(cfg.motion_gamma) * (
        cfg.max_motion_length_px - 1.0
    )

    return torch.where(t <= cfg.point_threshold_ms, short_length, long_length)


def generate_exposure_aware_motion_psf_torch(
    exposure_time_ms: ArrayLikeFloat,
    angle_deg: Optional[ArrayLikeFloat] = None,
    cfg: Optional[PSFConfig] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tuple[torch.Tensor, dict]:
    cfg = cfg or PSFConfig()
    dtype = dtype or torch.float32
    if device is None:
        device = exposure_time_ms.device if torch.is_tensor(exposure_time_ms) else torch.device('cpu')

    exposure = _as_batch_tensor(exposure_time_ms, device=device, dtype=dtype)
    if angle_deg is None:
        angle_deg = cfg.default_angle_deg
    angle = _as_batch_tensor(angle_deg, device=device, dtype=dtype)

    batch_size = max(exposure.numel(), angle.numel())
    if exposure.numel() == 1:
        exposure = exposure.expand(batch_size)
    if angle.numel() == 1:
        angle = angle.expand(batch_size)
    if exposure.numel() != batch_size or angle.numel() != batch_size:
        raise ValueError('exposure_time_ms and angle_deg must be scalar or match batch length.')

    exposure = torch.clamp(exposure, cfg.exposure_min_ms, cfg.exposure_max_ms)
    radius = cfg.kernel_size // 2
    axis = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    try:
        yy, xx = torch.meshgrid(axis, axis, indexing='ij')
    except TypeError:
        yy, xx = torch.meshgrid(axis, axis)
    xx = xx.unsqueeze(0)
    yy = yy.unsqueeze(0)

    theta = angle * (torch.pi / 180.0)
    cos_theta = torch.cos(theta).unsqueeze(-1).unsqueeze(-1)
    sin_theta = torch.sin(theta).unsqueeze(-1).unsqueeze(-1)
    x_rot = cos_theta * xx + sin_theta * yy
    y_rot = -sin_theta * xx + cos_theta * yy

    length_px = _exposure_to_motion_length(exposure, cfg)
    transition = _smoothstep(cfg.point_threshold_ms, cfg.exposure_max_ms, exposure)
    u_short = torch.clamp(
        (exposure - cfg.exposure_min_ms) / max(cfg.point_threshold_ms - cfg.exposure_min_ms, 1e-12),
        0.0,
        1.0,
    )

    sigma_major = cfg.short_sigma_major_min + u_short * (
        cfg.short_sigma_major_max - cfg.short_sigma_major_min
    )
    sigma_minor = cfg.short_sigma_minor_min + u_short * (
        cfg.short_sigma_minor_max - cfg.short_sigma_minor_min
    )

    psf_short = torch.exp(
        -0.5
        * (
            (x_rot / sigma_major.unsqueeze(-1).unsqueeze(-1)).pow(2)
            + (y_rot / sigma_minor.unsqueeze(-1).unsqueeze(-1)).pow(2)
        )
    )

    half_len = torch.clamp(length_px / 2.0, min=0.5)
    outside_distance = torch.clamp(torch.abs(x_rot) - half_len.unsqueeze(-1).unsqueeze(-1), min=0.0)
    endpoint = torch.exp(-0.5 * (outside_distance / cfg.endpoint_softness).pow(2))
    perp = torch.exp(-0.5 * (y_rot / cfg.long_sigma_perp).pow(2))
    psf_long = endpoint * perp

    psf = (1.0 - transition.unsqueeze(-1).unsqueeze(-1)) * psf_short + transition.unsqueeze(-1).unsqueeze(-1) * psf_long
    psf = torch.clamp(psf, min=0.0)
    psf = psf / psf.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)

    info = {
        'exposure_time_ms': exposure,
        'angle_deg': angle,
        'kernel_size': cfg.kernel_size,
        'length_px': length_px,
        'transition_weight': transition,
    }
    return psf.unsqueeze(1), info


def apply_exposure_motion_psf_torch(
    image: torch.Tensor,
    exposure_time_ms: ArrayLikeFloat,
    angle_deg: Optional[ArrayLikeFloat] = None,
    cfg: Optional[PSFConfig] = None,
    padding_mode: str = 'reflect',
) -> Tuple[torch.Tensor, torch.Tensor, dict]:
    if image.ndim != 4:
        raise ValueError('image must have shape [B, C, H, W].')

    cfg = cfg or PSFConfig()
    batch_size, channels, height, width = image.shape
    psf, info = generate_exposure_aware_motion_psf_torch(
        exposure_time_ms,
        angle_deg,
        cfg=cfg,
        device=image.device,
        dtype=image.dtype,
    )

    if psf.shape[0] == 1 and batch_size > 1:
        psf = psf.expand(batch_size, -1, -1, -1)
    if psf.shape[0] != batch_size:
        raise ValueError('PSF batch size does not match image batch size.')

    pad = cfg.kernel_size // 2
    weight = psf.repeat_interleave(channels, dim=0)
    padded = F.pad(
        image.reshape(1, batch_size * channels, height, width),
        (pad, pad, pad, pad),
        mode=padding_mode,
    )
    blurred = F.conv2d(padded, weight=weight, bias=None, stride=1, padding=0, groups=batch_size * channels)
    return blurred.reshape(batch_size, channels, height, width), psf, info