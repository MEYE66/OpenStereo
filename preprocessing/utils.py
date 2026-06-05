# Created: 2025-05-27
# Refactored: 2026-03-28

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from scipy.stats import entropy as shannon_entropy


def _find_repo_root(start_path: Path) -> Path:
    for parent in [start_path] + list(start_path.parents):
        if (parent / "stereo").is_dir():
            return parent
    raise RuntimeError("Could not locate repository root containing the stereo package.")


repo_root = _find_repo_root(Path(__file__).resolve())
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from stereo.modeling.models.exposure_motion_psf_torch_refactor import apply_exposure_motion_psf


def minmax_norm(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    min_val = float(image.min())
    max_val = float(image.max())
    if max_val - min_val < 1e-8:
        return np.zeros_like(image, dtype=np.float32)
    return (image - min_val) / (max_val - min_val)


def to_gray_u8(image_float01: np.ndarray) -> np.ndarray:
    image_u8 = np.clip(image_float01 * 255.0, 0.0, 255.0).astype(np.uint8)
    if image_u8.ndim == 2:
        return image_u8
    return cv2.cvtColor(image_u8, cv2.COLOR_BGR2GRAY)


def image_entropy(image_float01: np.ndarray) -> float:
    gray = to_gray_u8(image_float01)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).reshape(-1)
    prob = hist / max(float(hist.sum()), 1.0)
    prob = prob[prob > 0]
    return float(-(prob * np.log2(prob)).sum())


def exposure_value_equation(
    exposure_values: np.ndarray,
    time_limits: Tuple[float, float],
    gain_limits: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Numpy implementation aligned with stereo.modeling.models.ae_util."""
    exposure_values = np.asarray(exposure_values, dtype=np.float32)
    gain_limits_ratio = dB_to_ratio(np.asarray(gain_limits, dtype=np.float32))
    gain_ratio = np.clip(exposure_values / float(time_limits[1]), gain_limits_ratio[0], None)
    expo = exposure_values / gain_ratio
    gain_ratio = np.clip(gain_ratio, gain_limits_ratio[0], gain_limits_ratio[1])
    expo = np.clip(expo, float(time_limits[0]), float(time_limits[1]))
    gain = np.clip(ratio_to_dB(gain_ratio), float(gain_limits[0]), float(gain_limits[1]))
    return expo.astype(np.float32), gain.astype(np.float32)


def dB_to_ratio(dB) -> np.ndarray:
    return np.power(10.0, np.asarray(dB, dtype=np.float32) / 20.0).astype(np.float32)


def ratio_to_dB(ratio) -> np.ndarray:
    return (20.0 * np.log10(np.clip(np.asarray(ratio, dtype=np.float32), 1e-8, None))).astype(np.float32)


def radiance_scale(radiance: np.ndarray, capacity: float = 12.0) -> np.ndarray:
    radiance = (radiance - radiance.min()) / float(radiance.max() - radiance.min() + 1e-8)
    radiance = radiance.astype(np.float32)
    mean_val = float(radiance.mean())
    if mean_val <= 1e-8:
        return radiance
    return radiance * (float(capacity) / mean_val)


@dataclass
class ImageFormationNoiseConfig:
    gaussian_var: float = 0.8
    poisson_scale: float = 3.3e-4


class ImageFormationModel:
    """Pure numpy image formation simulator compatible with AE experiments."""

    def __init__(
        self,
        nbits: int = 10,
        motion_blur: bool = True,
        motion_angle_deg: Optional[float] = None,
        motion_blur_params: Optional[dict] = None,
        noise_cfg: Optional[ImageFormationNoiseConfig] = None,
        seed: Optional[int] = None,
    ):
        self.nbits = int(nbits)
        self.motion_blur = bool(motion_blur)
        self.noise_cfg = noise_cfg or ImageFormationNoiseConfig()
        self.rng = np.random.default_rng(seed)
        self.adc_noise_scale = np.float32(1.0 / float(self.nbits))
        self.motion_blur_params = {
            "velocity_scale": 0.8,
            "min_length": 1,
            "max_length": 15,
            "canvas_size": 31,
            "line_sigma": 0.55,
            "edge_softness": 0.75,
            "eps": 1e-8,
        }
        if motion_blur_params is not None:
            self.motion_blur_params.update(motion_blur_params)
        self.motion_angle_deg = 10.0 if motion_angle_deg is None else float(motion_angle_deg)
        self.motion_psf = None
        self.motion_psf_info = None

    def _apply_exposure_motion_blur(self, radiance: np.ndarray, exposure_time_ms: float) -> np.ndarray:
        if not self.motion_blur:
            return radiance

        radiance_tensor = torch.from_numpy(radiance.astype(np.float32, copy=False)).permute(2, 0, 1).unsqueeze(0)
        exposure_tensor = torch.tensor([float(exposure_time_ms)], dtype=torch.float32)
        angle_tensor = torch.tensor([self.motion_angle_deg], dtype=torch.float32)

        with torch.no_grad():
            blurred_radiance, motion_psf, motion_length = apply_exposure_motion_psf(
                radiance_tensor,
                exposure_time=exposure_tensor,
                angle=angle_tensor,
                **self.motion_blur_params,
            )

        self.motion_psf = motion_psf.squeeze(0).cpu().numpy()
        self.motion_psf_info = {
            "exposure_time_ms": float(exposure_time_ms),
            "angle_deg": float(self.motion_angle_deg),
            "kernel_size": int(motion_psf.shape[-1]),
            "length_px": int(motion_length.item()),
        }
        return blurred_radiance.squeeze(0).permute(1, 2, 0).cpu().numpy()

    def _quantize(self, image: np.ndarray) -> np.ndarray:
        max_val = float((2 ** self.nbits) - 1)
        quantized = np.clip(np.floor(image + 0.5), 0.0, max_val).astype(np.float32)
        return quantized / max_val

    def simulate(self, radiance: np.ndarray, exp_time: float, analog_gain: float) -> np.ndarray:
        radiance = np.clip(radiance.astype(np.float32), 0.0, None)

        t_pred = float(max(exp_time, 1e-6))
        g_pred = float(dB_to_ratio(float(analog_gain)))

        gauss_std = np.sqrt(self.noise_cfg.gaussian_var) * t_pred
        poisson_scale = max(self.noise_cfg.poisson_scale * t_pred, 1e-8)

        radiance = self._apply_exposure_motion_blur(radiance, t_pred)
        scaled_radiance = radiance * t_pred

        poisson_lambda = np.clip(scaled_radiance / poisson_scale, 0.0, None)
        shot_noise = self.rng.poisson(poisson_lambda).astype(np.float32) * poisson_scale * g_pred
        readout_noise = gauss_std * self.rng.standard_normal(radiance.shape).astype(np.float32) * g_pred
        adc_noise = (self.rng.random(radiance.shape).astype(np.float32) * 2.0 - 1.0) * self.adc_noise_scale

        noisy = np.clip(shot_noise + readout_noise + adc_noise, 0.0, None)
        return self._quantize(noisy)

    def subframes_fusion(self, radiance: np.ndarray, exp_time: float, analog_gain: float) -> np.ndarray:
        return self.simulate(radiance, exp_time=exp_time, analog_gain=analog_gain)


class ImageNoiseMetric:
    def __init__(self, p: float = 0.10, t_lower: float = 0.05, t_upper: float = 0.92):
        self.p = p
        self.t_lower = t_lower
        self.t_upper = t_upper
        self._kernel = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], dtype=np.float32)

    def evaluate(self, img_np: np.ndarray) -> float:
        img = minmax_norm(img_np)
        h, w, c = img.shape
        noise_vals = np.zeros(c, dtype=np.float32)

        for channel_idx in range(c):
            channel = img[:, :, channel_idx]
            gx = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
            grad_mag = np.sqrt(gx ** 2 + gy ** 2)

            grad_sorted = np.sort(grad_mag.reshape(-1))
            threshold_idx = min(max(int(h * w * self.p), 0), grad_sorted.size - 1)
            gth = grad_sorted[threshold_idx]

            homog_mask = grad_mag <= gth
            unsat_mask = (channel >= self.t_lower) & (channel <= self.t_upper)
            mask = homog_mask & unsat_mask

            lap = cv2.filter2D(channel, cv2.CV_32F, self._kernel)
            masked = lap[mask]
            ns = masked.size
            if ns == 0:
                noise_vals[channel_idx] = 0.0
            else:
                noise_vals[channel_idx] = (np.sqrt(np.pi / 2.0) / (6.0 * ns)) * np.sum(np.abs(masked))

        max_noise = float(np.max(noise_vals))
        if max_noise < 1e-8:
            return 0.0
        return float(np.mean(noise_vals / max_noise))


class ImageGradientMetric:
    def __init__(self, lam: float = 1e3, gamma: float = 0.3, num: int = 16):
        self.lam = lam
        self.gamma = gamma
        self.num = num

    def evaluate(self, img_np: np.ndarray) -> float:
        img = minmax_norm(img_np)
        gray = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY)
        h, w = gray.shape

        h_crop = (h // self.num) * self.num
        w_crop = (w // self.num) * self.num
        if h_crop == 0 or w_crop == 0:
            return 0.0
        gray = gray[:h_crop, :w_crop]

        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(gx ** 2 + gy ** 2)

        ng_val = np.log(self.lam * (1.0 - self.gamma) + 1.0)
        mapped = np.where(
            grad_mag >= self.gamma,
            (np.log(self.lam * (grad_mag - self.gamma) + 1.0) + 1e-7) / ng_val,
            0.0,
        )

        h_step, w_step = h_crop // self.num, w_crop // self.num
        mapped_grid = mapped.reshape(self.num, h_step, self.num, w_step)
        block_means = mapped_grid.mean(axis=(1, 3)).reshape(-1)

        max_val = float(block_means.max())
        if max_val < 1e-8:
            return 0.0
        return float(np.mean(block_means / max_val))


class ImageContrastMetric:
    def __init__(self, alpha: float = 10.0, beta: float = 0.2):
        self.alpha = alpha
        self.beta = beta

    def evaluate(self, img_np: np.ndarray) -> float:
        img = minmax_norm(img_np)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)
        contrast_std = float(np.std(magnitude))
        return float(1.0 / (1.0 + np.exp(-self.alpha * (contrast_std - self.beta))))


class ImageEntropyMetric:
    def __init__(self, weight: float = 0.125):
        self.weight = weight

    def evaluate(self, img_np: np.ndarray) -> float:
        img = minmax_norm(img_np)
        gray = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_RGB2GRAY)
        hist, _ = np.histogram(gray.reshape(-1), bins=256, density=True)
        return float(shannon_entropy(hist + 1e-12) * self.weight)


class ImageSemanticMetric:
    def __init__(self):
        self._use_saliency = hasattr(cv2, "saliency") and hasattr(cv2.saliency, "StaticSaliencyFineGrained_create")

    def compute_saliency(self, image: np.ndarray) -> np.ndarray:
        if not self._use_saliency:
            gray = cv2.cvtColor(minmax_norm(image), cv2.COLOR_RGB2GRAY)
            return np.clip(gray * 255.0, 0.0, 255.0).astype(np.uint8)

        saliency_func = cv2.saliency.StaticSaliencyFineGrained_create()
        _, saliency_map = saliency_func.computeSaliency(image.astype(np.float32))
        saliency_u8 = np.clip(minmax_norm(saliency_map) * 255.0, 0.0, 255.0).astype(np.uint8)
        thresh_map = cv2.threshold(saliency_u8, 20, 250, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        return thresh_map

    def evaluate(self, image: np.ndarray) -> float:
        img = minmax_norm(image)
        saliency_map = self.compute_saliency(img)
        saliency_rgb = np.repeat(saliency_map[:, :, np.newaxis], 3, axis=2).astype(np.float32) / 255.0
        image_saliency = saliency_rgb * img
        image_hist, _ = np.histogram(image_saliency.reshape(-1), bins=512, density=True)
        return float(shannon_entropy(image_hist + 1e-12))


class MixedImageMetric:
    def __init__(self, gradient_weight: float = 0.5, entropy_weight: float = 0.5, noise_weight: float = -0.4):
        self.noise_metric = ImageNoiseMetric()
        self.gradient_metric = ImageGradientMetric()
        self.entropy_metric = ImageEntropyMetric()
        self.noise_weight = noise_weight
        self.gradient_weight = gradient_weight
        self.entropy_weight = entropy_weight

    def evaluate(self, img_np: np.ndarray) -> float:
        img = minmax_norm(img_np)
        noise_value = self.noise_metric.evaluate(img)
        gradient_value = self.gradient_metric.evaluate(img)
        entropy_value = self.entropy_metric.evaluate(img)
        return float(
            self.noise_weight * noise_value
            + self.gradient_weight * gradient_value
            + self.entropy_weight * entropy_value
        )


def load_hdr_image(path: str) -> np.ndarray:
    src = Path(path)
    if src.suffix.lower() == ".npy":
        image = np.load(str(src), allow_pickle=False)
    else:
        image = cv2.imread(str(src), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise FileNotFoundError(f"Failed to read HDR image: {path}")

    if image.ndim not in (2, 3):
        raise ValueError(f"Unsupported HDR array shape {image.shape} from: {path}")
    if image.ndim == 2:
        image = np.repeat(image[..., None], 3, axis=2)

    if image.shape[2] == 1:
        image = np.repeat(image, 3, axis=2)
    image = image[..., :3].astype(np.float32)
    return minmax_norm(image)


def inverse_tmo(ldr_image, mu:float=1000.0, l_max: float = 3000.0)->np.ndarray:
    ldr_image = minmax_norm(ldr_image)
    ldr = np.clip(ldr_image, 0, 1)
    hdr = l_max * (((1.0 + mu) ** ldr - 1.0) / mu)
    hdr = minmax_norm(hdr)
    return hdr.astype(np.float32)


# def save_ldr(path: str, image_float01: np.ndarray) -> None:
#     image_u16 = np.clip(
#         minmax_norm(image_float01) * float(np.iinfo(np.uint16).max),
#         0.0,
#         float(np.iinfo(np.uint16).max),
#     ).astype(np.uint16)
#     cv2.imwrite(path, image_u16)

def save_ldr(path: str, image_float01: np.ndarray) -> None:
    # image_u16 = np.clip(
    #     minmax_norm(image_float01) * float(np.iinfo(np.uint16).max),
    #     0.0,
    #     float(np.iinfo(np.uint16).max),
    # ).astype(np.uint16)
    image_u8 = np.clip(minmax_norm(image_float01) * 255, 0, 255).astype(np.uint8)
    cv2.imwrite(path, image_u8)


# def save_ldr(path: str, image_float01: np.ndarray) -> None:
#     # image_u16 = np.clip(
#     #     minmax_norm(image_float01) * 1024.0,
#     #     0.0,
#     #     1023.0,
#     # ).astype(np.uint16)
#     image_u16 = np.clip(image_u16, 0, None).astype(np.uint16)
#     cv2.imwrite(path, image_u16)
