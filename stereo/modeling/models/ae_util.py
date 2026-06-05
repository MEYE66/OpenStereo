import sys
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.nn as nn



def _find_repo_root(start_path: Path) -> Path:
    for parent in [start_path] + list(start_path.parents):
        if (parent / 'stereo').is_dir():
            return parent
    raise RuntimeError('Could not locate repository root containing the stereo package.')


repo_root = _find_repo_root(Path(__file__).resolve())
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from stereo.modeling.models.exposure_motion_psf_torch_refactor import apply_exposure_motion_psf


def exposure_value_equation(ev_vaules, time_limits, gain_limits):
    """Prefer longer exposure and only increase gain when exposure hits the upper bound."""
    gain_limits_ratio = dB_to_ratio(gain_limits)
    gain_ratio = torch.clamp(ev_vaules / time_limits[1], min=gain_limits_ratio[0])
    expo = ev_vaules / gain_ratio
    gain_ratio = torch.clamp(gain_ratio, gain_limits_ratio[0], gain_limits_ratio[1])
    expo = torch.clamp(expo, time_limits[0], time_limits[1])
    gain = torch.clamp(ratio_to_dB(gain_ratio), gain_limits[0], gain_limits[1])
    return expo, gain


class QuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, nbits=10):
        # LDR image max value
        max_val = 2 ** nbits - 1
        # Quantize
        output = torch.clamp(torch.floor(input + 0.5), min=0, max=max_val)
        # x_dequantized = x_quantized / max_val
        output = output / max_val
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def dB_to_ratio(dB):
    return 10 ** (dB / 20.0)


def ratio_to_dB(ratio):
    if torch.is_tensor(ratio):
        return 20.0 * torch.log10(torch.clamp(ratio, min=1e-8))
    return 20.0 * np.log10(max(float(ratio), 1e-8))


class ImageFormationModel(nn.Module):
    def __init__(self, nbits=10, motion_blur=True, motion_angle_deg=None, motion_blur_params=None):
        super(ImageFormationModel, self).__init__()
        self.nbits = nbits
        self.motion_blur = motion_blur
        self.register_buffer('gaussian_var', torch.tensor(0.8, dtype=torch.float32))
        self.register_buffer('poisson_scale', torch.tensor(3.3e-4, dtype=torch.float32))
        self.register_buffer('adc_noise_scale', torch.tensor(1/nbits, dtype=torch.float32))
        self.motion_blur_params = {
            'velocity_scale': 0.8,
            'min_length': 1,
            'max_length': 15,
            'canvas_size': 31,
            'line_sigma': 0.55,
            'edge_softness': 0.75,
            'eps': 1e-8,
        }
        if motion_blur_params is not None:
            self.motion_blur_params.update(motion_blur_params)
        self.motion_angle_deg = 10.0 if motion_angle_deg is None else float(motion_angle_deg)
        self.motion_psf = None
        self.motion_psf_info = None
        self.quantization = QuantizeSTE(nbits=nbits)


    def _apply_exposure_motion_blur(self, radiance, exposure_time_ms):
        if not self.motion_blur:
            return radiance

        angle = exposure_time_ms.new_full(exposure_time_ms.shape, self.motion_angle_deg)
        blurred_radiance, self.motion_psf, motion_length = apply_exposure_motion_psf(
            radiance,
            exposure_time=exposure_time_ms,
            angle=angle,
            **self.motion_blur_params,
        )
        self.motion_psf_info = {
            'exposure_time_ms': exposure_time_ms,
            'angle_deg': angle,
            'kernel_size': self.motion_psf.shape[-1],
            'length_px': motion_length,
        }
        return blurred_radiance


    def forward(self, radiance, t_pred, g_pred):
        # add noise based on exposure value
        # radiance = self.radiance_scale(radiance)
        # exposure_time = t_pred.reshape(-1)
        t_pred = t_pred.view(-1, 1, 1, 1)
        g_pred = dB_to_ratio(g_pred.view(-1, 1, 1, 1))

        gauss_std = torch.sqrt(self.gaussian_var) * t_pred
        poisson_scale = torch.clamp(self.poisson_scale * t_pred, min=1e-8)

        radiance = self._apply_exposure_motion_blur(radiance, t_pred.reshape(-1))
        radiance = radiance * t_pred

        # Shot noise
        shot_noise = torch.poisson(radiance / poisson_scale) * poisson_scale * g_pred
        # Readout noise
        readout_noise = gauss_std * torch.randn_like(radiance) * g_pred
        # ADC quantization noise: U(-sigma, +sigma), sigma = 1 / nbits
        # sigma = 1.0 / float(self.nbits)
        # adc_noise = (torch.rand_like(radiance) * 2.0 - 1.0) * sigma
        adc_noise = (torch.rand_like(radiance) * 2.0 - 1.0) * self.adc_noise_scale 
        noise_radiance = shot_noise + readout_noise + adc_noise
        noise_radiance = torch.clamp(noise_radiance, 0.0, None)

        # print(f"noise radicne range: {noise_radiance.min().item()} to {noise_radiance.max().item()}")
        noise_radiance = self.quantization.apply(noise_radiance, self.nbits)
        # noise_radiance = noise_radiance / noise_radiance.max()
        return noise_radiance


def radiance_scale(radiance,  capacity=12):
    # radiance = radiance / radiance.max() 
    radiance = (radiance - radiance.min()) / (radiance.max() - radiance.min() + 1e-8) 
    mean_val = radiance.mean()
    scale = capacity / (mean_val + 1e-8)
    radiance = radiance * scale # scale to capacity
    return radiance



def _cfg_get(cfgs, key, default):
    if hasattr(cfgs, 'get'):
        return cfgs.get(key, default)
    return getattr(cfgs, key, default)


def _cfg_pair(cfgs, key, default):
    value = _cfg_get(cfgs, key, default)
    return [float(value[0]), float(value[1])]



class ExposureControlMixin:
    def _init_exposure_control(self, cfgs, default_time_limits=(5.0, 20.0), default_gain_limits=(1.0, 20.0)):
        time_limits = _cfg_pair(cfgs, 'TIME_LIMITS', default_time_limits)
        # gain limits and init gain are interpreted in dB.
        gain_limits = _cfg_pair(cfgs, 'GAIN_LIMITS', default_gain_limits)
        init_exposure = float(_cfg_get(cfgs, 'INIT_EXPOSURE', sum(time_limits) / 2.0))
        init_gain = float(_cfg_get(cfgs, 'INIT_GAIN', sum(gain_limits) / 2.0))

        self.image_formation_model = ImageFormationModel(nbits=10)
        self.exposure_controller = self._build_exposure_controller(
            cfgs=cfgs,
            time_limits=time_limits,
            gain_limits=gain_limits,
        )
        if not isinstance(self.exposure_controller, nn.Module):
            raise TypeError('Exposure controller must be an nn.Module instance.')

        self.register_buffer('time_limits', torch.tensor(time_limits, dtype=torch.float32))
        self.register_buffer('gain_limits', torch.tensor(gain_limits, dtype=torch.float32))
        self.register_buffer('init_exp', torch.tensor([init_exposure], dtype=torch.float32))
        self.register_buffer('init_gain', torch.tensor([init_gain], dtype=torch.float32))

    @staticmethod
    def _expand_scalar_buffer(buffer, batch_size):
        return buffer.view(1).expand(batch_size)

    def _build_seed_images(self, radiance_left, radiance_right):
        batch_size = radiance_left.shape[0]
        init_exp = self._expand_scalar_buffer(self.init_exp, batch_size)
        init_gain = self._expand_scalar_buffer(self.init_gain, batch_size)

        img_left = self.image_formation_model(radiance_left, init_exp, init_gain)
        img_right = self.image_formation_model(radiance_right, init_exp, init_gain)
        return img_left, img_right

    def _apply_exposure(self, radiance_left, radiance_right, exposure):
        expo_update, gain_update = exposure_value_equation(exposure, self.time_limits, self.gain_limits)
        img_left = self.image_formation_model(radiance_left, expo_update, gain_update)
        img_right = self.image_formation_model(radiance_right, expo_update, gain_update)
        return img_left, img_right

    def _build_state(self, img_left, img_right):
        return (img_left + img_right) / 2

    def _build_exposure_controller(self, cfgs, time_limits, gain_limits):
        raise NotImplementedError('Subclasses must implement _build_exposure_controller.')


def to_image(radiance):
    radiance = (radiance - radiance.min()) / (radiance.max() - radiance.min()) * 255.0
    return radiance.astype(np.uint8)


def apply_gtm(img, eps=1e-6, param=0.3):
    img = (img - np.min(img)) / (np.max(img) - np.min(img))
    Lw_ave = np.exp(np.mean(np.log(eps + img)))
    Lm = (param / Lw_ave) * img
    Lm_max = np.max(Lm)
    out = (Lm * (1 + (Lm / (Lm_max ** 2)))) / (1 + Lm)
    out =np.clip(out, 0.0, 1.0)
    return out

if __name__ == "__main__":
    # simulator
    nbits = 10
    image_formation = ImageFormationModel(nbits=nbits, motion_blur=False)
    # Example usage
    dataset_root = "/home/lgz/dataset/ADEC/carla_600x800/train/Experiment10/"
    left_path = dataset_root + "hdr_left/12.hdr"
    right_path = dataset_root + "hdr_right/12.hdr"
    print(left_path, right_path)

    left_hdr = cv2.imread(left_path, cv2.IMREAD_UNCHANGED) 
    right_hdr = cv2.imread(right_path, cv2.IMREAD_UNCHANGED) 

    # left_hdr = (left_hdr - left_hdr.min()) / (left_hdr.max() - left_hdr.min()) 
    # right_hdr = (right_hdr - right_hdr.min()) / (right_hdr.max() - right_hdr.min())
    left_hdr = radiance_scale(left_hdr, capacity=12)
    right_hdr = radiance_scale(right_hdr, capacity=12)

    # Convert to PyTorch tensors
    left_tensor = torch.from_numpy(left_hdr).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    right_tensor = torch.from_numpy(right_hdr).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

    print(f"{left_tensor.min()}, {left_tensor.max()}")
    print(f"{right_tensor.min()}, {right_tensor.max()}")
    # print(f"range after ITMO: {left_hdr.min().item()} to {left_hdr.max().item()}, {right_hdr.min().item()} to {right_hdr.max().item()}")

    # Example exposure values
    left_t, right_t = torch.tensor([10.0]), torch.tensor([2.0])
    left_g, right_g = torch.tensor([1.0]), torch.tensor([20.0])

    # Simulate noisy, quantized images
    left_noisy = image_formation(left_tensor, left_t, left_g)
    right_noisy = image_formation(right_tensor, right_t, right_g)

    print(f"noisy image range: {left_noisy.min().item()} to {left_noisy.max().item()}, {right_noisy.min().item()} to {right_noisy.max().item()}")

    left_ldr = left_noisy.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    right_ldr = right_noisy.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

    left_ldr = apply_gtm(left_ldr)
    

    cv2.imwrite(f"./{nbits}_left_ldr_sece.png", to_image(left_ldr))
    cv2.imwrite(f"./{nbits}_right_ldr_sece.png", to_image(right_ldr)) 
