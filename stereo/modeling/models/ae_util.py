import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def exposure_value_equation(ev_vaules, time_limits, gain_limits):
    gain = torch.clamp(ev_vaules / time_limits[1], 1., None)
    expo = ev_vaules / gain
    gain = torch.clamp(gain, gain_limits[0], gain_limits[1])
    expo = torch.clamp(expo, time_limits[0], time_limits[1])
    return expo, gain



class QuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, n=8):
        # LDR image max value
        max_val = 2**n - 1

        # Quantize
        # x_scaled = input * max_val
        # x_clamped = torch.clamp(x_scaled, 0, max_val)
        # x_clamped = torch.round(x_clamped)
        output = torch.clamp(torch.floor(input + 0.5), min=0, max=max_val)

        # Normalized to 0~1
        # output = output / max_val
        output = (output - output.min()) / (output.max() - output.min())
        return output
        # return x_clamped

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None



def electron_to_digital(electrons,  well_capacity=4e4, n_bits=8):
    digital_number = torch.clamp(electrons, 0, well_capacity)  # Clip to the range of 8-bit digital number
    # digital_number = digital_number / well_capacity * (2**n_bits - 1)  # Scale to the range of digital number
    return digital_number


class ImageFormationModel(nn.Module):
    def __init__(self, nbits=8,):
        super(ImageFormationModel, self).__init__()
        self.nbits = nbits

        # self.gaussian_var = torch.tensor(1.0e-3, requires_grad=True)
        # self.poisson_scale = torch.tensor(3.4e-4, requires_grad=True)
        # Keep noise statistics device-aware without treating them as trainable weights.
        self.register_buffer('gaussian_var', torch.tensor(3e-5, dtype=torch.float32))
        self.register_buffer('poisson_scale', torch.tensor(3.3e-4, dtype=torch.float32))
    def forward(self, radiance, t_pred, g_pred):
        # add noise based on exposure value
        # radiance = self.radiance_scale(radiance)
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
        noise_radiance = torch.clamp(noise_radiance, 0.0, None)

        noise_radiance = QuantizeSTE.apply(noise_radiance, self.nbits)
        return noise_radiance


def radiance_scale(radiance,  capacity=1e2):
    mean_val = radiance.mean()
    scale = capacity / mean_val
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
        gain_limits = _cfg_pair(cfgs, 'GAIN_LIMITS', default_gain_limits)
        init_exposure = float(_cfg_get(cfgs, 'INIT_EXPOSURE', sum(time_limits) / 2.0))
        init_gain = float(_cfg_get(cfgs, 'INIT_GAIN', sum(gain_limits) / 2.0))

        self.image_formation_model = ImageFormationModel(nbits=8)
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




def photons_collection(radiance, well_capacity=(2**32 - 1)):
    # Convert radiance to electrons (assuming linear response and a certain quantum efficiency)
    electrion = np.clip(radiance, 0, well_capacity)
    return electrion


def to_image(radiance):
    radiance = (radiance - radiance.min()) / (radiance.max() - radiance.min()) * 255.0
    return radiance.astype(np.uint8)


if __name__ == "__main__":
    # simulator
    image_formation = ImageFormationModel(nbits=8)

    # Example usage
    dataset_root = "/home/lgz/dataset/ADEC/carla/dataset/Experiment1/"
    left_path = dataset_root + "hdr_left/0.hdr"
    right_path = dataset_root + "hdr_right/0.hdr"
    print(left_path, right_path)

    left_hdr = cv2.imread(left_path, cv2.IMREAD_UNCHANGED) 
    right_hdr = cv2.imread(right_path, cv2.IMREAD_UNCHANGED) 
    print(f"input radiance: {left_hdr.min()} to {left_hdr.max()}, {right_hdr.min()} to {right_hdr.max()}")
    # left_hdr = photons_collection(left_hdr, well_capacity=(2**16 - 1))
    # right_hdr = photons_collection(right_hdr, well_capacity=(2**16 - 1))

    # left_hdr = to_image(left_hdr)
    # right_hdr = to_image(right_hdr)
    # cv2.imwrite("left_hdr.png", left_hdr)
    # cv2.imwrite("right_hdr.png", right_hdr) 
    # exit(234) 
    left_hdr = (left_hdr - left_hdr.min()) / (left_hdr.max() - left_hdr.min()) 
    right_hdr = (right_hdr - right_hdr.min()) / (right_hdr.max() - right_hdr.min())


    left_hdr = radiance_scale(left_hdr, capacity=1)
    right_hdr = radiance_scale(right_hdr, capacity=1)

    
    # print(f"radiance 90% percentile: {np.percentile(left_hdr, 90)}, {np.percentile(right_hdr, 90)}")
    # exit(0)

    # Convert to PyTorch tensors
    left_tensor = torch.from_numpy(left_hdr).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
    right_tensor = torch.from_numpy(right_hdr).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

    
    # Example exposure values
    t_pred = torch.tensor([10.0])  # exposure multiplier
    g_pred = torch.tensor([5.0])  # gain multiplier 
    # Simulate noisy, quantized images
    left_noisy = image_formation(left_tensor, t_pred, g_pred)
    right_noisy = image_formation(right_tensor, t_pred, g_pred)
    print("Left noisy image :", left_noisy.shape, left_noisy.min().item(), left_noisy.max().item())
    print("Right noisy image :", right_noisy.shape, right_noisy.min().item(), right_noisy.max().item())
    

    # left_ldr = left_noisy.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    # right_ldr = right_noisy.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
    # left_ldr = (left_ldr - left_ldr.min()) / (left_ldr.max() - left_ldr.min() )* 255.0
    # right_ldr = (right_ldr - right_ldr.min()) / (right_ldr.max() - right_ldr.min()) * 255.0
    # print(f"output LDR: {left_ldr.min()} to {left_ldr.max()}, {right_ldr.min()} to {right_ldr.max()}")
    # cv2.imwrite("left_ldr.png", left_ldr.astype(np.uint8))
    # cv2.imwrite("right_ldr.png", right_ldr.astype(np.uint8)) 
