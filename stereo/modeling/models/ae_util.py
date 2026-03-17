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
    def forward(ctx, input, n=12):
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



def electron_to_digital(electrons,  well_capacity=4e4, n_bits=12):
    digital_number = torch.clamp(electrons, 0, well_capacity)  # Clip to the range of 12-bit digital number
    # digital_number = digital_number / well_capacity * (2**n_bits - 1)  # Scale to the range of digital number
    return digital_number


class ImageFormationModel(nn.Module):
    def __init__(self, nbits=12, capacity=1e2):
        super(ImageFormationModel, self).__init__()
        self.nbits = nbits
        self.capacity = capacity

        # self.gaussian_var = torch.tensor(1.0e-3, requires_grad=True)
        # self.poisson_scale = torch.tensor(3.4e-4, requires_grad=True)
        # [5., 1.] default values for gaussian and possion
        self.gaussian_var = torch.tensor(5., requires_grad=True)
        self.poisson_scale = torch.tensor(1., requires_grad=True)

    def radiance_scale(self, radiance):
        mean_val = radiance.mean()
        scale = self.capacity / mean_val
        radiance = radiance * scale # scale to capacity
        return radiance


    def forward(self, radiance, t_pred, g_pred):
        # add noise based on exposure value
        # adjust exposure
        # print(radiance.shape, t_pred.shape, g_pred.shape)
        radiance = self.radiance_scale(radiance)
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