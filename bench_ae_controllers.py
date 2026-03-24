#!/usr/bin/env python
import json
from pathlib import Path

import torch
import sys

sys.path.insert(0, '/home/lgz/workspace/OpenStereo')

from stereo.modeling.models.gradnet.grad_ae import GradientExposureController as GradAE_Controller
from stereo.modeling.models.avenet.average_ae import StatBasedAutoExposure as AveAE_Controller
from stereo.modeling.models.neuralnet.submodules import NeuralExposureController as NeuralAE_Controller
from stereo.modeling.models.stereonet.stereo_ae_2 import StereoExposureController as StereoAE_Controller
from stereo.modeling.models.comp_util import build_ae_controller_inputs, profile_exposure_controller_cost

RES_1K = (1, 3, 544, 960)
RES_2K = (1, 3, 1088, 1920)
WARMUP = 10
ITERS = 20


class StereoControllerWrapper(torch.nn.Module):
    """Wrapper for StereoExposureController to make it callable with dict input."""
    def __init__(self, stereo_controller):
        super().__init__()
        self.stereo_controller = stereo_controller

    def forward(self, inputs):
        if isinstance(inputs, dict):
            left = inputs['image_left']
            right = inputs['image_right']
            # Generate dummy exposure values
            batch_size = left.shape[0]
            exp_left = torch.ones(batch_size, device=left.device)
            exp_right = torch.ones(batch_size, device=right.device)
            alpha_left = torch.ones(batch_size, device=left.device)
            alpha_right = torch.ones(batch_size, device=right.device)
            return self.stereo_controller(left, right, exp_left, exp_right, alpha_left, alpha_right)
        else:
            return self.stereo_controller(inputs)


def profile_gradient_ae(shape, warmup, iters):
    """Profile GradAE controller."""
    controller = GradAE_Controller(target_grad=0.12, min_exposure=5.0, max_exposure=280.0)
    img_input = torch.rand(shape).to('cuda')
    single_input = img_input
    return profile_exposure_controller_cost(controller, single_input, warmup=warmup, iters=iters)


def profile_average_ae(shape, warmup, iters):
    """Profile AveAE controller."""
    controller = AveAE_Controller(target_mean=0.5, momentum=0.5)
    img_input = torch.rand(shape).to('cuda')
    current_ev = torch.ones(shape[0], device='cuda')
    # Create a wrapper to handle the two-input case
    class AveAEWrapper(torch.nn.Module):
        def __init__(self, ctrl):
            super().__init__()
            self.ctrl = ctrl
        
        def forward(self, img):
            return self.ctrl(img, current_ev)
    
    wrapper = AveAEWrapper(controller)
    return profile_exposure_controller_cost(wrapper, img_input, warmup=warmup, iters=iters)


def profile_neural_ae(shape, warmup, iters):
    """Profile NeuralAE controller."""
    controller = NeuralAE_Controller()
    img_input = torch.rand(shape).to('cuda')
    return profile_exposure_controller_cost(controller, img_input, warmup=warmup, iters=iters)


def profile_stereo_ae(shape, warmup, iters):
    """Profile StereoAE controller."""
    controller = StereoAE_Controller()
    # Create inputs (left + right)
    inputs = {
        'image_left': torch.rand(shape).to('cuda'),
        'image_right': torch.rand(shape).to('cuda'),
    }
    wrapper = StereoControllerWrapper(controller)
    return profile_exposure_controller_cost(wrapper, inputs, warmup=warmup, iters=iters)


rows = []

# Test each controller at 1K and 2K resolutions
controllers = [
    ("GradAE", profile_gradient_ae),
    ("AveAE", profile_average_ae),
    ("NeuralAE", profile_neural_ae),
    ("StereoAE", profile_stereo_ae),
]

for name, profile_fn in controllers:
    row = {"controller": name}
    try:
        r1 = profile_fn(RES_1K, WARMUP, ITERS)
        row.update({
            "params_m": r1["params_m"],
            "flops_1k_g": r1["flops_g"],
            "latency_1k_ms": r1["latency_ms"],
        })
    except Exception as e:
        row.update({"err_1k": repr(e)})

    try:
        r2 = profile_fn(RES_2K, WARMUP, ITERS)
        row.update({
            "flops_2k_g": r2["flops_g"],
            "latency_2k_ms": r2["latency_ms"],
        })
    except Exception as e:
        row.update({"err_2k": repr(e)})

    rows.append(row)

print(json.dumps(rows, indent=2, ensure_ascii=False))
