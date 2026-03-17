import sys
import os
from pathlib import Path
import torch
import torch.nn as nn
from types import SimpleNamespace


repo_root = Path(__file__).resolve().parents[4]
# print(f"Adding repo root to sys.path: {repo_root}")
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
try:
    # Prefer absolute imports so this file can be run directly.
    from stereo.modeling.models.neuralnet.submodules import NeuralExposureController,FloorDivSTE
    from stereo.modeling.models.ae_util import ImageFormationModel
    from stereo.modeling.models.gwcnet.gwcnet import GwcNet as BaseGwcNet
    from stereo.modeling.models.psmnet.psmnet import PSMNet as BasePSMNet
except ModuleNotFoundError:
    # Fallback for environments where package root is preconfigured.
    from .submodules import NeuralExposureController, FloorDivSTE
    from ..ae_util import ImageFormationModel
    from ..gwcnet.gwcnet import GwcNet as BaseGwcNet
    from ..psmnet.psmnet import PSMNet as BasePSMNet


def exposure_value_equation(ev_vaules, time_limits, gain_limits):
    gain = torch.clamp(ev_vaules / time_limits[1], 1., None)
    expo = ev_vaules / gain
    gain = torch.clamp(gain, gain_limits[0], gain_limits[1])
    expo = torch.clamp(expo, time_limits[0], time_limits[1])
    return expo, gain


class NeuralAEGwcNet(BaseGwcNet):
    def __init__(self, cfgs, time_limits=[1., 20.], gain_limits=[1., 14.], iter=3, mu=0.8):
        super(NeuralAEGwcNet, self).__init__(cfgs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.image_formation_model = ImageFormationModel().to(self.device)
        self.exposure_controller = NeuralExposureController().to(self.device)
        self.floor_div = FloorDivSTE()
        
        
        self.time_limits = torch.tensor(time_limits, requires_grad=True, device=self.device)
        self.gain_limits = torch.tensor(gain_limits, requires_grad=True, device=self.device)

        self.init_exp = torch.tensor((time_limits[0] + time_limits[1]) / 2, requires_grad=True, device=self.device)
        self.init_gain = torch.tensor((gain_limits[0] + gain_limits[1]) / 2, requires_grad=True, device=self.device)
        
        self.iters = iter
        self.mu = mu
        
    def update_function(self, e_t, u_t):
        log_e_t = self.mu * torch.log(e_t) + (1 - self.mu) * torch.log(e_t * u_t)
        new_e_t = torch.exp(log_e_t)
        gain = torch.max(self.gain_limits[0], torch.min(self.gain_limits[1], self.floor_div.apply(new_e_t, self.time_limits[1])))
        exp_time = torch.max(self.time_limits[0], torch.min(self.time_limits[1], new_e_t / gain))
        return exp_time, gain    
    

    def forward(self, inputs):
        radiance_left, radiance_right = inputs['left'], inputs['right']
        
        
        img_left_updated = self.image_formation_model(radiance_left, self.init_exp, self.init_gain)
        img_right_updated = self.image_formation_model(radiance_right, self.init_exp, self.init_gain)
        expo_update = self.init_exp
        for _ in range(self.iters):
            values = self.exposure_controller((img_left_updated + img_right_updated) / 2)
            # expo_update, gain_update = self.update_function(expo_values, self.time_limits, self.gain_limits)
            expo_update, gain_update = self.update_function(expo_update, values) # exp_time[B, 1], gain [B, 1]
            

            img_left_updated = self.image_formation_model(radiance_left, expo_update, gain_update)
            img_right_updated = self.image_formation_model(radiance_right, expo_update, gain_update)

        disp_pred = super(NeuralAEGwcNet, self).forward({'left': img_left_updated, 'right': img_right_updated})
        return disp_pred


class NeuralAEPSMNet(BasePSMNet):
    def __init__(self, cfgs, time_limits=[1., 20.], gain_limits=[1., 14.], iter=3, mu=0.8):
        super(NeuralAEPSMNet, self).__init__(cfgs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.image_formation_model = ImageFormationModel().to(self.device)
        self.exposure_controller = NeuralExposureController().to(self.device)
        self.floor_div = FloorDivSTE()

        self.time_limits = torch.tensor(time_limits, requires_grad=True, device=self.device)
        self.gain_limits = torch.tensor(gain_limits, requires_grad=True, device=self.device)

        self.init_exp = torch.tensor((time_limits[0] + time_limits[1]) / 2, requires_grad=True, device=self.device)
        self.init_gain = torch.tensor((gain_limits[0] + gain_limits[1]) / 2, requires_grad=True, device=self.device)

        self.iters = iter
        self.mu = mu

    def update_function(self, e_t, u_t):
        log_e_t = self.mu * torch.log(e_t) + (1 - self.mu) * torch.log(e_t * u_t)
        new_e_t = torch.exp(log_e_t)
        gain = torch.max(self.gain_limits[0], torch.min(self.gain_limits[1], self.floor_div.apply(new_e_t, self.time_limits[1])))
        exp_time = torch.max(self.time_limits[0], torch.min(self.time_limits[1], new_e_t / gain))
        return exp_time, gain

    def forward(self, inputs):
        radiance_left, radiance_right = inputs['left'], inputs['right']

        img_left_updated = self.image_formation_model(radiance_left, self.init_exp, self.init_gain)
        img_right_updated = self.image_formation_model(radiance_right, self.init_exp, self.init_gain)
        expo_update = self.init_exp
        for _ in range(self.iters):
            values = self.exposure_controller((img_left_updated + img_right_updated) / 2)
            expo_update, gain_update = self.update_function(expo_update, values)

            img_left_updated = self.image_formation_model(radiance_left, expo_update, gain_update)
            img_right_updated = self.image_formation_model(radiance_right, expo_update, gain_update)

        disp_pred = super(NeuralAEPSMNet, self).forward({'left': img_left_updated, 'right': img_right_updated})
        return disp_pred


if __name__ == '__main__':
    cfgs = SimpleNamespace(
        MAX_DISP=int(192),
        USE_CONCAT_VOLUME=bool(False),
        CONCAT_CHANNELS=int(8),
        DOWNSAMPLE=int(4),
        NUM_GROUPS=int(8),
    )
    model = NeuralAEGwcNet(cfgs=cfgs)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    radiance_left = torch.rand(1, 3, 256, 512, device=device)
    radiance_right = torch.rand(1, 3, 256, 512, device=device)

    with torch.no_grad():
        disp_pred = model({'left': radiance_left, 'right': radiance_right})
    print(disp_pred)
