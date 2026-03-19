import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace


repo_root = Path(__file__).resolve().parents[4]
# print(f"Adding repo root to sys.path: {repo_root}")
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
try:
    # Prefer absolute imports so this file can be run directly.
    from stereo.modeling.models.ae_util import ImageFormationModel, exposure_value_equation
    from stereo.modeling.models.gwcnet.gwcnet import GwcNet as BaseGwcNet
    from stereo.modeling.models.psmnet.psmnet import PSMNet as BasePSMNet
    from stereo.modeling.models.gwcnet.gwcnet_sequence import GwcSequenceNet
except ModuleNotFoundError:
    # Fallback for environments where package root is preconfigured.
    from ..ae_util import ImageFormationModel, exposure_value_equation
    from ..gwcnet.gwcnet import GwcNet as BaseGwcNet
    from ..gwcnet.gwcnet_sequence import GwcSequenceNet
    from ..psmnet.psmnet import PSMNet as BasePSMNet


def _grad_score(gray):
    kernel_x = gray.new_tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]).view(1, 1, 3, 3)
    kernel_y = gray.new_tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]).view(1, 1, 3, 3)
    gray = F.pad(gray, (1, 1, 1, 1), mode='reflect')
    gx = F.conv2d(gray, kernel_x)
    gy = F.conv2d(gray, kernel_y)
    return torch.sqrt(gx * gx + gy * gy + 1e-12).mean(dim=(1, 2, 3))


class GradientExposureController(nn.Module):
    def __init__(self, target_grad=0.12, min_exposure=0.5, max_exposure=2.0):
        super().__init__()
        self.target_grad = target_grad
        self.min_exposure = min_exposure
        self.max_exposure = max_exposure

    def forward(self, image):
        gray = image.mean(dim=1, keepdim=True)
        grad_strength = _grad_score(gray)
        exp_val = self.target_grad / (grad_strength + 1e-6)
        return torch.clamp(exp_val, self.min_exposure, self.max_exposure)


class GradientAEGwcNet(BaseGwcNet):
    def __init__(self, cfgs, time_limits=[1., 20.], gain_limits=[1., 14.]):
        super(GradientAEGwcNet, self).__init__(cfgs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.image_formation_model = ImageFormationModel().to(self.device)
        self.exposure_controller = GradientExposureController().to(self.device)

        self.time_limits = torch.tensor(time_limits, requires_grad=True, device=self.device)
        self.gain_limits = torch.tensor(gain_limits, requires_grad=True, device=self.device)

        self.init_exp = torch.tensor((time_limits[0] + time_limits[1]) / 2, requires_grad=True, device=self.device)
        self.init_gain = torch.tensor((gain_limits[0] + gain_limits[1]) / 2, requires_grad=True, device=self.device)


    def forward(self, inputs):
        radiance_left, radiance_right = inputs['left'], inputs['right']

        img_left = self.image_formation_model(radiance_left, self.init_exp, self.init_gain)
        img_right = self.image_formation_model(radiance_right, self.init_exp, self.init_gain)

        expo_values = self.exposure_controller((img_left + img_right) / 2)
        expo_update, gain_update = exposure_value_equation(expo_values, self.time_limits, self.gain_limits)

        img_left_updated = self.image_formation_model(radiance_left, expo_update, gain_update)
        img_right_updated = self.image_formation_model(radiance_right, expo_update, gain_update)

        disp_pred = super(GradientAEGwcNet, self).forward({'left': img_left_updated, 'right': img_right_updated})
        return disp_pred


class GardientAEPSMNet(BasePSMNet):
    def __init__(self, cfgs, time_limits=[1., 20.], gain_limits=[1., 14.]):
        super(GardientAEPSMNet, self).__init__(cfgs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.image_formation_model = ImageFormationModel().to(self.device)
        self.exposure_controller = GradientExposureController().to(self.device)

        self.time_limits = torch.tensor(time_limits, requires_grad=True, device=self.device)
        self.gain_limits = torch.tensor(gain_limits, requires_grad=True, device=self.device)

        self.init_exp = torch.tensor((time_limits[0] + time_limits[1]) / 2, requires_grad=True, device=self.device)
        self.init_gain = torch.tensor((gain_limits[0] + gain_limits[1]) / 2, requires_grad=True, device=self.device)

    def forward(self, inputs):
        radiance_left, radiance_right = inputs['left'], inputs['right']

        img_left = self.image_formation_model(radiance_left, self.init_exp, self.init_gain)
        img_right = self.image_formation_model(radiance_right, self.init_exp, self.init_gain)

        expo_values = self.exposure_controller((img_left + img_right) / 2)
        expo_update, gain_update = exposure_value_equation(expo_values, self.time_limits, self.gain_limits)

        img_left_updated = self.image_formation_model(radiance_left, expo_update, gain_update)
        img_right_updated = self.image_formation_model(radiance_right, expo_update, gain_update)

        disp_pred = super(GardientAEPSMNet, self).forward({'left': img_left_updated, 'right': img_right_updated})
        return disp_pred





class GradientAEGwcSequenceNet(GwcSequenceNet):
    def __init__(self, cfgs, time_limits=[1., 20.], gain_limits=[1., 14.]):
        super(GradientAEGwcSequenceNet, self).__init__(cfgs)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.image_formation_model = ImageFormationModel().to(self.device)
        self.exposure_controller = GradientExposureController().to(self.device)

        self.time_limits = torch.tensor(time_limits, requires_grad=True, device=self.device)
        self.gain_limits = torch.tensor(gain_limits, requires_grad=True, device=self.device)

        self.init_exp = torch.tensor((time_limits[0] + time_limits[1]) / 2, requires_grad=True, device=self.device)
        self.init_gain = torch.tensor((gain_limits[0] + gain_limits[1]) / 2, requires_grad=True, device=self.device)

    def forward(self, inputs):
        curr_radiance_left, curr_radiance_right = inputs['left_1'], inputs['right_1']
        next_radiance_left, next_radiance_right = inputs['left_2'], inputs['right_2']

        curr_img_left = self.image_formation_model(curr_radiance_left, self.init_exp, self.init_gain)
        next_img_left = self.image_formation_model(next_radiance_left, self.init_exp, self.init_gain)
        expo_values = self.exposure_controller((curr_img_left + next_img_left) / 2)
        expo_update, gain_update = exposure_value_equation(expo_values, self.time_limits, self.gain_limits)

        curr_left_updated, curr_right_updated = self.image_formation_model(curr_radiance_left, expo_update, gain_update), self.image_formation_model(curr_radiance_right, expo_update, gain_update)
        next_left_updated, next_right_updated = self.image_formation_model(next_radiance_left, expo_update, gain_update), self.image_formation_model(next_radiance_right, expo_update, gain_update)

        disp_pred = super(GradientAEGwcSequenceNet, self).forward({'left_1': curr_left_updated, 'right_1': curr_right_updated, 'left_2': next_left_updated, 'right_2': next_right_updated})
        return disp_pred


def stereo_test():
    cfgs = SimpleNamespace(
        MAX_DISP=int(192),
        USE_CONCAT_VOLUME=bool(False),
        CONCAT_CHANNELS=int(8),
        DOWNSAMPLE=int(4),
        NUM_GROUPS=int(8),
    )
    model = GradientAEGwcNet(cfgs=cfgs)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    radiance_left = torch.rand(1, 3, 256, 512, device=device)
    radiance_right = torch.rand(1, 3, 256, 512, device=device)

    with torch.no_grad():
        disp_pred = model({'left': radiance_left, 'right': radiance_right})
    print(disp_pred)


def sequence_test():
    cfgs = SimpleNamespace(
        MAX_DISP=int(192),
        USE_CONCAT_VOLUME=bool(False),
        CONCAT_CHANNELS=int(8),
        DOWNSAMPLE=int(4),
        NUM_GROUPS=int(8),
    )
    model = GradientAEGwcSequenceNet(cfgs=cfgs)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    radiance_left = torch.rand(1, 3, 256, 512, device=device)
    radiance_right = torch.rand(1, 3, 256, 512, device=device)

    # prev_inputs = {'left': inputs['left_1'], 'right': inputs['right_1']}
    with torch.no_grad():
        disp_pred = model({'left_1': radiance_left, 'right_1': radiance_right,'left_2': radiance_left, 'right_2': radiance_right})
    print(disp_pred)



if __name__ == '__main__':
    # stereo_test()
    sequence_test()



