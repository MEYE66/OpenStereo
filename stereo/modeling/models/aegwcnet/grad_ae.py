import sys

from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F





repo_root = Path(__file__).resolve().parents[4]
# print(f"Adding repo root to sys.path: {repo_root}")
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from stereo.modeling.models.aegwcnet.ae_util import ImageFormationModel,exposure_value_equation
from stereo.modeling.models.gwcnet.gwcnet import GwcNet as BaseGwcNet




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


class GradientAEStereoNet(nn.Module):
    def __init__(self, max_disp=192, time_limits=[1., 20.], gain_limits=[1., 14.]):
        super(GradientAEStereoNet, self).__init__()
        self.image_formation_model = ImageFormationModel().cuda()
        self.exposure_controller = GradientExposureController().cuda()
        self.disp_model = BaseGwcNet(max_disp=max_disp).cuda()

        self.time_limits = torch.tensor(time_limits, requires_grad=True).cuda()
        self.gain_limits = torch.tensor(gain_limits, requires_grad=True).cuda()

        self.init_exp = torch.tensor((time_limits[0] + time_limits[1]) / 2, requires_grad=True).cuda()
        self.init_gain = torch.tensor((gain_limits[0] + gain_limits[1]) / 2, requires_grad=True).cuda()

    def forward(self, radiance_left, radiance_right):
        img_left = self.image_formation_model(radiance_left, self.init_exp, self.init_gain)
        img_right = self.image_formation_model(radiance_right, self.init_exp, self.init_gain)

        expo_values = self.exposure_controller((img_left + img_right) / 2)
        expo_update, gain_update = exposure_value_equation(expo_values, self.time_limits, self.gain_limits)

        img_left_updated = self.image_formation_model(radiance_left, expo_update, gain_update)
        img_right_updated = self.image_formation_model(radiance_right, expo_update, gain_update)

        disp_pred = self.disp_model(img_left_updated, img_right_updated)
        return disp_pred


if __name__ == '__main__':
    cuda_available = torch.cuda.is_available()
    model = GradientAEStereoNet()
    radiance_left = torch.rand(1, 3, 256, 512)
    radiance_right = torch.rand(1, 3, 256, 512)

    if cuda_available:
        model = model.cuda()
        radiance_left = radiance_left.cuda()
        radiance_right = radiance_right.cuda()

    disp_pred = model(radiance_left, radiance_right)
    print(disp_pred)
