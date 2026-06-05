import torch
import torch.nn as nn

from stereo.modeling.models.aegwcnet.ae_util import ImageFormationModel
from stereo.modeling.models.ae_util import dB_to_ratio, exposure_value_equation as shared_exposure_value_equation
from stereo.modeling.models.gwcnet.gwcnet import GwcNet as BaseGwcNet


def exposure_value_equation(ev_vaules, time_limits, gain_limits):
    return shared_exposure_value_equation(ev_vaules, time_limits, gain_limits)


def _to_unit_range(img):
    b = img.shape[0]
    flat = img.view(b, -1)
    min_val = flat.min(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
    max_val = flat.max(dim=1, keepdim=True)[0].view(b, 1, 1, 1)
    return (img - min_val) / (max_val - min_val + 1e-6)


class NeuralExposureController(nn.Module):
    def __init__(self, min_exposure=0.5, max_exposure=2.0):
        super().__init__()
        self.min_exposure = min_exposure
        self.max_exposure = max_exposure

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(32, 1)

    def forward(self, image):
        x01 = _to_unit_range(image)
        feat = self.encoder(x01).flatten(1)
        alpha = torch.sigmoid(self.head(feat)).squeeze(1)
        exp_val = self.min_exposure + (self.max_exposure - self.min_exposure) * alpha
        return exp_val


class NeuralAEStereoNet(nn.Module):
    def __init__(self, max_disp=192, time_limits=[1., 20.], gain_limits=[1., 14.]):
        super(NeuralAEStereoNet, self).__init__()
        self.image_formation_model = ImageFormationModel().cuda()
        self.exposure_controller = NeuralExposureController().cuda()
        self.disp_model = BaseGwcNet(max_disp=max_disp).cuda()

        self.time_limits = torch.tensor(time_limits, requires_grad=True).cuda()
        self.gain_limits = torch.tensor(gain_limits, requires_grad=True).cuda()

        self.init_exp = torch.tensor((time_limits[0] + time_limits[1]) / 2, requires_grad=True).cuda()
        self.init_gain = torch.tensor((gain_limits[0] + gain_limits[1]) / 2, requires_grad=True).cuda()

    def forward(self, radiance_left, radiance_right):
        init_gain_ratio = dB_to_ratio(self.init_gain)
        img_left = self.image_formation_model(radiance_left, self.init_exp, init_gain_ratio)
        img_right = self.image_formation_model(radiance_right, self.init_exp, init_gain_ratio)

        expo_values = self.exposure_controller((img_left + img_right) / 2)
        expo_update, gain_update = exposure_value_equation(expo_values, self.time_limits, self.gain_limits)
        gain_update_ratio = dB_to_ratio(gain_update)

        img_left_updated = self.image_formation_model(radiance_left, expo_update, gain_update_ratio)
        img_right_updated = self.image_formation_model(radiance_right, expo_update, gain_update_ratio)

        disp_pred = self.disp_model(img_left_updated, img_right_updated)
        return disp_pred


if __name__ == '__main__':
    cuda_available = torch.cuda.is_available()
    model = NeuralAEStereoNet()
    radiance_left = torch.rand(1, 3, 256, 512)
    radiance_right = torch.rand(1, 3, 256, 512)

    if cuda_available:
        model = model.cuda()
        radiance_left = radiance_left.cuda()
        radiance_right = radiance_right.cuda()

    disp_pred = model(radiance_left, radiance_right)
    print(disp_pred)
