import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


repo_root = Path(__file__).resolve().parents[4]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from stereo.modeling.models.ae_util import _cfg_get, ExposureControlMixin
    from stereo.modeling.models.gwcnet.gwcnet import GwcNet as BaseGwcNet
    from stereo.modeling.models.psmnet.psmnet import PSMNet as BasePSMNet
except ModuleNotFoundError:
    from ..ae_util import _cfg_get, ExposureControlMixin
    from ..gwcnet.gwcnet import GwcNet as BaseGwcNet
    from ..psmnet.psmnet import PSMNet as BasePSMNet


class StatBasedAutoExposure(nn.Module):
    def __init__(self, target_mean=0.5, momentum=0.5):
        super().__init__()
        self.target_mean = float(target_mean)
        self.momentum = float(momentum)

    def compute_target_ev(self, image, current_ev):
        current_stat = image.detach().float().mean(dim=(1, 2, 3))
        current_stat = torch.clamp(current_stat, min=1e-5)

        ratio = self.target_mean / current_stat
        ideal_target_ev = current_ev * ratio
        next_ev = current_ev * (1.0 - self.momentum) + ideal_target_ev * self.momentum
        return next_ev

    def forward(self, image: torch.Tensor, current_exposure: torch.Tensor) -> torch.Tensor:
        if image.ndim != 4:
            raise ValueError('image must be (B, C, H, W)')
        return self.compute_target_ev(image, current_exposure)


class AverageExposureControlMixin(ExposureControlMixin):
    def _build_exposure_controller(self, cfgs, time_limits, gain_limits):
        target_mean = float(_cfg_get(cfgs, 'AE_TARGET_MEAN', 0.5))
        momentum = float(_cfg_get(cfgs, 'AE_MOMENTUM', 0.5))
        return StatBasedAutoExposure(target_mean=target_mean, momentum=momentum)


class AverageAEGwcNet(AverageExposureControlMixin, BaseGwcNet):
    def __init__(self, cfgs, time_limits=(5.0, 20.0), gain_limits=(1.0, 20.0)):
        super(AverageAEGwcNet, self).__init__(cfgs)
        self._init_exposure_control(cfgs, default_time_limits=time_limits, default_gain_limits=gain_limits)

    def _forward_stereo(self, left_img, right_img):
        return super(AverageAEGwcNet, self).forward({'left': left_img, 'right': right_img})

    def forward(self, inputs):
        radiance_left, radiance_right = inputs['left'], inputs['right']
        seed_left, seed_right = self._build_seed_images(radiance_left, radiance_right)
        batch_size = radiance_left.shape[0]
        current_exp = self._expand_scalar_buffer(self.init_exp, batch_size)
        current_gain = self._expand_scalar_buffer(self.init_gain, batch_size)
        current_ev = current_exp * current_gain
        exposure = self.exposure_controller(self._build_state(seed_left, seed_right), current_ev)
        act_left, act_right = self._apply_exposure(radiance_left, radiance_right, exposure)
        return self._forward_stereo(act_left, act_right)


class AverageAEPSMNet(AverageExposureControlMixin, BasePSMNet):
    def __init__(self, cfgs, time_limits=(5.0, 20.0), gain_limits=(1.0, 20.0)):
        super(AverageAEPSMNet, self).__init__(cfgs)
        self._init_exposure_control(cfgs, default_time_limits=time_limits, default_gain_limits=gain_limits)

    def _forward_stereo(self, left_img, right_img):
        return super(AverageAEPSMNet, self).forward({'left': left_img, 'right': right_img})

    def forward(self, inputs):
        radiance_left, radiance_right = inputs['left'], inputs['right']
        seed_left, seed_right = self._build_seed_images(radiance_left, radiance_right)
        batch_size = radiance_left.shape[0]
        current_exp = self._expand_scalar_buffer(self.init_exp, batch_size)
        current_gain = self._expand_scalar_buffer(self.init_gain, batch_size)
        current_ev = current_exp * current_gain
        exposure = self.exposure_controller(self._build_state(seed_left, seed_right), current_ev)
        act_left, act_right = self._apply_exposure(radiance_left, radiance_right, exposure)
        return self._forward_stereo(act_left, act_right)


def unit_case_run_model():
    cfgs = SimpleNamespace(
        MAX_DISP=int(192),
        USE_CONCAT_VOLUME=bool(False),
        CONCAT_CHANNELS=int(8),
        DOWNSAMPLE=int(4),
        NUM_GROUPS=int(8),
    )

    model = AverageAEPSMNet(cfgs=cfgs)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    radiance_left = torch.rand(1, 3, 256, 512, device=device)
    radiance_right = torch.rand(1, 3, 256, 512, device=device)

    with torch.no_grad():
        disp_pred = model({'left': radiance_left, 'right': radiance_right})

    print(disp_pred)


if __name__ == '__main__':
    unit_case_run_model()
