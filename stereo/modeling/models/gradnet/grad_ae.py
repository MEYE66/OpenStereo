import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


repo_root = Path(__file__).resolve().parents[4]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from stereo.modeling.models.ae_util import _cfg_get, ExposureControlMixin
    from stereo.modeling.models.gwcnet.gwcnet import GwcNet as BaseGwcNet
    from stereo.modeling.models.gwcnet.gwcnet_lidar_sparse import LidarGwcNet
    from stereo.modeling.models.psmnet.psmnet import PSMNet as BasePSMNet
    from stereo.modeling.models.gwcnet.gwcnet_sequence import GwcSequenceNet

except ModuleNotFoundError:
    from ..ae_util import _cfg_get, ExposureControlMixin
    from ..gwcnet.gwcnet import GwcNet as BaseGwcNet
    from ..gwcnet.gwcnet_lidar_sparse import LidarGwcNet
    from ..psmnet.psmnet import PSMNet as BasePSMNet

    from ..gwcnet.gwcnet_sequence import GwcSequenceNet




def _grad_score(gray):
    kernel_x = gray.new_tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]).view(1, 1, 3, 3)
    kernel_y = gray.new_tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]).view(1, 1, 3, 3)
    gray = F.pad(gray, (1, 1, 1, 1), mode='reflect')
    gx = F.conv2d(gray, kernel_x)
    gy = F.conv2d(gray, kernel_y)
    return torch.sqrt(gx * gx + gy * gy + 1e-12).mean(dim=(1, 2, 3))


class GradientExposureController(nn.Module):
    def __init__(self, target_grad=0.12, min_exposure=1.0, max_exposure=20.0):
        super().__init__()
        self.target_grad = float(target_grad)
        self.min_exposure = float(min_exposure)
        self.max_exposure = float(max_exposure)

    def forward(self, image):
        gray = image.mean(dim=1, keepdim=True)
        grad_strength = _grad_score(gray)
        exp_val = self.target_grad / (grad_strength + 1e-6)
        return torch.clamp(exp_val, self.min_exposure, self.max_exposure)


class GradientExposureControlMixin(ExposureControlMixin):
    def _build_exposure_controller(self, cfgs, time_limits, gain_limits):
        target_grad = float(_cfg_get(cfgs, 'AE_TARGET_GRAD', _cfg_get(cfgs, 'TARGET_GRAD', 0.12)))
        return GradientExposureController(
            target_grad=target_grad,
            min_exposure=time_limits[0]*gain_limits[0],
            max_exposure=time_limits[1]*gain_limits[1],
        )


class GradientAEGwcNet(GradientExposureControlMixin, BaseGwcNet):
    def __init__(self, cfgs, time_limits=(1.0, 20.0), gain_limits=(1.0, 14.0)):
        super(GradientAEGwcNet, self).__init__(cfgs)
        self._init_exposure_control(cfgs, default_time_limits=time_limits, default_gain_limits=gain_limits)

    def _forward_stereo(self, left_img, right_img):
        return super(GradientAEGwcNet, self).forward({'left': left_img, 'right': right_img})

    def forward(self, inputs):
        radiance_left, radiance_right = inputs['left'], inputs['right']
        seed_left, seed_right = self._build_seed_images(radiance_left, radiance_right)
        exposure = self.exposure_controller(self._build_state(seed_left, seed_right))
        act_left, act_right = self._apply_exposure(radiance_left, radiance_right, exposure)
        return self._forward_stereo(act_left, act_right)


class GradientAELidarGwcNet(GradientExposureControlMixin, LidarGwcNet):
    def __init__(self, cfgs, time_limits=(1.0, 20.0), gain_limits=(1.0, 14.0)):
        super(GradientAELidarGwcNet, self).__init__(cfgs)
        self._init_exposure_control(cfgs, default_time_limits=time_limits, default_gain_limits=gain_limits)

    def _forward_stereo(self, left_img, right_img):
        return LidarGwcNet.forward(self, {'left': left_img, 'right': right_img})

    def forward(self, inputs):
        radiance_left, radiance_right = inputs['left'], inputs['right']
        seed_left, seed_right = self._build_seed_images(radiance_left, radiance_right)
        exposure = self.exposure_controller(self._build_state(seed_left, seed_right))
        act_left, act_right = self._apply_exposure(radiance_left, radiance_right, exposure)
        return self._forward_stereo(act_left, act_right)


class GardientAEPSMNet(GradientExposureControlMixin, BasePSMNet):
    def __init__(self, cfgs, time_limits=(1.0, 20.0), gain_limits=(1.0, 14.0)):
        super(GardientAEPSMNet, self).__init__(cfgs)
        self._init_exposure_control(cfgs, default_time_limits=time_limits, default_gain_limits=gain_limits)

    def _forward_stereo(self, left_img, right_img):
        return super(GardientAEPSMNet, self).forward({'left': left_img, 'right': right_img})

    def forward(self, inputs):
        radiance_left, radiance_right = inputs['left'], inputs['right']
        seed_left, seed_right = self._build_seed_images(radiance_left, radiance_right)
        exposure = self.exposure_controller(self._build_state(seed_left, seed_right))
        act_left, act_right = self._apply_exposure(radiance_left, radiance_right, exposure)
        return self._forward_stereo(act_left, act_right)


class GradientAEGwcSequenceNet(GradientExposureControlMixin, GwcSequenceNet):
    def __init__(self, cfgs, time_limits=(1.0, 20.0), gain_limits=(1.0, 14.0)):
        super(GradientAEGwcSequenceNet, self).__init__(cfgs)
        self._init_exposure_control(cfgs, default_time_limits=time_limits, default_gain_limits=gain_limits)

    def forward(self, inputs):
        curr_radiance_left, curr_radiance_right = inputs['left_1'], inputs['right_1']
        next_radiance_left, next_radiance_right = inputs['left_2'], inputs['right_2']

        curr_seed_left, _ = self._build_seed_images(curr_radiance_left, curr_radiance_right)
        next_seed_left, _ = self._build_seed_images(next_radiance_left, next_radiance_right)
        exposure = self.exposure_controller(self._build_state(curr_seed_left, next_seed_left))
        curr_left, curr_right = self._apply_exposure(curr_radiance_left, curr_radiance_right, exposure)
        next_left, next_right = self._apply_exposure(next_radiance_left, next_radiance_right, exposure)

        return super(GradientAEGwcSequenceNet, self).forward({
            'left_1': curr_left,
            'right_1': curr_right,
            'left_2': next_left,
            'right_2': next_right
        })


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
    print(disp_pred.keys())


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

    with torch.no_grad():
        disp_pred = model({
            'left_1': radiance_left,
            'right_1': radiance_right,
            'left_2': radiance_left,
            'right_2': radiance_right
        })
    print(disp_pred)


if __name__ == '__main__':
    # stereo_test()
    sequence_test()
