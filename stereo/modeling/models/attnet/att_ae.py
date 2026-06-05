import sys
from pathlib import Path
from types import SimpleNamespace

import torch


repo_root = Path(__file__).resolve().parents[4]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from stereo.modeling.models.ae_util import _cfg_get, ExposureControlMixin
    from stereo.modeling.models.gwcnet.gwcnet import GwcNet as BaseGwcNet
    from stereo.modeling.models.psmnet.psmnet import PSMNet as BasePSMNet
    from stereo.modeling.models.attnet.submodules import AttExposureController
except ModuleNotFoundError:
    from ..ae_util import _cfg_get, ExposureControlMixin
    from ..gwcnet.gwcnet import GwcNet as BaseGwcNet
    from ..psmnet.psmnet import PSMNet as BasePSMNet
    from .submodules import AttExposureController


class AttExposureControlMixin(ExposureControlMixin):
    def _build_exposure_controller(self, cfgs, time_limits, gain_limits):
        base_channels = int(_cfg_get(cfgs, 'ATT_BASE_CHANNELS', 16))
        max_exposure = float(_cfg_get(cfgs, 'ATT_MAX_EXPOSURE', 10.0))
        sigmoid_scale = float(_cfg_get(cfgs, 'ATT_SIGMOID_SCALE', 3.0))
        return AttExposureController(
            in_channels=3,
            base_channels=base_channels,
            max_exposure=max_exposure,
            sigmoid_scale=sigmoid_scale,
        )

    def _init_att_exposure_control(
        self,
        cfgs,
        default_time_limits=(5.0, 20.0),
        default_gain_limits=(1.0, 20.0),
        default_iter=3,
        default_mu=0.8,
    ):
        self._init_exposure_control(
            cfgs,
            default_time_limits=default_time_limits,
            default_gain_limits=default_gain_limits,
        )
        self.iters = int(_cfg_get(cfgs, 'AE_ITERS', default_iter))
        self.mu = float(_cfg_get(cfgs, 'AE_MU', default_mu))

    @staticmethod
    def _stabilize_positive(x, eps=1e-6):
        return torch.clamp(x, min=eps)

    def update_function(self, e_t, u_t):
        e_t = self._stabilize_positive(e_t)
        u_t = self._stabilize_positive(u_t)
        new_log_ev = self.mu * torch.log(e_t) + (1.0 - self.mu) * torch.log(e_t * u_t)
        return torch.exp(new_log_ev)


class AttAEGwcNet(AttExposureControlMixin, BaseGwcNet):
    def __init__(self, cfgs, time_limits=(5.0, 20.0), gain_limits=(1.0, 20.0), iter=3, mu=0.8):
        super(AttAEGwcNet, self).__init__(cfgs)
        self._init_att_exposure_control(
            cfgs,
            default_time_limits=time_limits,
            default_gain_limits=gain_limits,
            default_iter=iter,
            default_mu=mu,
        )

    def _forward_stereo(self, left_img, right_img):
        return super(AttAEGwcNet, self).forward({'left': left_img, 'right': right_img})

    def forward(self, inputs):
        radiance_left, radiance_right = inputs['left'], inputs['right']
        img_left_updated, img_right_updated = self._build_seed_images(radiance_left, radiance_right)

        batch_size = radiance_left.shape[0]
        expo_update = self._expand_scalar_buffer(self.init_exp, batch_size)
        gain_update = self._expand_scalar_buffer(self.init_gain, batch_size)
        current_ev = expo_update * gain_update

        for _ in range(self.iters):
            state = self._build_state(img_left_updated, img_right_updated)
            control_ratio = self.exposure_controller(state)
            current_ev = self.update_function(current_ev, control_ratio)
            img_left_updated, img_right_updated = self._apply_exposure(radiance_left, radiance_right, current_ev)

        return self._forward_stereo(img_left_updated, img_right_updated)


class AttAEPSMNet(AttExposureControlMixin, BasePSMNet):
    def __init__(self, cfgs, time_limits=(1.0, 20.0), gain_limits=(1.0, 14.0), iter=3, mu=0.8):
        super(AttAEPSMNet, self).__init__(cfgs)
        self._init_att_exposure_control(
            cfgs,
            default_time_limits=time_limits,
            default_gain_limits=gain_limits,
            default_iter=iter,
            default_mu=mu,
        )

    def _forward_stereo(self, left_img, right_img):
        return super(AttAEPSMNet, self).forward({'left': left_img, 'right': right_img})

    def forward(self, inputs):
        radiance_left, radiance_right = inputs['left'], inputs['right']
        img_left_updated, img_right_updated = self._build_seed_images(radiance_left, radiance_right)

        batch_size = radiance_left.shape[0]
        expo_update = self._expand_scalar_buffer(self.init_exp, batch_size)
        gain_update = self._expand_scalar_buffer(self.init_gain, batch_size)
        current_ev = expo_update * gain_update

        for _ in range(self.iters):
            state = self._build_state(img_left_updated, img_right_updated)
            control_ratio = self.exposure_controller(state)
            current_ev = self.update_function(current_ev, control_ratio)
            img_left_updated, img_right_updated = self._apply_exposure(radiance_left, radiance_right, current_ev)

        return self._forward_stereo(img_left_updated, img_right_updated)


if __name__ == '__main__':
    cfgs = SimpleNamespace(
        MAX_DISP=int(192),
        USE_CONCAT_VOLUME=bool(False),
        CONCAT_CHANNELS=int(8),
        DOWNSAMPLE=int(4),
        NUM_GROUPS=int(8),
    )
    model = AttAEGwcNet(cfgs=cfgs)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    radiance_left = torch.rand(2, 3, 256, 512, device=device)
    radiance_right = torch.rand(2, 3, 256, 512, device=device)

    with torch.no_grad():
        disp_pred = model({'left': radiance_left, 'right': radiance_right})
    print(disp_pred)
