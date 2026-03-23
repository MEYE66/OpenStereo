import sys
from pathlib import Path
from types import SimpleNamespace

import torch


repo_root = Path(__file__).resolve().parents[4]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from stereo.modeling.models.neuralnet.submodules import NeuralExposureController, FloorDivSTE
    from stereo.modeling.models.ae_util import _cfg_get, ExposureControlMixin
    from stereo.modeling.models.gwcnet.gwcnet import GwcNet as BaseGwcNet
    from stereo.modeling.models.psmnet.psmnet import PSMNet as BasePSMNet
    from stereo.modeling.models.gwcnet.gwcnet_sequence import GwcSequenceNet
except ModuleNotFoundError:
    from .submodules import NeuralExposureController, FloorDivSTE
    from ..ae_util import _cfg_get, ExposureControlMixin
    from ..gwcnet.gwcnet import GwcNet as BaseGwcNet
    from ..gwcnet.gwcnet_sequence import GwcSequenceNet
    from ..psmnet.psmnet import PSMNet as BasePSMNet


class NeuralExposureControlMixin(ExposureControlMixin):
    def _build_exposure_controller(self, cfgs, time_limits, gain_limits):
        return NeuralExposureController()

    def _init_neural_exposure_control(
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
        self.floor_div = FloorDivSTE()
        self.iters = int(_cfg_get(cfgs, 'AE_ITERS', default_iter))
        self.mu = float(_cfg_get(cfgs, 'AE_MU', default_mu))

    @staticmethod
    def _to_batch_vector(x):
        if x.ndim == 1:
            return x
        return x.reshape(x.shape[0], -1).mean(dim=1)

    def update_function(self, e_t, u_t):
        e_t = self._to_batch_vector(e_t)
        u_t = self._to_batch_vector(u_t)
        log_e_t = self.mu * torch.log(e_t) + (1 - self.mu) * torch.log(e_t * u_t)
        new_e_t = torch.exp(log_e_t)
        gain = torch.max(
            self.gain_limits[0],
            torch.min(self.gain_limits[1], self.floor_div.apply(new_e_t, self.time_limits[1])),
        )
        exp_time = torch.max(self.time_limits[0], torch.min(self.time_limits[1], new_e_t / gain))
        return exp_time, gain


class NeuralAEGwcNet(NeuralExposureControlMixin, BaseGwcNet):
    def __init__(self, cfgs, time_limits=(5.0, 20.0), gain_limits=(1.0, 20.0), iter=3, mu=0.8):
        super(NeuralAEGwcNet, self).__init__(cfgs)
        self._init_neural_exposure_control(
            cfgs,
            default_time_limits=time_limits,
            default_gain_limits=gain_limits,
            default_iter=iter,
            default_mu=mu,
        )

    def _forward_stereo(self, left_img, right_img):
        return super(NeuralAEGwcNet, self).forward({'left': left_img, 'right': right_img})

    def forward(self, inputs):
        radiance_left, radiance_right = inputs['left'], inputs['right']
        img_left_updated, img_right_updated = self._build_seed_images(radiance_left, radiance_right)
        batch_size = radiance_left.shape[0]
        expo_update = self._expand_scalar_buffer(self.init_exp, batch_size)

        for _ in range(self.iters):
            values = self._to_batch_vector(self.exposure_controller((img_left_updated + img_right_updated) / 2))
            expo_update, gain_update = self.update_function(expo_update, values)
            img_left_updated = self.image_formation_model(radiance_left, expo_update, gain_update)
            img_right_updated = self.image_formation_model(radiance_right, expo_update, gain_update)

        return self._forward_stereo(img_left_updated, img_right_updated)


class NeuralAEGwcSequenceNet(NeuralExposureControlMixin, GwcSequenceNet):
    def __init__(self, cfgs, time_limits=(5.0, 20.0), gain_limits=(1.0, 20.0), iter=3, mu=0.8):
        super(NeuralAEGwcSequenceNet, self).__init__(cfgs)
        self._init_neural_exposure_control(
            cfgs,
            default_time_limits=time_limits,
            default_gain_limits=gain_limits,
            default_iter=iter,
            default_mu=mu,
        )

    def forward(self, inputs):
        curr_radiance_left, curr_radiance_right = inputs['left_1'], inputs['right_1']
        next_radiance_left, next_radiance_right = inputs['left_2'], inputs['right_2']

        curr_img_left, _ = self._build_seed_images(curr_radiance_left, curr_radiance_right)
        next_img_left, _ = self._build_seed_images(next_radiance_left, next_radiance_right)

        batch_size = curr_radiance_left.shape[0]
        expo_update = self._expand_scalar_buffer(self.init_exp, batch_size)

        for _ in range(self.iters):
            values = self._to_batch_vector(self.exposure_controller((curr_img_left + next_img_left) / 2))
            expo_update, gain_update = self.update_function(expo_update, values)

            curr_left_updated = self.image_formation_model(curr_radiance_left, expo_update, gain_update)
            curr_right_updated = self.image_formation_model(curr_radiance_right, expo_update, gain_update)
            next_left_updated = self.image_formation_model(next_radiance_left, expo_update, gain_update)
            next_right_updated = self.image_formation_model(next_radiance_right, expo_update, gain_update)

        return super(NeuralAEGwcSequenceNet, self).forward(
            {
                'left_1': curr_left_updated,
                'right_1': curr_right_updated,
                'left_2': next_left_updated,
                'right_2': next_right_updated,
            }
        )


class NeuralAEPSMNet(NeuralExposureControlMixin, BasePSMNet):
    def __init__(self, cfgs, time_limits=(1.0, 20.0), gain_limits=(1.0, 14.0), iter=3, mu=0.8):
        super(NeuralAEPSMNet, self).__init__(cfgs)
        self._init_neural_exposure_control(
            cfgs,
            default_time_limits=time_limits,
            default_gain_limits=gain_limits,
            default_iter=iter,
            default_mu=mu,
        )

    def _forward_stereo(self, left_img, right_img):
        return super(NeuralAEPSMNet, self).forward({'left': left_img, 'right': right_img})

    def forward(self, inputs):
        radiance_left, radiance_right = inputs['left'], inputs['right']
        img_left_updated, img_right_updated = self._build_seed_images(radiance_left, radiance_right)
        batch_size = radiance_left.shape[0]
        expo_update = self._expand_scalar_buffer(self.init_exp, batch_size)

        for _ in range(self.iters):
            values = self._to_batch_vector(self.exposure_controller((img_left_updated + img_right_updated) / 2))
            expo_update, gain_update = self.update_function(expo_update, values)
            img_left_updated = self.image_formation_model(radiance_left, expo_update, gain_update)
            img_right_updated = self.image_formation_model(radiance_right, expo_update, gain_update)

        return self._forward_stereo(img_left_updated, img_right_updated)


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
    radiance_left = torch.rand(2, 3, 256, 512, device=device)
    radiance_right = torch.rand(2, 3, 256, 512, device=device)

    with torch.no_grad():
        disp_pred = model({'left': radiance_left, 'right': radiance_right})
    print(disp_pred)
