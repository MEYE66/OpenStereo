import torch
from stereo.modeling.models.ae_util import _cfg_get, ExposureControlMixin
from stereo.modeling.models.naenet.submodules import FloorDivSTE, NeuralExposureController
from stereo.modeling.models.raftstereo.model import RAFT_Stereo_Dual


class _RAFTConfig(dict):
    def __getattr__(self, name):
        return self[name]


def _build_raft_config(cfgs):
    return _RAFTConfig(
        {
            'MAX_DISP': int(_cfg_get(cfgs, 'MAX_DISP', 192)),
            'CORR_IMPLEMENTATION': _cfg_get(cfgs, 'CORR_IMPLEMENTATION', 'reg'),
            'CORR_LEVELS': int(_cfg_get(cfgs, 'CORR_LEVELS', 4)),
            'CORR_RADIUS': int(_cfg_get(cfgs, 'CORR_RADIUS', 4)),
            'CONTEXT_NORM': _cfg_get(cfgs, 'CONTEXT_NORM', 'batch'),
            'HIDDEN_DIMS': list(_cfg_get(cfgs, 'HIDDEN_DIMS', [128, 128, 128])),
            'MIXED_PRECISION': bool(_cfg_get(cfgs, 'MIXED_PRECISION', False)),
            'N_DOWNSAMPLE': int(_cfg_get(cfgs, 'N_DOWNSAMPLE', 2)),
            'N_GRU_LAYERS': int(_cfg_get(cfgs, 'N_GRU_LAYERS', 3)),
            'SHARED_BACKBONE': bool(_cfg_get(cfgs, 'SHARED_BACKBONE', False)),
            'SLOW_FAST_GRU': bool(_cfg_get(cfgs, 'SLOW_FAST_GRU', False)),
            'TRAIN_ITERS': int(_cfg_get(cfgs, 'TRAIN_ITERS', 22)),
            'EVAL_ITERS': int(_cfg_get(cfgs, 'EVAL_ITERS', 32)),
            'LOSS_GAMMA': float(_cfg_get(cfgs, 'LOSS_GAMMA', 0.9)),
        }
    )


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

    def _render_pair(self, radiance_left, radiance_right, exposure, gain):
        left_img = self.image_formation_model(radiance_left, exposure, gain)
        right_img = self.image_formation_model(radiance_right, exposure, gain)
        return left_img, right_img


    def _render_frame(self, radiance, exposure, gain):
        return self.image_formation_model(radiance, exposure, gain)

    def _estimate_pair_exposure(self, radiance_left, radiance_right):
        batch_size = radiance_left.shape[0]
        exposure = self._expand_scalar_buffer(self.init_exp, batch_size)
        gain = self._expand_scalar_buffer(self.init_gain, batch_size)
        left_img, right_img = self._render_pair(radiance_left, radiance_right, exposure, gain)

        for _ in range(self.iters):
            values = self._to_batch_vector(self.exposure_controller((left_img + right_img) / 2))
            exposure, gain = self.update_function(exposure, values)
            left_img, right_img = self._render_pair(radiance_left, radiance_right, exposure, gain)

        return left_img, right_img

    def _estimate_frame_exposure(self, radiance):
        batch_size = radiance.shape[0]
        exposure = self._expand_scalar_buffer(self.init_exp, batch_size)
        gain = self._expand_scalar_buffer(self.init_gain, batch_size)
        image = self._render_frame(radiance, exposure, gain)

        for _ in range(self.iters):
            values = self._to_batch_vector(self.exposure_controller(image))
            exposure, gain = self.update_function(exposure, values)
            image = self._render_frame(radiance, exposure, gain)

        return image

class NeuralAERAFTStereoDual(NeuralExposureControlMixin, RAFT_Stereo_Dual):
    def __init__(self, cfgs, time_limits=(5.0, 20.0), gain_limits=(1.0, 20.0), iter=3, mu=0.8):
        super().__init__(_build_raft_config(cfgs))
        self._init_neural_exposure_control(
            cfgs,
            default_time_limits=time_limits,
            default_gain_limits=gain_limits,
            default_iter=iter,
            default_mu=mu,
        )

    def forward(self, inputs, iters=None, flow_init=None, test_mode=False):
        curr_left, curr_right = self._estimate_pair_exposure(inputs['left_1'], inputs['right_1'])
        next_left = self._estimate_frame_exposure(inputs['left_2'])
        next_right = self._estimate_frame_exposure(inputs['right_2'])

        return super().forward(
            {
                'left_1': curr_left,
                'right_1': curr_right,
                'left_2': next_left,
                'right_2': next_right,
            },
            iters=iters,
            flow_init=flow_init,
            test_mode=test_mode,
        )

