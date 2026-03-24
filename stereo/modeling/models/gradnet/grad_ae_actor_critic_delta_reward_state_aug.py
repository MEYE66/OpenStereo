import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


repo_root = Path(__file__).resolve().parents[4]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from stereo.modeling.models.ae_util import (
        _cfg_get,
        _cfg_pair,
        ImageFormationModel,
        exposure_value_equation,
    )
    from stereo.modeling.models.rl_util import _grad_score, _masked_abs_error_per_sample
    from stereo.modeling.models.gwcnet.gwcnet import GwcNet as BaseGwcNet
    from stereo.modeling.models.psmnet.psmnet import PSMNet as BasePSMNet
    from stereo.modeling.models.gradnet.grad_ae import GradientExposureController
except ModuleNotFoundError:
    from ..ae_util import (
        _cfg_get,
        _cfg_pair,
        ImageFormationModel,
        exposure_value_equation,
    )
    from ..rl_util import _grad_score, _masked_abs_error_per_sample
    from ..gwcnet.gwcnet import GwcNet as BaseGwcNet
    from ..psmnet.psmnet import PSMNet as BasePSMNet
    from .grad_ae import GradientExposureController


class _ConvEncoder(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        feat = self.net(x)
        return F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)


class ActorCriticExposureControllerStateAug(nn.Module):
    def __init__(
        self,
        time_delta_scale=2.0,
        gain_delta_scale=2.0,
        hidden_dim=64,
        init_log_std=-1.0,
        input_size=128,
    ):
        super().__init__()
        self.time_delta_scale = float(time_delta_scale)
        self.gain_delta_scale = float(gain_delta_scale)
        self.input_size = int(input_size)

        self.register_buffer(
            'delta_scale',
            torch.tensor(
                [
                    self.time_delta_scale,
                    self.gain_delta_scale,
                    self.time_delta_scale,
                    self.gain_delta_scale,
                ],
                dtype=torch.float32,
            ),
        )

        # Policy/Value use the same encoder architecture, but separate parameters.
        # Input channels are aligned to 10 for both:
        # 6(image pair) + 4(extra channels)
        self.policy_encoder = _ConvEncoder(in_channels=10)
        self.value_encoder = _ConvEncoder(in_channels=10)

        self.actor_head = nn.Sequential(
            nn.Linear(36, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 4),
        )
        self.critic_head = nn.Sequential(
            nn.Linear(36, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        self.log_std = nn.Parameter(torch.full((4,), float(init_log_std), dtype=torch.float32))

    def _downsample_pair(self, left_image, right_image):
        size = (self.input_size, self.input_size)
        left_ds = F.interpolate(left_image, size=size, mode='bilinear', align_corners=False)
        right_ds = F.interpolate(right_image, size=size, mode='bilinear', align_corners=False)
        left_ds = torch.clamp(left_ds, 0.0, 1.0)
        right_ds = torch.clamp(right_ds, 0.0, 1.0)
        return left_ds, right_ds

    @staticmethod
    def _normalize_action(action, time_limits, gain_limits):
        t_min, t_max = time_limits[0], time_limits[1]
        g_min, g_max = gain_limits[0], gain_limits[1]
        t_left = (action[:, 0] - t_min) / (t_max - t_min + 1e-6)
        g_left = (action[:, 1] - g_min) / (g_max - g_min + 1e-6)
        t_right = (action[:, 2] - t_min) / (t_max - t_min + 1e-6)
        g_right = (action[:, 3] - g_min) / (g_max - g_min + 1e-6)
        return torch.stack([t_left, g_left, t_right, g_right], dim=1).clamp(0.0, 1.0)

    def _policy_extra_channels(self, base_action, time_limits, gain_limits, h, w):
        encoded = self._normalize_action(base_action, time_limits, gain_limits)
        return encoded.view(encoded.shape[0], 4, 1, 1).expand(-1, -1, h, w)

    @staticmethod
    def _luminance(img):
        return 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]

    @staticmethod
    def _saturation(img):
        max_c = img.max(dim=1, keepdim=True).values
        min_c = img.min(dim=1, keepdim=True).values
        return (max_c - min_c) / (max_c + 1e-6)

    def _value_extra_channels(self, left_image, right_image, h, w):
        left_lum = self._luminance(left_image)
        right_lum = self._luminance(right_image)

        lum_mean = 0.5 * (
            left_lum.mean(dim=(1, 2, 3), keepdim=True) + right_lum.mean(dim=(1, 2, 3), keepdim=True)
        )

        contrast = 0.5 * (
            left_lum.std(dim=(1, 2, 3), keepdim=True, unbiased=False)
            + right_lum.std(dim=(1, 2, 3), keepdim=True, unbiased=False)
        )

        sat_mean = 0.5 * (
            self._saturation(left_image).mean(dim=(1, 2, 3), keepdim=True)
            + self._saturation(right_image).mean(dim=(1, 2, 3), keepdim=True)
        )

        # Pad one zero channel to keep policy/value encoder input channels identical (10).
        zero_pad = torch.zeros_like(lum_mean)
        stats = torch.cat([lum_mean, contrast, sat_mean, zero_pad], dim=1)
        return stats.expand(-1, -1, h, w)

    @staticmethod
    def _obs_context(left_image, right_image):
        left_gray = left_image.mean(dim=1, keepdim=True)
        right_gray = right_image.mean(dim=1, keepdim=True)
        left_mean = left_gray.mean(dim=(1, 2, 3), keepdim=True)
        right_mean = right_gray.mean(dim=(1, 2, 3), keepdim=True)
        left_grad = _grad_score(left_gray).view(-1, 1)
        right_grad = _grad_score(right_gray).view(-1, 1)
        return torch.cat([left_mean.view(-1, 1), right_mean.view(-1, 1), left_grad, right_grad], dim=1)

    def _build_distribution(self, residual_mean, residual_scale):
        max_scale = float(residual_scale.max().item())
        std = torch.exp(self.log_std).clamp(min=1e-4, max=max(max_scale, 1.0))
        return Normal(residual_mean, std.expand_as(residual_mean))

    def forward(self, left_image, right_image, base_action, time_limits, gain_limits, deterministic=False):
        left_ds, right_ds = self._downsample_pair(left_image, right_image)
        h, w = left_ds.shape[-2], left_ds.shape[-1]

        policy_extra = self._policy_extra_channels(base_action, time_limits, gain_limits, h, w)
        value_extra = self._value_extra_channels(left_ds, right_ds, h, w)

        policy_in = torch.cat([left_ds, right_ds, policy_extra], dim=1)
        value_in = torch.cat([left_ds, right_ds, value_extra], dim=1)

        policy_feat = self.policy_encoder(policy_in)
        value_feat = self.value_encoder(value_in)

        context_feat = self._obs_context(left_ds, right_ds)
        policy_latent = torch.cat([policy_feat, context_feat], dim=1)
        value_latent = torch.cat([value_feat, context_feat], dim=1)

        residual_scale = self.delta_scale.view(1, 4)
        residual_mean = residual_scale * torch.tanh(self.actor_head(policy_latent))
        value = self.critic_head(value_latent).squeeze(1)

        dist = self._build_distribution(residual_mean, residual_scale)
        sampled_residual = residual_mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(sampled_residual).sum(dim=1)
        entropy = dist.entropy().sum(dim=1)
        residual_action = torch.clamp(sampled_residual, -residual_scale, residual_scale)

        return residual_action, log_prob, entropy, value


class StereoExposureAgent(nn.Module):
    def __init__(self, cfgs, default_time_limits=(5.0, 20.0), default_gain_limits=(1.0, 20.0)):
        super().__init__()
        time_limits = _cfg_pair(cfgs, 'TIME_LIMITS', default_time_limits)
        gain_limits = _cfg_pair(cfgs, 'GAIN_LIMITS', default_gain_limits)

        init_exposure = float(_cfg_get(cfgs, 'INIT_EXPOSURE', sum(time_limits) / 2.0))
        init_gain = float(_cfg_get(cfgs, 'INIT_GAIN', sum(gain_limits) / 2.0))
        target_grad = float(_cfg_get(cfgs, 'AE_TARGET_GRAD', _cfg_get(cfgs, 'TARGET_GRAD', 0.12)))

        self.image_formation_model = ImageFormationModel()
        self.base_controller = GradientExposureController(
            target_grad=target_grad,
            min_exposure=time_limits[0] * gain_limits[0],
            max_exposure=time_limits[1] * gain_limits[1],
        )

        time_delta_scale = float(_cfg_get(cfgs, 'RL_TIME_DELTA_SCALE', _cfg_get(cfgs, 'RL_ACTION_SCALE', 2.0)))
        gain_delta_scale = float(_cfg_get(cfgs, 'RL_GAIN_DELTA_SCALE', _cfg_get(cfgs, 'RL_ACTION_SCALE', 2.0)))
        hidden_dim = int(_cfg_get(cfgs, 'RL_HIDDEN_DIM', 64))
        init_log_std = float(_cfg_get(cfgs, 'RL_INIT_LOG_STD', -1.0))
        input_size = int(_cfg_get(cfgs, 'RL_INPUT_SIZE', 128))

        self.rl_controller = ActorCriticExposureControllerStateAug(
            time_delta_scale=time_delta_scale,
            gain_delta_scale=gain_delta_scale,
            hidden_dim=hidden_dim,
            init_log_std=init_log_std,
            input_size=input_size,
        )

        self.register_buffer('time_limits', torch.tensor(time_limits, dtype=torch.float32))
        self.register_buffer('gain_limits', torch.tensor(gain_limits, dtype=torch.float32))
        self.register_buffer('init_exp', torch.tensor([init_exposure], dtype=torch.float32))
        self.register_buffer('init_gain', torch.tensor([init_gain], dtype=torch.float32))

    @staticmethod
    def _expand_scalar_buffer(buffer, batch_size):
        return buffer.view(1).expand(batch_size)

    def _build_seed_images(self, radiance_left, radiance_right):
        batch_size = radiance_left.shape[0]
        init_exp = self._expand_scalar_buffer(self.init_exp, batch_size)
        init_gain = self._expand_scalar_buffer(self.init_gain, batch_size)

        seed_left = self.image_formation_model(radiance_left, init_exp, init_gain)
        seed_right = self.image_formation_model(radiance_right, init_exp, init_gain)
        return seed_left, seed_right

    def _build_base_action(self, seed_left, seed_right):
        base_exposure_left = self.base_controller(seed_left)
        base_exposure_right = self.base_controller(seed_right)

        base_time_left, base_gain_left = exposure_value_equation(base_exposure_left, self.time_limits, self.gain_limits)
        base_time_right, base_gain_right = exposure_value_equation(base_exposure_right, self.time_limits, self.gain_limits)

        return torch.stack([base_time_left, base_gain_left, base_time_right, base_gain_right], dim=1)

    def _clamp_action(self, action):
        time_left = torch.clamp(action[:, 0], self.time_limits[0], self.time_limits[1])
        gain_left = torch.clamp(action[:, 1], self.gain_limits[0], self.gain_limits[1])
        time_right = torch.clamp(action[:, 2], self.time_limits[0], self.time_limits[1])
        gain_right = torch.clamp(action[:, 3], self.gain_limits[0], self.gain_limits[1])
        return torch.stack([time_left, gain_left, time_right, gain_right], dim=1)

    def _render_pair(self, radiance_left, radiance_right, action):
        left_img = self.image_formation_model(radiance_left, action[:, 0], action[:, 1])
        right_img = self.image_formation_model(radiance_right, action[:, 2], action[:, 3])
        left_img = torch.clamp(left_img, 0.0, 1.0)
        right_img = torch.clamp(right_img, 0.0, 1.0)
        return left_img, right_img

    @staticmethod
    def _ev_from_action(action):
        left_ev = action[:, 0] * action[:, 1]
        right_ev = action[:, 2] * action[:, 3]
        return (left_ev + right_ev) / 2.0

    def sample_actions(self, radiance_left, radiance_right, deterministic=False):
        seed_left, seed_right = self._build_seed_images(radiance_left, radiance_right)
        seed_left = torch.clamp(seed_left, 0.0, 1.0)
        seed_right = torch.clamp(seed_right, 0.0, 1.0)

        base_action = self._build_base_action(seed_left, seed_right)
        residual_action, log_prob, entropy, value = self.rl_controller(
            seed_left,
            seed_right,
            base_action,
            self.time_limits,
            self.gain_limits,
            deterministic=deterministic,
        )

        action = self._clamp_action(base_action + residual_action)
        act_left, act_right = self._render_pair(radiance_left, radiance_right, action)

        return {
            'left_img': act_left,
            'right_img': act_right,
            'seed_left_img': seed_left,
            'seed_right_img': seed_right,
            'log_prob': log_prob,
            'entropy': entropy,
            'value': value,
            'residual_action': residual_action,
            'action_params': action,
            'base_action_params': base_action,
            'exposure_action': self._ev_from_action(action),
            'base_exposure': self._ev_from_action(base_action),
        }


def _set_stereo_requires_grad(model, enabled):
    for name, param in model.named_parameters():
        if name.startswith('exposure_agent.rl_controller.'):
            continue
        param.requires_grad = enabled
    for param in model.exposure_agent.rl_controller.parameters():
        param.requires_grad = True


def _get_rl_parameters(model, rl_cfg=None):
    if rl_cfg is None:
        rl_cfg = {}
    train_stereo = bool(rl_cfg.get('TRAIN_STEREO', True))
    if train_stereo:
        return [p for p in model.parameters() if p.requires_grad]
    return list(model.exposure_agent.rl_controller.parameters())


def _forward_rl_common(model, inputs, deterministic=False):
    radiance_left, radiance_right = inputs['left'], inputs['right']
    action_info = model.exposure_agent.sample_actions(
        radiance_left, radiance_right, deterministic=deterministic
    )

    model_pred = model._forward_stereo(action_info['left_img'], action_info['right_img'])
    with torch.no_grad():
        state_t_pred = model._forward_stereo(action_info['seed_left_img'], action_info['seed_right_img'])

    rl_info = {
        'log_prob': action_info['log_prob'],
        'entropy': action_info['entropy'],
        'value': action_info['value'],
        'exposure_action': action_info['exposure_action'],
        'base_exposure': action_info['base_exposure'],
        'residual_action': action_info['residual_action'],
        'action_params': action_info['action_params'],
        'base_action_params': action_info['base_action_params'],
        'state_t_pred': state_t_pred,
    }
    return model_pred, rl_info


def _get_rl_loss_common(model, model_preds, input_data, rl_info, rl_cfg=None):
    if rl_cfg is None:
        rl_cfg = {}

    disp_loss, tb_info = model.get_loss(model_preds, input_data)

    disp_gt = input_data['disp']
    mask = (disp_gt < model.maxdisp) & (disp_gt > 0)
    pred_error = _masked_abs_error_per_sample(model_preds['disp_pred'].squeeze(1), disp_gt, mask)

    with torch.no_grad():
        state_t_error = _masked_abs_error_per_sample(
            rl_info['state_t_pred']['disp_pred'].squeeze(1), disp_gt, mask
        )

    # r_d = disp_loss(s_{t+1}) - disp_loss(s_t)
    reward = pred_error.detach() - state_t_error

    action_penalty_weight = float(rl_cfg.get('ACTION_PENALTY_WEIGHT', 0.0))
    if action_penalty_weight > 0.0:
        reward = reward - action_penalty_weight * rl_info['residual_action'].detach().abs().mean(dim=1)

    reward = reward * float(rl_cfg.get('REWARD_SCALE', 1.0))

    advantage = reward - rl_info['value'].detach()
    actor_loss = -(rl_info['log_prob'] * advantage).mean()
    actor_loss = actor_loss - float(rl_cfg.get('ENTROPY_WEIGHT', 0.001)) * rl_info['entropy'].mean()
    critic_loss = F.mse_loss(rl_info['value'], reward)

    disp_loss_weight = float(rl_cfg.get('DISP_LOSS_WEIGHT', 1.0))
    actor_loss_weight = float(rl_cfg.get('ACTOR_LOSS_WEIGHT', 1.0))
    critic_loss_weight = float(rl_cfg.get('CRITIC_LOSS_WEIGHT', 0.5))

    total_loss = disp_loss_weight * disp_loss + actor_loss_weight * actor_loss + critic_loss_weight * critic_loss

    tb_info.update({
        'scalar/train/loss_actor': actor_loss.item(),
        'scalar/train/loss_critic': critic_loss.item(),
        'scalar/train/reward': reward.mean().item(),
        'scalar/train/pred_error': pred_error.mean().item(),
        'scalar/train/state_t_error': state_t_error.mean().item(),
        'scalar/train/exposure_action': rl_info['exposure_action'].mean().item(),
        'scalar/train/residual_action': rl_info['residual_action'].mean().item(),
        'scalar/train/time_left': rl_info['action_params'][:, 0].mean().item(),
        'scalar/train/gain_left': rl_info['action_params'][:, 1].mean().item(),
        'scalar/train/time_right': rl_info['action_params'][:, 2].mean().item(),
        'scalar/train/gain_right': rl_info['action_params'][:, 3].mean().item(),
        'scalar/train/base_exposure': rl_info['base_exposure'].mean().item(),
    })
    return total_loss, tb_info


class RLGradientAEGwcNetStateAug(BaseGwcNet):
    def __init__(self, cfgs, time_limits=(1.0, 20.0), gain_limits=(1.0, 14.0)):
        super(RLGradientAEGwcNetStateAug, self).__init__(cfgs)
        self.exposure_agent = StereoExposureAgent(
            cfgs=cfgs,
            default_time_limits=time_limits,
            default_gain_limits=gain_limits,
        )

    def _forward_stereo(self, left_img, right_img):
        return BaseGwcNet.forward(self, {'left': left_img, 'right': right_img})

    def set_stereo_requires_grad(self, enabled):
        _set_stereo_requires_grad(self, enabled)

    def get_rl_parameters(self, rl_cfg=None):
        return _get_rl_parameters(self, rl_cfg)

    def forward(self, inputs):
        model_pred, _ = self.forward_rl(inputs, deterministic=not self.training)
        return model_pred

    def forward_rl(self, inputs, deterministic=False):
        return _forward_rl_common(self, inputs, deterministic=deterministic)

    def get_rl_loss(self, model_preds, input_data, rl_info, rl_cfg=None):
        return _get_rl_loss_common(self, model_preds, input_data, rl_info, rl_cfg)


class RLGardientAEPSMNetStateAug(BasePSMNet):
    def __init__(self, cfgs, time_limits=(1.0, 20.0), gain_limits=(1.0, 14.0)):
        super(RLGardientAEPSMNetStateAug, self).__init__(cfgs)
        self.exposure_agent = StereoExposureAgent(
            cfgs=cfgs,
            default_time_limits=time_limits,
            default_gain_limits=gain_limits,
        )

    def _forward_stereo(self, left_img, right_img):
        return BasePSMNet.forward(self, {'left': left_img, 'right': right_img})

    def set_stereo_requires_grad(self, enabled):
        _set_stereo_requires_grad(self, enabled)

    def get_rl_parameters(self, rl_cfg=None):
        return _get_rl_parameters(self, rl_cfg)

    def forward(self, inputs):
        model_pred, _ = self.forward_rl(inputs, deterministic=not self.training)
        return model_pred

    def forward_rl(self, inputs, deterministic=False):
        return _forward_rl_common(self, inputs, deterministic=deterministic)

    def get_rl_loss(self, model_preds, input_data, rl_info, rl_cfg=None):
        return _get_rl_loss_common(self, model_preds, input_data, rl_info, rl_cfg)


if __name__ == '__main__':
    import types

    cfgs = types.SimpleNamespace(
        MAX_DISP=int(192),
        USE_CONCAT_VOLUME=bool(False),
        CONCAT_CHANNELS=int(8),
        DOWNSAMPLE=int(4),
        NUM_GROUPS=int(8),
        TIME_LIMITS=(1.0, 20.0),
        GAIN_LIMITS=(1.0, 14.0),
        INIT_EXPOSURE=10.0,
        INIT_GAIN=7.0,
        AE_TARGET_GRAD=0.12,
        RL_HIDDEN_DIM=128,
        RL_ACTION_SCALE=1.0,
        RL_INIT_LOG_STD=-1.0,
        RL_INPUT_SIZE=128,
    )

    model = RLGradientAEGwcNetStateAug(cfgs=cfgs)
    inputs = {
        'left': torch.rand(2, 3, 256, 512),
        'right': torch.rand(2, 3, 256, 512),
    }
    model_pred, rl_info = model.forward_rl(inputs)
    print(model_pred.keys())
    print(rl_info.keys())
