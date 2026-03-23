import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .ae_util import _cfg_get, exposure_value_equation


def _grad_score(gray):
    kernel_x = gray.new_tensor([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]]).view(1, 1, 3, 3)
    kernel_y = gray.new_tensor([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]).view(1, 1, 3, 3)
    gray = F.pad(gray, (1, 1, 1, 1), mode='reflect')
    gx = F.conv2d(gray, kernel_x)
    gy = F.conv2d(gray, kernel_y)
    return torch.sqrt(gx * gx + gy * gy + 1e-12).mean(dim=(1, 2, 3))


def _masked_abs_error_per_sample(disp_pred, disp_gt, mask):
    error = torch.abs(disp_pred - disp_gt).float().view(disp_pred.shape[0], -1)
    valid = mask.float().view(mask.shape[0], -1)
    valid_num = valid.sum(dim=1).clamp(min=1.0)
    return (error * valid).sum(dim=1) / valid_num


class ActorCriticExposureController(nn.Module):
    def __init__(
        self,
        time_delta_scale=2.0,
        gain_delta_scale=2.0,
        hidden_dim=64,
        init_log_std=-1.0,
    ):
        super().__init__()
        self.time_delta_scale = float(time_delta_scale)
        self.gain_delta_scale = float(gain_delta_scale)
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

        self.encoder = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )
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

    def _encode(self, left_image, right_image):
        feat = self.encoder(torch.cat([left_image, right_image], dim=1))
        feat = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        left_gray = left_image.mean(dim=1, keepdim=True)
        right_gray = right_image.mean(dim=1, keepdim=True)
        left_mean = left_gray.mean(dim=(1, 2, 3), keepdim=True)
        right_mean = right_gray.mean(dim=(1, 2, 3), keepdim=True)
        left_grad = _grad_score(left_gray).view(-1, 1)
        right_grad = _grad_score(right_gray).view(-1, 1)
        return torch.cat([feat, left_mean.view(-1, 1), right_mean.view(-1, 1), left_grad, right_grad], dim=1)

    def _build_distribution(self, residual_mean, residual_scale):
        max_scale = float(residual_scale.max().item())
        std = torch.exp(self.log_std).clamp(min=1e-4, max=max(max_scale, 1.0))
        return Normal(residual_mean, std.expand_as(residual_mean))

    def forward(self, left_image, right_image, deterministic=False):
        obs_feat = self._encode(left_image, right_image)
        residual_scale = self.delta_scale.view(1, 4)
        residual_mean = residual_scale * torch.tanh(self.actor_head(obs_feat))
        value = self.critic_head(obs_feat).squeeze(1)

        dist = self._build_distribution(residual_mean, residual_scale)
        sampled_residual = residual_mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(sampled_residual).sum(dim=1)
        entropy = dist.entropy().sum(dim=1)
        residual_action = torch.clamp(sampled_residual, -residual_scale, residual_scale)

        return residual_action, log_prob, entropy, value


class ActorCriticExposureMixin:
    def _init_rl_controller(self, cfgs):
        time_delta_scale = float(_cfg_get(cfgs, 'RL_TIME_DELTA_SCALE', _cfg_get(cfgs, 'RL_ACTION_SCALE', 2.0)))
        gain_delta_scale = float(_cfg_get(cfgs, 'RL_GAIN_DELTA_SCALE', _cfg_get(cfgs, 'RL_ACTION_SCALE', 2.0)))
        hidden_dim = int(_cfg_get(cfgs, 'RL_HIDDEN_DIM', 64))
        init_log_std = float(_cfg_get(cfgs, 'RL_INIT_LOG_STD', -1.0))

        self.rl_controller = ActorCriticExposureController(
            time_delta_scale=time_delta_scale,
            gain_delta_scale=gain_delta_scale,
            hidden_dim=hidden_dim,
            init_log_std=init_log_std,
        )

    def set_stereo_requires_grad(self, enabled):
        for name, param in self.named_parameters():
            if name.startswith('rl_controller.'):
                continue
            param.requires_grad = enabled
        for param in self.rl_controller.parameters():
            param.requires_grad = True

    def get_rl_parameters(self, rl_cfg=None):
        if rl_cfg is None:
            rl_cfg = {}
        train_stereo = bool(rl_cfg.get('TRAIN_STEREO', True))
        if train_stereo:
            return [p for p in self.parameters() if p.requires_grad]
        return list(self.rl_controller.parameters())

    def forward(self, inputs):
        model_pred, _ = self.forward_rl(inputs, deterministic=not self.training)
        return model_pred

    def forward_rl(self, inputs, deterministic=False):
        radiance_left, radiance_right = inputs['left'], inputs['right']

        seed_left, seed_right = self._build_seed_images(radiance_left, radiance_right)
        state = self._build_state(seed_left, seed_right)

        base_exposure = self.exposure_controller(state)
        base_time, base_gain = exposure_value_equation(base_exposure, self.time_limits, self.gain_limits)
        base_action = torch.stack([base_time, base_gain, base_time, base_gain], dim=1)

        residual_action, log_prob, entropy, value = self.rl_controller(
            seed_left, seed_right, deterministic=deterministic
        )
        action = base_action + residual_action
        action[:, 0] = torch.clamp(action[:, 0], self.time_limits[0], self.time_limits[1])
        action[:, 2] = torch.clamp(action[:, 2], self.time_limits[0], self.time_limits[1])
        action[:, 1] = torch.clamp(action[:, 1], self.gain_limits[0], self.gain_limits[1])
        action[:, 3] = torch.clamp(action[:, 3], self.gain_limits[0], self.gain_limits[1])

        act_left = self.image_formation_model(radiance_left, action[:, 0], action[:, 1])
        act_right = self.image_formation_model(radiance_right, action[:, 2], action[:, 3])
        model_pred = self._forward_stereo(act_left, act_right)

        baseline_pred = None
        with torch.no_grad():
            base_left = self.image_formation_model(radiance_left, base_action[:, 0], base_action[:, 1])
            base_right = self.image_formation_model(radiance_right, base_action[:, 2], base_action[:, 3])
            baseline_pred = self._forward_stereo(base_left, base_right)

        exposure_action = (action[:, 0] * action[:, 1] + action[:, 2] * action[:, 3]) / 2.0
        base_exposure_action = (base_action[:, 0] * base_action[:, 1] + base_action[:, 2] * base_action[:, 3]) / 2.0

        rl_info = {
            'log_prob': log_prob,
            'entropy': entropy,
            'value': value,
            'exposure_action': exposure_action,
            'base_exposure': base_exposure_action,
            'residual_action': residual_action,
            'action_params': action,
            'base_action_params': base_action,
            'baseline_pred': baseline_pred,
        }
        return model_pred, rl_info

    def get_rl_loss(self, model_preds, input_data, rl_info, rl_cfg=None):
        if rl_cfg is None:
            rl_cfg = {}

        disp_loss, tb_info = self.get_loss(model_preds, input_data)

        disp_gt = input_data['disp']
        mask = (disp_gt < self.maxdisp) & (disp_gt > 0)
        pred_error = _masked_abs_error_per_sample(model_preds['disp_pred'].squeeze(1), disp_gt, mask)

        if rl_info['baseline_pred'] is not None:
            with torch.no_grad():
                baseline_error = _masked_abs_error_per_sample(
                    rl_info['baseline_pred']['disp_pred'].squeeze(1), disp_gt, mask
                )
            reward = baseline_error - pred_error.detach()
        else:
            reward = -pred_error.detach()

        action_penalty_weight = float(rl_cfg.get('ACTION_PENALTY_WEIGHT', 0.0))
        if action_penalty_weight > 0.0:
            reward = reward - action_penalty_weight * rl_info['residual_action'].detach().abs()

        reward = reward * float(rl_cfg.get('REWARD_SCALE', 1.0))

        advantage = reward - rl_info['value'].detach()
        actor_loss = -(rl_info['log_prob'] * advantage).mean()
        actor_loss = actor_loss - float(rl_cfg.get('ENTROPY_WEIGHT', 0.001)) * rl_info['entropy'].mean()
        critic_loss = F.mse_loss(rl_info['value'], reward)

        disp_loss_weight = float(rl_cfg.get('DISP_LOSS_WEIGHT', 1.0 if rl_cfg.get('TRAIN_STEREO', True) else 0.0))
        actor_loss_weight = float(rl_cfg.get('ACTOR_LOSS_WEIGHT', 1.0))
        critic_loss_weight = float(rl_cfg.get('CRITIC_LOSS_WEIGHT', 0.5))

        total_loss = disp_loss_weight * disp_loss + actor_loss_weight * actor_loss + critic_loss_weight * critic_loss

        tb_info.update({
            'scalar/train/loss_actor': actor_loss.item(),
            'scalar/train/loss_critic': critic_loss.item(),
            'scalar/train/reward': reward.mean().item(),
            'scalar/train/pred_error': pred_error.mean().item(),
            'scalar/train/exposure_action': rl_info['exposure_action'].mean().item(),
            'scalar/train/residual_action': rl_info['residual_action'].mean().item(),
            'scalar/train/time_left': rl_info['action_params'][:, 0].mean().item(),
            'scalar/train/gain_left': rl_info['action_params'][:, 1].mean().item(),
            'scalar/train/time_right': rl_info['action_params'][:, 2].mean().item(),
            'scalar/train/gain_right': rl_info['action_params'][:, 3].mean().item(),
        })
        if rl_info['base_exposure'] is not None:
            tb_info['scalar/train/base_exposure'] = rl_info['base_exposure'].mean().item()
        return total_loss, tb_info
