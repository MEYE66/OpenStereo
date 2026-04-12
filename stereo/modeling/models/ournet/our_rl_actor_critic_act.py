import sys
from pathlib import Path

import torch
import torch.nn.functional as F

repo_root = Path(__file__).resolve().parents[4]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from stereo.modeling.models.gwcnet.gwcnet import GwcNet as BaseGwcNet
    from stereo.modeling.models.ournet.submodules_act import ActorModel, ExposureEnvModel, ValueModel
except ModuleNotFoundError:
    from ..gwcnet.gwcnet import GwcNet as BaseGwcNet
    from .submodules_act import ActorModel, ExposureEnvModel, ValueModel


def _cfg_get(cfgs, key, default=None):
    if isinstance(cfgs, dict):
        return cfgs.get(key, default)
    return getattr(cfgs, key, default)


def _cfg_pair(cfgs, key, default=(0.0, 0.0), fallback_key=None):
    value = _cfg_get(cfgs, key, None)
    if value is None and fallback_key is not None:
        value = _cfg_get(cfgs, fallback_key, None)
    if value is None:
        value = default
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return float(value[0]), float(value[0])
        return float(value[0]), float(value[1])
    value = float(value)
    return value, value


class RLOurAEGwcNet(BaseGwcNet):
    def __init__(self, cfgs):
        super(RLOurAEGwcNet, self).__init__(cfgs)

        self.rollout_steps = int(_cfg_get(cfgs, 'ROLLOUT_STEPS', 3))
        self.time_limits = _cfg_pair(cfgs, 'TIME_LIMITS', default=(1.0, 20.0))
        self.gain_limits = _cfg_pair(cfgs, 'GAIN_LIMITS', default=(1.0, 20.0))
        self.init_time = _cfg_pair(cfgs, 'INIT_TIME', default=(10.0, 10.0))
        self.init_gain = _cfg_pair(cfgs, 'INIT_GAIN', default=(10.0, 10.0))
        self.nbits = int(_cfg_get(cfgs, 'NBITS', 8))

        feature_dim = int(_cfg_get(cfgs, 'RL_FEATURE_DIM', 32))
        hidden_dim = int(_cfg_get(cfgs, 'RL_HIDDEN_DIM', feature_dim))
        downsample_size = int(_cfg_get(cfgs, 'RL_INPUT_SIZE', 128))
        init_log_std = float(_cfg_get(cfgs, 'RL_INIT_LOG_STD', -0.5))

        action_scale = float(_cfg_get(cfgs, 'RL_ACTION_SCALE', 2.0))
        self.time_delta_limit = float(_cfg_get(cfgs, 'TIME_DELTA_LIMIT', _cfg_get(cfgs, 'MU_DELTA_LIMIT', action_scale)))
        self.gain_delta_limit = float(_cfg_get(cfgs, 'GAIN_DELTA_LIMIT', _cfg_get(cfgs, 'DELTA_DELTA_LIMIT', action_scale)))
        self.disp_reward_weight = float(_cfg_get(cfgs, 'DISP_REWARD_WEIGHT', 1.0))
        self.cover_reward_weight = float(_cfg_get(cfgs, 'COVER_REWARD_WEIGHT', 0.2))
        self.cover_low_thresh = float(_cfg_get(cfgs, 'COVER_LOW_THRESH', 0.05))
        self.cover_high_thresh = float(_cfg_get(cfgs, 'COVER_HIGH_THRESH', 0.95))
        cover_use_delta = _cfg_get(cfgs, 'COVER_USE_DELTA', True)
        if isinstance(cover_use_delta, str):
            cover_use_delta = cover_use_delta.lower() in ('1', 'true', 'yes', 'y')
        self.cover_use_delta = bool(cover_use_delta)

        self.env = ExposureEnvModel(
            time_limits=self.time_limits,
            gain_limits=self.gain_limits,
            init_time=self.init_time,
            init_gain=self.init_gain,
            nbits=self.nbits,
        )
        self.actor = ActorModel(
            nums_in=6,
            nums_feat=feature_dim,
            nums_out=hidden_dim,
            action_dim=4,
            state_dim=5,
            downsample_size=downsample_size,
            init_log_std=init_log_std,
            time_limits=self.time_limits,
            gain_limits=self.gain_limits,
            time_delta_limit=self.time_delta_limit,
            gain_delta_limit=self.gain_delta_limit,
        )
        self.critic = ValueModel(
            nums_in=7,
            nums_feat=feature_dim,
            nums_out=hidden_dim,
            state_dim=5,
            downsample_size=downsample_size,
            time_limits=self.time_limits,
            gain_limits=self.gain_limits,
        )

    def _stereo_modules(self):
        return [self.Backbone, self.CostProcessor, self.DispProcessor]

    def _forward_stereo(self, left_img, right_img):
        return BaseGwcNet.forward(self, {'left': left_img, 'right': right_img})

    def set_stereo_requires_grad(self, enabled):
        for module in self._stereo_modules():
            for param in module.parameters():
                param.requires_grad = False
        for param in self.actor.parameters():
            param.requires_grad = True
        for param in self.critic.parameters():
            param.requires_grad = True

    def set_stereo_train_mode(self, enabled):
        for module in self._stereo_modules():
            module.eval()

    def get_actor_parameters(self):
        return list(self.actor.parameters())

    def get_critic_parameters(self):
        return list(self.critic.parameters())

    def get_stereo_parameters(self):
        params = []
        for module in self._stereo_modules():
            params.extend(list(module.parameters()))
        return params

    def get_component_state_dicts(self):
        stereo_state = {}
        for module in self._stereo_modules():
            for key, value in module.state_dict().items():
                stereo_state[f'{module.__class__.__name__}.{key}'] = value
        return {
            'stereo_state': stereo_state,
            'actor_state': self.actor.state_dict(),
            'critic_state': self.critic.state_dict(),
        }

    def _compute_error_or_zero(self, model_pred, batch):
        disp_gt = batch['disp']
        disp_pred = model_pred['disp_pred']

        disp_gt = disp_gt.to(disp_pred.dtype)
        valid = (disp_gt < self.maxdisp) & (disp_gt > 0)
        per_pixel_error = F.smooth_l1_loss(disp_pred, disp_gt, reduction='none')
        masked_error = per_pixel_error.masked_fill(~valid, float('nan'))
        per_sample_error = torch.nanmean(masked_error.flatten(1), dim=1)
        return torch.nan_to_num(per_sample_error, nan=0.0)

    @staticmethod
    def _to_luma(image):
        if image.shape[1] == 1:
            return image[:, 0]
        if image.shape[1] >= 3:
            red = image[:, 0]
            green = image[:, 1]
            blue = image[:, 2]
            return 0.299 * red + 0.587 * green + 0.114 * blue
        return image.mean(dim=1)

    def _compute_low_high_ratio(self, image, low_thresh, high_thresh):
        luma = self._to_luma(image)
        low_ratio = (luma <= float(low_thresh)).to(luma.dtype).flatten(1).mean(dim=1)
        high_ratio = (luma >= float(high_thresh)).to(luma.dtype).flatten(1).mean(dim=1)
        return low_ratio, high_ratio

    def _compute_joint_coverage_score(self, left_image, right_image, low_thresh, high_thresh):
        low_left, high_left = self._compute_low_high_ratio(left_image, low_thresh, high_thresh)
        low_right, high_right = self._compute_low_high_ratio(right_image, low_thresh, high_thresh)

        joint_low = torch.minimum(low_left, low_right)
        joint_high = torch.minimum(high_left, high_right)
        coverage = low_left.new_tensor(1.0) - 0.5 * (joint_low + joint_high)
        stats = {
            'low_left': low_left,
            'high_left': high_left,
            'low_right': low_right,
            'high_right': high_right,
        }
        return coverage, stats

    def _shape_joint_coverage_reward(self, current_left, current_right, next_left, next_right, low_thresh, high_thresh):
        current_cover, stats_current = self._compute_joint_coverage_score(
            current_left,
            current_right,
            low_thresh,
            high_thresh,
        )
        next_cover, stats_next = self._compute_joint_coverage_score(
            next_left,
            next_right,
            low_thresh,
            high_thresh,
        )
        reward_cover_delta = next_cover - current_cover
        return reward_cover_delta, current_cover, next_cover, stats_current, stats_next

    @staticmethod
    def _action_penalty(action):
        return action.detach().abs().mean(dim=1)

    def _shape_disp_reward(self, err_before, err_after):
        return err_before - err_after

    def _shape_reward(self, err_before, err_after):
        return self._shape_disp_reward(err_before, err_after)

    def rollout_episode(
        self,
        batch,
        steps,
        deterministic=False,
        action_penalty_weight=0.0,
    ):
        left_radiance = batch['left']
        right_radiance = batch['right']
        batch_size = left_radiance.shape[0]
        total_steps = max(int(steps), 1)

        current_left, current_right, state_t, state_g = self.env.reset(left_radiance, right_radiance)
        seed_images = (current_left, current_right)

        with torch.no_grad():
            current_pred = self._forward_stereo(current_left, current_right)
        seed_pred = current_pred

        transitions = []
        for step_idx in range(total_steps):
            remaining_ratio = current_left.new_full(
                (batch_size, 1),
                float(total_steps - step_idx) / float(total_steps),
            )

            current_disp = current_pred['disp_pred'].detach()
            action, log_prob, entropy = self.actor(
                current_left,
                current_right,
                state_t,
                state_g,
                remaining_ratio,
                deterministic=deterministic,
            )
            value = self.critic(
                current_left,
                current_right,
                current_disp,
                state_t,
                state_g,
                remaining_ratio,
            )
            next_left, next_right, next_t, next_g = self.env.step(
                left_radiance,
                right_radiance,
                state_t,
                state_g,
                action,
            )

            with torch.no_grad():
                next_pred = self._forward_stereo(next_left, next_right)
            err_before = self._compute_error_or_zero(current_pred, batch)
            err_after = self._compute_error_or_zero(next_pred, batch)
            reward_disp = self._shape_disp_reward(err_before, err_after)
            reward_cover_delta, current_cover, next_cover, stats_current, stats_next = self._shape_joint_coverage_reward(
                current_left,
                current_right,
                next_left,
                next_right,
                self.cover_low_thresh,
                self.cover_high_thresh,
            )
            reward_cover = reward_cover_delta if self.cover_use_delta else next_cover

            reward = self.disp_reward_weight * reward_disp + self.cover_reward_weight * reward_cover
            if action_penalty_weight > 0.0:
                reward = reward - float(action_penalty_weight) * self._action_penalty(action)

            decoded = self.env.decode_states_t_g(state_t.detach(), state_g.detach())
            transitions.append({
                'value': value,
                'log_prob': log_prob,
                'entropy': entropy,
                'action': action,
                'reward': reward.detach(),
                'reward_disp': reward_disp.detach(),
                'reward_cover': reward_cover.detach(),
                'reward_cover_delta': reward_cover_delta.detach(),
                'cover_current': current_cover.detach(),
                'cover_next': next_cover.detach(),
                'low_ratio_left_current': stats_current['low_left'].detach(),
                'high_ratio_left_current': stats_current['high_left'].detach(),
                'low_ratio_right_current': stats_current['low_right'].detach(),
                'high_ratio_right_current': stats_current['high_right'].detach(),
                'low_ratio_left_next': stats_next['low_left'].detach(),
                'high_ratio_left_next': stats_next['high_left'].detach(),
                'low_ratio_right_next': stats_next['low_right'].detach(),
                'high_ratio_right_next': stats_next['high_right'].detach(),
                'epe_before': err_before.detach(),
                'epe_after': err_after.detach(),
                'state_t': state_t.detach(),
                'state_g': state_g.detach(),
                'decoded_ev_lr': decoded['ev_lr'].detach(),
                'decoded_t': decoded['time'].detach(),
                'decoded_g': decoded['gain'].detach(),
            })

            current_left, current_right = next_left, next_right
            state_t, state_g = next_t, next_g
            current_pred = next_pred

        final_images = (current_left, current_right)
        final_pred = current_pred

        with torch.no_grad():
            bootstrap_remaining = current_left.new_zeros((batch_size, 1))
            bootstrap_disp = final_pred['disp_pred'].detach()
            bootstrap_value = self.critic(
                current_left,
                current_right,
                bootstrap_disp,
                state_t,
                state_g,
                bootstrap_remaining,
            )

        return {
            'seed_pred': seed_pred,
            'final_pred': final_pred,
            'seed_images': seed_images,
            'final_images': final_images,
            'transitions': transitions,
            'bootstrap_value': bootstrap_value,
        }

    def forward(self, inputs):
        rollout = self.rollout_episode(
            batch=inputs,
            steps=self.rollout_steps,
            deterministic=True,
        )
        return rollout['final_pred']
