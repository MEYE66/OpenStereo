import sys
from pathlib import Path

import torch


repo_root = Path(__file__).resolve().parents[4]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from stereo.modeling.models.gwcnet.gwcnet import GwcNet as BaseGwcNet
    from stereo.modeling.models.ournet.submodules import ActorModel, ExposureEnvModel, ValueModel
except ModuleNotFoundError:
    from ..gwcnet.gwcnet import GwcNet as BaseGwcNet
    from .submodules import ActorModel, ExposureEnvModel, ValueModel


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


def _extract_disp_tensor(disp_pred):
    if disp_pred.ndim == 4 and disp_pred.shape[1] == 1:
        return disp_pred.squeeze(1)
    return disp_pred


def _masked_abs_error_per_sample(model_pred, disp_gt, max_disp):
    disp_pred = _extract_disp_tensor(model_pred['disp_pred'])
    disp_gt = _extract_disp_tensor(disp_gt).to(disp_pred.dtype)
    valid = torch.isfinite(disp_gt) & (disp_gt > 0)
    if max_disp is not None:
        valid = valid & (disp_gt < float(max_disp))
    valid_f = valid.to(disp_pred.dtype)
    error = torch.abs(disp_pred - disp_gt)
    denom = valid_f.reshape(valid_f.shape[0], -1).sum(dim=1).clamp_min(1.0)
    return (error * valid_f).reshape(error.shape[0], -1).sum(dim=1) / denom


class RLOurAEGwcNet(BaseGwcNet):
    def __init__(self, cfgs):
        super(RLOurAEGwcNet, self).__init__(cfgs)

        self.rollout_steps = int(_cfg_get(cfgs, 'ROLLOUT_STEPS', 3))
        feature_dim = int(_cfg_get(cfgs, 'RL_FEATURE_DIM', 32))
        hidden_dim = int(_cfg_get(cfgs, 'RL_HIDDEN_DIM', feature_dim))
        downsample_size = int(_cfg_get(cfgs, 'RL_INPUT_SIZE', 128))
        time_delta_limit = float(
            _cfg_get(cfgs, 'TIME_DELTA_LIMIT', _cfg_get(cfgs, 'RL_TIME_DELTA_SCALE', _cfg_get(cfgs, 'RL_ACTION_SCALE', 2.0)))
        )
        gain_delta_limit = float(
            _cfg_get(cfgs, 'GAIN_DELTA_LIMIT', _cfg_get(cfgs, 'RL_GAIN_DELTA_SCALE', _cfg_get(cfgs, 'RL_ACTION_SCALE', 2.0)))
        )
        init_log_std = float(_cfg_get(cfgs, 'RL_INIT_LOG_STD', -1.0))

        self.env = ExposureEnvModel()
        self.actor = ActorModel(
            nums_in=6,
            nums_feat=feature_dim,
            nums_out=hidden_dim,
            action_dim=4,
            downsample_size=downsample_size,
        )
        self.critic = ValueModel(
            nums_in=7,
            nums_feat=feature_dim,
            nums_out=hidden_dim,
            downsample_size=downsample_size,
        )

    def _stereo_modules(self):
        return [self.Backbone, self.CostProcessor, self.DispProcessor]

    def _forward_stereo(self, left_img, right_img):
        return BaseGwcNet.forward(self, {'left': left_img, 'right': right_img})

    def _forward_stereo_observe(self, left_img, right_img):
        was_training = self.training
        self.training = False
        try:
            return BaseGwcNet.forward(self, {'left': left_img, 'right': right_img})
        finally:
            self.training = was_training

    def set_stereo_requires_grad(self, enabled):
        for module in self._stereo_modules():
            for param in module.parameters():
                param.requires_grad = enabled
        for param in self.actor.parameters():
            param.requires_grad = True
        for param in self.critic.parameters():
            param.requires_grad = True

    def set_stereo_train_mode(self, enabled):
        for module in self._stereo_modules():
            module.train(enabled)

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
        disp_gt = batch.get('disp', None)
        if disp_gt is None:
            batch_size = batch['left'].shape[0]
            return batch['left'].new_zeros(batch_size)
        return _masked_abs_error_per_sample(model_pred, disp_gt, self.maxdisp)

    @staticmethod
    def _action_penalty(action):
        return action.detach().abs().mean(dim=1)

    def rollout_episode(
        self,
        batch,
        steps,
        deterministic=False,
        need_stereo_grad=False,
        action_penalty_weight=0.0,
    ):
        left_radiance = batch['left']
        right_radiance = batch['right']
        batch_size = left_radiance.shape[0]
        total_steps = max(int(steps), 1)

        current_left, current_right, state_t, state_g = self.env.reset(left_radiance, right_radiance)
        seed_images = (current_left, current_right)

        with torch.no_grad():
            current_pred = self._forward_stereo_observe(current_left, current_right)
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
                current_disp,
                state_t,
                state_g,
                remaining_ratio,
                deterministic=deterministic,
            )
            value = self.critic(
                current_left,
                current_right,
                current_disp,
            )

            next_left, next_right, next_t, next_g = self.env.step(
                left_radiance,
                right_radiance,
                state_t,
                state_g,
                action,
            )

            with torch.no_grad():
                next_pred = self._forward_stereo_observe(next_left, next_right)

            epe_before = self._compute_error_or_zero(current_pred, batch)
            epe_after = self._compute_error_or_zero(next_pred, batch)
            reward = epe_before - epe_after
            if action_penalty_weight > 0.0:
                reward = reward - float(action_penalty_weight) * self._action_penalty(action)

            transitions.append({
                'value': value,
                'log_prob': log_prob,
                'entropy': entropy,
                'action': action,
                'reward': reward.detach(),
                'epe_before': epe_before.detach(),
                'epe_after': epe_after.detach(),
                'state_t': state_t.detach(),
                'state_g': state_g.detach(),
            })

            current_left, current_right = next_left, next_right
            state_t, state_g = next_t, next_g
            current_pred = next_pred

        final_images = (current_left, current_right)
        final_pred = current_pred
        if need_stereo_grad:
            final_pred = self._forward_stereo(current_left, current_right)

        return {
            'seed_pred': seed_pred,
            'final_pred': final_pred,
            'seed_images': seed_images,
            'final_images': final_images,
            'transitions': transitions,
        }

    def forward(self, inputs):
        rollout = self.rollout_episode(
            batch=inputs,
            steps=self.rollout_steps,
            deterministic=True,
            need_stereo_grad=self.training,
        )
        return rollout['final_pred']
