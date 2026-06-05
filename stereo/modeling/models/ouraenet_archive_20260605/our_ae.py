import sys
from pathlib import Path

import torch
import torch.nn.functional as F


repo_root = Path(__file__).resolve().parents[4]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from stereo.modeling.models.raftstereo.raft_stereo import RAFTStereoFusionDual
    from stereo.modeling.models.ouraenet.submodules import (
        FusionExposureEnvModel,
        OurAEActorModel,
        OurAEValueModel,
    )
except ModuleNotFoundError:
    from ..raftstereo.raft_stereo import RAFTStereoFusionDual
    from .submodules import FusionExposureEnvModel, OurAEActorModel, OurAEValueModel


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


def _adapt_raft_cfgs(cfgs):
    if hasattr(cfgs, 'get') and not isinstance(cfgs, dict):
        return cfgs

    if isinstance(cfgs, dict):
        cfg_dict = dict(cfgs)
    else:
        cfg_dict = dict(vars(cfgs))

    class _CfgAdapter(dict):
        def __getattr__(self, item):
            try:
                return self[item]
            except KeyError as exc:
                raise AttributeError(item) from exc

    return _CfgAdapter(cfg_dict)


class OurAENet(RAFTStereoFusionDual):
    def __init__(self, cfgs):
        cfgs = _adapt_raft_cfgs(cfgs)
        super().__init__(cfgs)

        self.rollout_steps = int(_cfg_get(cfgs, 'ROLLOUT_STEPS', 3))
        self.time_limits = _cfg_pair(cfgs, 'TIME_LIMITS', default=(1.0, 20.0))
        self.gain_limits = _cfg_pair(cfgs, 'GAIN_LIMITS', default=(1.0, 20.0))
        self.init_time = _cfg_pair(cfgs, 'INIT_TIME', default=(7.5, 7.5))
        self.init_gain = _cfg_pair(cfgs, 'INIT_GAIN', default=(7.5, 7.5))
        self.nbits = int(_cfg_get(cfgs, 'NBITS', 10))

        feature_dim = int(_cfg_get(cfgs, 'RL_FEATURE_DIM', 32))
        hidden_dim = int(_cfg_get(cfgs, 'RL_HIDDEN_DIM', feature_dim))
        downsample_size = int(_cfg_get(cfgs, 'RL_INPUT_SIZE', 256))
        init_log_std = float(_cfg_get(cfgs, 'RL_INIT_LOG_STD', -1.0))

        action_scale = float(_cfg_get(cfgs, 'RL_ACTION_SCALE', 1.0))
        self.time_delta_limit = float(_cfg_get(cfgs, 'TIME_DELTA_LIMIT', action_scale))
        self.gain_delta_limit = float(_cfg_get(cfgs, 'GAIN_DELTA_LIMIT', action_scale))
        self.disp_reward_weight = float(_cfg_get(cfgs, 'DISP_REWARD_WEIGHT', 1.0))
        self.reward_disp_clip = float(_cfg_get(cfgs, 'REWARD_DISP_CLIP', 1.0))

        self.env = FusionExposureEnvModel(
            time_limits=self.time_limits,
            gain_limits=self.gain_limits,
            init_time=self.init_time,
            init_gain=self.init_gain,
            nbits=self.nbits,
        )
        self.actor = OurAEActorModel(
            nums_in=12,
            nums_feat=feature_dim,
            nums_out=hidden_dim,
            downsample_size=downsample_size,
            init_log_std=init_log_std,
            time_limits=self.time_limits,
            gain_limits=self.gain_limits,
            time_delta_limit=self.time_delta_limit,
            gain_delta_limit=self.gain_delta_limit,
        )
        self.critic = OurAEValueModel(
            nums_feat=feature_dim,
            nums_out=hidden_dim,
            downsample_size=downsample_size,
            time_limits=self.time_limits,
            gain_limits=self.gain_limits,
        )

    def _stereo_modules(self):
        return [
            self.cnet,
            self.fnet,
            self.feature_fusion,
            self.update_block,
            self.context_zqr_convs,
        ]

    def set_stereo_requires_grad(self, enabled):
        for module in self._stereo_modules():
            for param in module.parameters():
                param.requires_grad = bool(enabled)
        for param in self.actor.parameters():
            param.requires_grad = True
        for param in self.critic.parameters():
            param.requires_grad = True

    def set_stereo_train_mode(self, enabled):
        for module in self._stereo_modules():
            module.train(mode=bool(enabled))

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
        for module_name in [
            'cnet',
            'fnet',
            'feature_fusion',
            'update_block',
            'context_zqr_convs',
        ]:
            module = getattr(self, module_name)
            for key, value in module.state_dict().items():
                stereo_state[f'{module_name}.{key}'] = value
        return {
            'stereo_state': stereo_state,
            'actor_state': self.actor.state_dict(),
            'critic_state': self.critic.state_dict(),
        }

    def _forward_stereo_fusion(self, left_1, right_1, left_2, right_2):
        return RAFTStereoFusionDual.forward(
            self,
            {
                'left_1': left_1,
                'right_1': right_1,
                'left_2': left_2,
                'right_2': right_2,
            },
        )

    @staticmethod
    def _masked_mean_per_sample(values, valid_mask):
        masked_values = values.masked_fill(~valid_mask, float('nan'))
        per_sample = torch.nanmean(masked_values.flatten(1), dim=1)
        return torch.nan_to_num(per_sample, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_disp_target(self, batch):
        disp_gt = batch.get('disp', batch.get('disp_1'))
        if disp_gt is None:
            raise KeyError('OurAENet requires disp or disp_1 in the batch.')
        if disp_gt.ndim == 4:
            disp_gt = disp_gt.squeeze(1)
        return disp_gt

    def _get_valid_mask(self, batch, disp_gt):
        valid = torch.isfinite(disp_gt) & (disp_gt > 0) & (disp_gt < self.max_disp)
        input_valid = batch.get('valid', batch.get('valid_1'))
        if torch.is_tensor(input_valid):
            if input_valid.ndim == 4:
                input_valid = input_valid.squeeze(1)
            valid = valid & input_valid.bool()
        return valid

    @staticmethod
    def _sanitize_disp_target(disp_gt):
        return torch.nan_to_num(disp_gt, nan=0.0, posinf=0.0, neginf=0.0)

    def _compute_stereo_loss(self, model_pred, batch):
        raw_disp_gt = self._get_disp_target(batch).to(model_pred['disp_pred'].dtype)
        valid = self._get_valid_mask(batch, raw_disp_gt)
        disp_gt = self._sanitize_disp_target(raw_disp_gt)

        if 'disp_preds' in model_pred:
            disp_preds = model_pred['disp_preds']
        else:
            disp_preds = [model_pred['disp_pred']]

        n_predictions = len(disp_preds)
        adjusted_loss_gamma = 1.0
        if n_predictions > 1:
            adjusted_loss_gamma = self.loss_gamma ** (15 / (n_predictions - 1))

        loss = disp_gt.new_zeros(disp_gt.shape[0])
        for idx, disp_pred in enumerate(disp_preds):
            weight = adjusted_loss_gamma ** (n_predictions - idx - 1)
            if disp_pred.ndim == 4:
                disp_pred = disp_pred.squeeze(1)
            per_pixel_loss = F.smooth_l1_loss(disp_pred, disp_gt, reduction='none')
            loss = loss + float(weight) * self._masked_mean_per_sample(per_pixel_loss, valid)
        return loss

    def _get_disp_target_stats(self, batch):
        disp_target = self._get_disp_target(batch)
        valid = self._get_valid_mask(batch, disp_target)
        finite = torch.isfinite(disp_target)
        flat_size = disp_target.flatten(1).shape[1]
        return {
            'disp_finite_ratio': finite.flatten(1).to(torch.float32).sum(dim=1) / float(flat_size),
            'disp_valid_ratio': valid.flatten(1).to(torch.float32).sum(dim=1) / float(flat_size),
        }

    @staticmethod
    def _get_exposure_stats(exposure_state):
        return {
            'exposure_time': exposure_state[:, [0, 2]],
            'gain': exposure_state[:, [1, 3]],
        }

    def _shape_disp_reward(self, loss_before, loss_after):
        return loss_before - loss_after

    @staticmethod
    def _clip_reward_component(reward_component, reward_clip):
        reward_clip = float(reward_clip)
        if reward_clip > 0.0:
            return torch.clamp(reward_component, -reward_clip, reward_clip)
        return reward_component

    @staticmethod
    def _require_dual_frame_batch(batch):
        required_keys = ['left_1', 'right_1', 'left_2', 'right_2']
        missing_keys = [key for key in required_keys if key not in batch]
        if missing_keys:
            raise KeyError(
                'OurAENet requires four sub-frame inputs. Missing keys: {}'.format(
                    ', '.join(missing_keys)
                )
            )

    def rollout_episode(self, batch, steps, deterministic=False):
        self._require_dual_frame_batch(batch)
        left_radiance_1 = batch['left_1']
        right_radiance_1 = batch['right_1']
        left_radiance_2 = batch['left_2']
        right_radiance_2 = batch['right_2']
        total_steps = max(int(steps), 1)
        disp_stats = self._get_disp_target_stats(batch)

        with torch.no_grad():
            exposure_state = self.env.get_initial_state(left_radiance_1)
            current_left_1, current_right_1, current_left_2, current_right_2 = self.env.render_dual_frames(
                left_radiance_1,
                right_radiance_1,
                left_radiance_2,
                right_radiance_2,
                exposure_state,
            )
            current_pred = self._forward_stereo_fusion(
                current_left_1,
                current_right_1,
                current_left_2,
                current_right_2,
            )

        seed_images = (current_left_1, current_right_1)
        seed_pred = current_pred

        transitions = []
        for _ in range(total_steps):
            action, log_prob, entropy = self.actor(
                current_left_1.detach(),
                current_right_1.detach(),
                current_left_2.detach(),
                current_right_2.detach(),
                exposure_state.detach(),
                deterministic=deterministic,
            )
            value = self.critic(current_pred['disp_pred'].detach(), exposure_state.detach())
            next_exposure_state = self.env.apply_action_to_state(exposure_state, action)

            with torch.no_grad():
                next_left_1, next_right_1, next_left_2, next_right_2 = self.env.render_dual_frames(
                    left_radiance_1,
                    right_radiance_1,
                    left_radiance_2,
                    right_radiance_2,
                    next_exposure_state.detach(),
                )
                next_pred = self._forward_stereo_fusion(
                    next_left_1,
                    next_right_1,
                    next_left_2,
                    next_right_2,
                )

            loss_before = self._compute_stereo_loss(current_pred, batch)
            loss_after = self._compute_stereo_loss(next_pred, batch)
            reward_disp_raw = self._shape_disp_reward(loss_before, loss_after)
            reward_disp = self._clip_reward_component(reward_disp_raw, self.reward_disp_clip)
            reward = self.disp_reward_weight * reward_disp

            transition = {
                'value': value,
                'log_prob': log_prob,
                'entropy': entropy,
                'action': action.detach(),
                'reward': reward.detach(),
                'stereo_loss_before': loss_before.detach(),
                'stereo_loss_after': loss_after.detach(),
                'exposure_state': next_exposure_state.detach(),
            }
            for key, value in self._get_exposure_stats(next_exposure_state.detach()).items():
                transition[key] = value
            for key, value in disp_stats.items():
                transition[key] = value.detach()
            transitions.append(transition)

            exposure_state = next_exposure_state.detach()
            current_left_1 = next_left_1.detach()
            current_right_1 = next_right_1.detach()
            current_left_2 = next_left_2.detach()
            current_right_2 = next_right_2.detach()
            current_pred = next_pred

        final_images = (current_left_1, current_right_1)
        final_pred = current_pred

        with torch.no_grad():
            bootstrap_value = self.critic(final_pred['disp_pred'].detach(), exposure_state.detach())

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
