import sys
from pathlib import Path

import torch
import torch.nn.functional as F


repo_root = Path(__file__).resolve().parents[4]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    from stereo.modeling.models.raftstereo.raft_stereo import RAFTStereo
    from stereo.modeling.models.adaptiveaenet.exposure_fusion import MertensExposureFusion
    from stereo.modeling.models.adaptiveaenet.submodules import (
        A2CActor,
        AbsoluteExposureEnv,
        DispCritic,
        require_finite_tensor,
    )
except ModuleNotFoundError:
    from ..raftstereo.raft_stereo import RAFTStereo
    from .exposure_fusion import MertensExposureFusion
    from .submodules import A2CActor, AbsoluteExposureEnv, DispCritic, require_finite_tensor


def _cfg_get(cfgs, key, default=None):
    if isinstance(cfgs, dict):
        return cfgs.get(key, default)
    return getattr(cfgs, key, default)


def _cfg_pair(cfgs, key, default):
    value = _cfg_get(cfgs, key, default)
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


class AdaptiveAENet(RAFTStereo):
    def __init__(self, cfgs):
        cfgs = _adapt_raft_cfgs(cfgs)
        super().__init__(cfgs)

        self.rollout_steps = int(_cfg_get(cfgs, 'ROLLOUT_STEPS', 3))
        self.time_limits = _cfg_pair(cfgs, 'TIME_LIMITS', default=(1.0, 20.0))
        self.gain_limits = _cfg_pair(cfgs, 'GAIN_LIMITS', default=(1.0, 16.0))
        self.init_time = _cfg_pair(cfgs, 'INIT_TIME', default=(8.0, 8.0))
        self.init_gain = _cfg_pair(cfgs, 'INIT_GAIN', default=(4.0, 4.0))
        self.time_classes = _cfg_get(cfgs, 'TIME_CLASSES', [1, 2, 3, 5, 8, 10, 15, 20])
        self.gain_classes = _cfg_get(cfgs, 'GAIN_CLASSES', [1, 2, 4, 8, 16])
        self.nbits = int(_cfg_get(cfgs, 'NBITS', 10))

        feature_dim = int(_cfg_get(cfgs, 'RL_FEATURE_DIM', 256))
        hidden_dim = int(_cfg_get(cfgs, 'RL_HIDDEN_DIM', feature_dim))
        input_size = int(_cfg_get(cfgs, 'RL_INPUT_SIZE', 128))
        self.debug_nan = bool(_cfg_get(cfgs, 'DEBUG_NAN', False))

        self.disp_reward_weight = float(_cfg_get(cfgs, 'DISP_REWARD_WEIGHT', 1.0))
        self.reward_disp_clip = float(_cfg_get(cfgs, 'REWARD_DISP_CLIP', 1.0))
        self.exposure_fusion = MertensExposureFusion(
            n_levels=int(_cfg_get(cfgs, 'EXPOSURE_FUSION_LEVELS', 4)),
            w_cont=float(_cfg_get(cfgs, 'EXPOSURE_FUSION_W_CONT', 1.0)),
            w_sat=float(_cfg_get(cfgs, 'EXPOSURE_FUSION_W_SAT', 1.0)),
            w_exp=float(_cfg_get(cfgs, 'EXPOSURE_FUSION_W_EXP', 1.0)),
            well_exposed_sigma=float(_cfg_get(cfgs, 'EXPOSURE_FUSION_SIGMA', 0.2)),
            clamp_output=bool(_cfg_get(cfgs, 'EXPOSURE_FUSION_CLAMP_OUTPUT', True)),
        )

        self.env = AbsoluteExposureEnv(
            time_limits=self.time_limits,
            gain_limits=self.gain_limits,
            init_time=self.init_time,
            init_gain=self.init_gain,
            nbits=self.nbits,
        )
        self.actor = A2CActor(
            image_channels=12,
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            input_size=input_size,
            time_limits=self.time_limits,
            gain_limits=self.gain_limits,
            time_classes=self.time_classes,
            gain_classes=self.gain_classes,
            debug_checks=self.debug_nan,
        )
        self.critic = DispCritic(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            input_size=input_size,
        )

    def _stereo_modules(self):
        stereo_modules = [self.cnet, self.update_block, self.context_zqr_convs]
        if self.shared_backbone:
            stereo_modules.append(self.conv2)
        else:
            stereo_modules.append(self.fnet)
        return stereo_modules

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
        for module_name in ['cnet', 'update_block', 'context_zqr_convs']:
            module = getattr(self, module_name)
            for key, value in module.state_dict().items():
                stereo_state[f'{module_name}.{key}'] = value
        branch_name = 'conv2' if self.shared_backbone else 'fnet'
        branch_module = getattr(self, branch_name)
        for key, value in branch_module.state_dict().items():
            stereo_state[f'{branch_name}.{key}'] = value
        return {
            'stereo_state': stereo_state,
            'actor_state': self.actor.state_dict(),
            'critic_state': self.critic.state_dict(),
        }

    def _forward_stereo_dual(self, left_1, right_1, left_2, right_2, _state):
        left_fused = self.exposure_fusion(left_1, left_2)
        right_fused = self.exposure_fusion(right_1, right_2)
        pred = RAFTStereo.forward(
            self,
            {
                'left': left_fused,
                'right': right_fused,
            },
        )
        return pred

    @staticmethod
    def _masked_mean_per_sample(values, valid_mask):
        masked_values = values.masked_fill(~valid_mask, float('nan'))
        per_sample = torch.nanmean(masked_values.flatten(1), dim=1)
        return torch.nan_to_num(per_sample, nan=0.0, posinf=0.0, neginf=0.0)

    def _get_disp_target(self, batch):
        disp_gt = batch.get('disp', batch.get('disp_1'))
        if disp_gt is None:
            raise KeyError('AdaptiveAENet requires disp or disp_1 in the batch.')
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
                'AdaptiveAENet requires dual-frame inputs. Missing keys: {}'.format(
                    ', '.join(missing_keys)
                )
            )

    def _check_finite(self, name, tensor):
        if self.debug_nan:
            require_finite_tensor(name, tensor)

    def _check_pred_finite(self, name, model_pred):
        if not self.debug_nan:
            return
        if 'disp_pred' in model_pred:
            self._check_finite(f'{name}.disp_pred', model_pred['disp_pred'])
        for idx, disp_pred in enumerate(model_pred.get('disp_preds', [])):
            self._check_finite(f'{name}.disp_preds[{idx}]', disp_pred)

    def _check_batch_finite(self, batch):
        if not self.debug_nan:
            return
        for key in ['left_1', 'right_1', 'left_2', 'right_2']:
            self._check_finite(f'batch.{key}', batch[key])
        disp_target = self._get_disp_target(batch)
        finite_count = torch.isfinite(disp_target).flatten(1).sum(dim=1)
        if torch.any(finite_count == 0):
            raise FloatingPointError('batch.disp has a sample with no finite disparity values.')

    def _get_disp_target_stats(self, batch):
        disp_target = self._get_disp_target(batch)
        valid = self._get_valid_mask(batch, disp_target)
        finite = torch.isfinite(disp_target)
        flat_size = disp_target.flatten(1).shape[1]
        return {
            'disp_finite_ratio': finite.flatten(1).to(torch.float32).sum(dim=1) / float(flat_size),
            'disp_valid_ratio': valid.flatten(1).to(torch.float32).sum(dim=1) / float(flat_size),
        }

    def _add_actor_stats(self, transition):
        for key, value in getattr(self.actor, 'last_policy_stats', {}).items():
            transition[key] = value.detach()
        return transition

    def rollout_episode(self, batch, steps, deterministic=False):
        self._require_dual_frame_batch(batch)
        self._check_batch_finite(batch)
        left_radiance_1 = batch['left_1']
        right_radiance_1 = batch['right_1']
        left_radiance_2 = batch['left_2']
        right_radiance_2 = batch['right_2']
        total_steps = max(int(steps), 1)
        disp_target_stats = self._get_disp_target_stats(batch)

        with torch.no_grad():
            state = self.env.get_initial_state(left_radiance_1)
            self._check_finite('rollout.initial_state', state)
            current_left_1, current_right_1, current_left_2, current_right_2 = self.env.render_dual_frames(
                left_radiance_1,
                right_radiance_1,
                left_radiance_2,
                right_radiance_2,
                state,
            )
            self._check_finite('rollout.seed_left_1', current_left_1)
            self._check_finite('rollout.seed_right_1', current_right_1)
            self._check_finite('rollout.seed_left_2', current_left_2)
            self._check_finite('rollout.seed_right_2', current_right_2)
            current_pred = self._forward_stereo_dual(
                current_left_1,
                current_right_1,
                current_left_2,
                current_right_2,
                state,
            )
            self._check_pred_finite('rollout.seed_pred', current_pred)

        seed_images = (current_left_1, current_right_1)
        seed_pred = current_pred

        transitions = []
        for step_idx in range(total_steps):
            action, log_prob, entropy = self.actor(
                current_left_1.detach(),
                current_right_1.detach(),
                current_left_2.detach(),
                current_right_2.detach(),
                state.detach(),
                deterministic=deterministic,
            )
            self._check_finite(f'rollout.step{step_idx}.action', action)
            self._check_finite(f'rollout.step{step_idx}.log_prob', log_prob)
            self._check_finite(f'rollout.step{step_idx}.entropy', entropy)
            value = self.critic(current_pred['disp_pred'].detach())
            self._check_finite(f'rollout.step{step_idx}.value', value)
            next_state = self.env.apply_action(action)
            self._check_finite(f'rollout.step{step_idx}.next_state', next_state)

            with torch.no_grad():
                next_left_1, next_right_1, next_left_2, next_right_2 = self.env.render_dual_frames(
                    left_radiance_1,
                    right_radiance_1,
                    left_radiance_2,
                    right_radiance_2,
                    next_state.detach(),
                )
                self._check_finite(f'rollout.step{step_idx}.next_left_1', next_left_1)
                self._check_finite(f'rollout.step{step_idx}.next_right_1', next_right_1)
                self._check_finite(f'rollout.step{step_idx}.next_left_2', next_left_2)
                self._check_finite(f'rollout.step{step_idx}.next_right_2', next_right_2)
                next_pred = self._forward_stereo_dual(
                    next_left_1,
                    next_right_1,
                    next_left_2,
                    next_right_2,
                    next_state,
                )
                self._check_pred_finite(f'rollout.step{step_idx}.next_pred', next_pred)

            loss_before = self._compute_stereo_loss(current_pred, batch)
            loss_after = self._compute_stereo_loss(next_pred, batch)
            self._check_finite(f'rollout.step{step_idx}.loss_before', loss_before)
            self._check_finite(f'rollout.step{step_idx}.loss_after', loss_after)
            reward_disp_raw = self._shape_disp_reward(loss_before, loss_after)
            reward_disp = self._clip_reward_component(reward_disp_raw, self.reward_disp_clip)
            reward = self.disp_reward_weight * reward_disp
            self._check_finite(f'rollout.step{step_idx}.reward_disp_raw', reward_disp_raw)
            self._check_finite(f'rollout.step{step_idx}.reward_disp', reward_disp)
            self._check_finite(f'rollout.step{step_idx}.reward', reward)

            transition = {
                'value': value,
                'log_prob': log_prob,
                'entropy': entropy,
                'action': next_state.detach(),
                'raw_action': action.detach(),
                'state': state.detach(),
                'next_state': next_state.detach(),
                'reward': reward.detach(),
                'reward_disp': reward_disp.detach(),
                'reward_disp_raw': reward_disp_raw.detach(),
                'stereo_loss_before': loss_before.detach(),
                'stereo_loss_after': loss_after.detach(),
                'exposure_time': next_state[:, [0, 2]].detach(),
                'gain': next_state[:, [1, 3]].detach(),
                'exposure_time_1': next_state[:, 0].detach(),
                'gain_1': next_state[:, 1].detach(),
                'exposure_time_2': next_state[:, 2].detach(),
                'gain_2': next_state[:, 3].detach(),
            }
            for key, value in disp_target_stats.items():
                transition[key] = value.detach()
            transitions.append(self._add_actor_stats(transition))

            state = next_state.detach()
            current_left_1 = next_left_1.detach()
            current_right_1 = next_right_1.detach()
            current_left_2 = next_left_2.detach()
            current_right_2 = next_right_2.detach()
            current_pred = next_pred

        final_images = (current_left_1, current_right_1)
        final_pred = current_pred

        with torch.no_grad():
            bootstrap_value = self.critic(final_pred['disp_pred'].detach())
            self._check_finite('rollout.bootstrap_value', bootstrap_value)

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
