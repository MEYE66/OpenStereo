# @Time    : 2026/3/31
# @Author  : OpenAI Codex
# @Modified: 2026/4/1 - Added advantage/reward normalization for stable training
import copy
import glob
import os
import time
from functools import partial

import torch
import torch.distributed as dist

from stereo.evaluation.metric_per_image import d1_metric, epe_metric, threshold_metric
from stereo.modeling.trainer_template import TrainerTemplate
from stereo.utils import common_utils
from stereo.utils.common_utils import color_map_tensorboard, write_tensorboard
from stereo.utils.lamb import Lamb
from stereo.utils.warmup import LinearWarmup


class A2CTrainerTemplate(TrainerTemplate):
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer, model):
        self.rl_cfg = self._resolve_rl_cfg(cfgs)
        self.rollout_steps = int(self._cfg_get(self.rl_cfg, 'ROLLOUT_STEPS', 3))
        self.gamma = float(self._cfg_get(self.rl_cfg, 'GAMMA', 0.99))
        self.gae_lambda = float(self._cfg_get(self.rl_cfg, 'GAE_LAMBDA', 0.95))
        self.entropy_weight = float(self._cfg_get(self.rl_cfg, 'ENTROPY_WEIGHT', 0.001))
        self.actor_loss_weight = float(self._cfg_get(self.rl_cfg, 'ACTOR_LOSS_WEIGHT', 1.0))
        self.critic_loss_weight = float(self._cfg_get(self.rl_cfg, 'CRITIC_LOSS_WEIGHT', 0.5))
        # self.disp_loss_weight = float(self._cfg_get(self.rl_cfg, 'DISP_LOSS_WEIGHT', 0.0))
        self.normalize_rewards = bool(self._cfg_get(self.rl_cfg, 'NORMALIZE_REWARDS', True))
        self.reward_norm_eps = float(self._cfg_get(self.rl_cfg, 'REWARD_NORM_EPS', 1e-6))
        self.train_stereo_from_start = bool(self._cfg_get(self.rl_cfg, 'TRAIN_STEREO', False))
        self.unfreeze_epoch = int(self._cfg_get(self.rl_cfg, 'UNFREEZE_EPOCH', -1))
        self.deterministic_eval = bool(self._cfg_get(self.rl_cfg, 'DETERMINISTIC_EVAL', True))
        self.debug_nan = bool(
            self._cfg_get(
                self.rl_cfg,
                'DEBUG_NAN',
                self._cfg_get(getattr(cfgs, 'MODEL', None), 'DEBUG_NAN', True),
            )
        )
        
        

        self.actor_optimizer = None
        self.actor_scheduler = None
        self.actor_warmup_scheduler = None
        self.critic_optimizer = None
        self.critic_scheduler = None
        self.critic_warmup_scheduler = None

        self.actor_scheduler_cfg = None
        self.critic_scheduler_cfg = None
        self.stereo_scheduler_cfg = None

        super().__init__(args, cfgs, local_rank, global_rank, logger, tb_writer, model)

    @staticmethod
    def _cfg_get(cfg, key, default=None):
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def _resolve_rl_cfg(self, cfgs):
        if hasattr(cfgs, 'RL'):
            return cfgs.RL
        if hasattr(cfgs, 'OPTIMIZATION') and hasattr(cfgs.OPTIMIZATION, 'RL'):
            return cfgs.OPTIMIZATION.RL
        return {}

    def _get_model_ref(self):
        return self.model.module if self.args.dist_mode else self.model

    def _get_component_cfg(self, name):
        cfg = self._cfg_get(self.cfgs.OPTIMIZATION, name, None)
        if cfg is None:
            raise KeyError('Missing OPTIMIZATION config field: {}'.format(name))
        return copy.deepcopy(cfg)

    def _build_optimizer(self, params, optimizer_cfg):
        params = list(params)
        if len(params) == 0:
            raise ValueError('Empty parameter list for optimizer build.')

        optimizer_name = self._cfg_get(optimizer_cfg, 'NAME', None)
        if optimizer_name is None:
            raise KeyError('Optimizer NAME must be set in OPTIMIZATION config.')
        if optimizer_name == 'Lamb':
            optimizer_cls = Lamb
        else:
            optimizer_cls = getattr(torch.optim, optimizer_name)

        valid_opt_arg = common_utils.get_valid_args(optimizer_cls, optimizer_cfg, ['name'])
        return optimizer_cls(params=params, **valid_opt_arg)

    def _build_scheduler(self, optimizer, scheduler_cfg):
        scheduler_name = self._cfg_get(scheduler_cfg, 'NAME', None)
        if scheduler_name is None:
            raise KeyError('Scheduler NAME must be set in OPTIMIZATION config.')
        if isinstance(scheduler_cfg, dict):
            scheduler_cfg['TOTAL_STEPS'] = self.max_iter
        else:
            scheduler_cfg.TOTAL_STEPS = self.max_iter
        scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_name)
        valid_sch_arg = common_utils.get_valid_args(scheduler_cls, scheduler_cfg, ['name', 'on_epoch'])
        return scheduler_cls(optimizer, **valid_sch_arg)

    def _schedulers_on_epoch(self):
        actor_on_epoch = bool(self._cfg_get(self.actor_scheduler_cfg, 'ON_EPOCH', False))
        critic_on_epoch = bool(self._cfg_get(self.critic_scheduler_cfg, 'ON_EPOCH', False))
        if actor_on_epoch != critic_on_epoch:
            raise ValueError('ACTOR/CRITIC scheduler ON_EPOCH must be consistent.')
        return actor_on_epoch

    def _build_component_warmup(self, optimizer, scheduler_cfg):
        last_step = (self.last_epoch + 1) * len(self.train_loader) - 1
        warmup_cfg = self._cfg_get(scheduler_cfg, 'WARMUP', None)
        warmup_steps = self._cfg_get(warmup_cfg, 'WARM_STEPS', 1) if warmup_cfg is not None else 1
        return LinearWarmup(optimizer, warmup_period=warmup_steps, last_step=last_step)


    def build_optimizer_and_scheduler(self):
        model_ref = self._get_model_ref()
        if not hasattr(model_ref, 'get_actor_parameters'):
            raise AttributeError('A2CTrainerTemplate requires model.get_actor_parameters().')
        if not hasattr(model_ref, 'get_critic_parameters'):
            raise AttributeError('A2CTrainerTemplate requires model.get_critic_parameters().')

        model_ref.set_stereo_requires_grad(False)

        actor_optimizer_cfg = self._get_component_cfg('ACTOR_OPTIMIZER')
        actor_scheduler_cfg = self._get_component_cfg('ACTOR_SCHEDULER')
        critic_optimizer_cfg = self._get_component_cfg('CRITIC_OPTIMIZER')
        critic_scheduler_cfg = self._get_component_cfg('CRITIC_SCHEDULER')

        self.actor_scheduler_cfg = copy.deepcopy(actor_scheduler_cfg)
        self.critic_scheduler_cfg = copy.deepcopy(critic_scheduler_cfg)
        self.stereo_scheduler_cfg = None

        self.actor_optimizer = self._build_optimizer(model_ref.get_actor_parameters(), actor_optimizer_cfg)
        self.actor_scheduler = self._build_scheduler(self.actor_optimizer, actor_scheduler_cfg)

        self.critic_optimizer = self._build_optimizer(model_ref.get_critic_parameters(), critic_optimizer_cfg)
        self.critic_scheduler = self._build_scheduler(self.critic_optimizer, critic_scheduler_cfg)

        return self.actor_optimizer, self.actor_scheduler

    def build_warmup(self):
        self.actor_warmup_scheduler = self._build_component_warmup(self.actor_optimizer, self.actor_scheduler_cfg)
        self.critic_warmup_scheduler = self._build_component_warmup(self.critic_optimizer, self.critic_scheduler_cfg)
        return self.actor_warmup_scheduler

    def resume_ckpt(self):
        self.logger.info('Resume from ckpt:%d' % self.cfgs.MODEL.CKPT)
        ckpt_path = str(os.path.join(self.args.ckpt_dir, 'checkpoint_epoch_%d.pth' % self.cfgs.MODEL.CKPT))
        checkpoint = torch.load(ckpt_path, map_location='cuda:%d' % self.local_rank)
        self.last_epoch = checkpoint['epoch']

        model_ref = self._get_model_ref()
        if 'model_state' in checkpoint:
            model_ref.load_state_dict(checkpoint['model_state'], strict=False)
        else:
            if 'actor_state' in checkpoint and hasattr(model_ref, 'actor'):
                model_ref.actor.load_state_dict(checkpoint['actor_state'], strict=False)
            if 'critic_state' in checkpoint and hasattr(model_ref, 'critic'):
                model_ref.critic.load_state_dict(checkpoint['critic_state'], strict=False)

        if 'optimizer_state' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        if 'scheduler_state' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
        if 'actor_optimizer_state' in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state'])
        if 'actor_scheduler_state' in checkpoint:
            self.actor_scheduler.load_state_dict(checkpoint['actor_scheduler_state'])
        if 'critic_optimizer_state' in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state'])
        if 'critic_scheduler_state' in checkpoint:
            self.critic_scheduler.load_state_dict(checkpoint['critic_scheduler_state'])
        if 'scaler_state' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state'])

    def _move_batch_to_device(self, data):
        for key, value in data.items():
            data[key] = value.to(self.local_rank) if torch.is_tensor(value) else value
        return data

    def _prepare_train_modes(self, ):
        model_ref = self._get_model_ref()
        self.model.train()
        model_ref.actor.train()
        model_ref.critic.train()
        model_ref.env.train()
        model_ref.set_stereo_train_mode(False)
        model_ref.set_stereo_requires_grad(False)
        if self.cfgs.OPTIMIZATION.get('FREEZE_BN', False):
            self.model = common_utils.freeze_bn(self.model)
            model_ref.set_stereo_train_mode(False)

    def train(self, current_epoch, tbar):
        # train_stereo = self._should_train_stereo(current_epoch)
        self._prepare_train_modes()
        if self.args.dist_mode:
            self.train_sampler.set_epoch(current_epoch)

        self.train_one_epoch(current_epoch=current_epoch, tbar=tbar)

        if self.args.dist_mode:
            dist.barrier()

        if self._schedulers_on_epoch():
            self.actor_scheduler.step()
            self.critic_scheduler.step()
            self.actor_warmup_scheduler.lrs = [group['lr'] for group in self.actor_optimizer.param_groups]
            self.critic_warmup_scheduler.lrs = [group['lr'] for group in self.critic_optimizer.param_groups]

    def _collect_transition_tensors(self, transitions):
        rewards = torch.stack([transition['reward'] for transition in transitions], dim=0)
        values = torch.stack([transition['value'] for transition in transitions], dim=0)
        log_probs = torch.stack([transition['log_prob'] for transition in transitions], dim=0)
        entropies = torch.stack([transition['entropy'] for transition in transitions], dim=0)
        return rewards, values, log_probs, entropies

    @staticmethod
    def _tensor_stats_message(tensor):
        detached = tensor.detach()
        finite_mask = torch.isfinite(detached)
        finite_count = int(finite_mask.sum().item())
        total_count = detached.numel()
        message = 'shape={} dtype={} device={} finite={}/{}'.format(
            tuple(detached.shape),
            detached.dtype,
            detached.device,
            finite_count,
            total_count,
        )
        if finite_count > 0:
            finite_values = detached[finite_mask].to(dtype=torch.float32)
            message += ' min={:.6g} max={:.6g} mean={:.6g} std={:.6g}'.format(
                finite_values.min().item(),
                finite_values.max().item(),
                finite_values.mean().item(),
                finite_values.std(unbiased=False).item(),
            )
        return message

    def _require_finite(self, name, tensor):
        if not self.debug_nan or not torch.is_tensor(tensor):
            return
        if not torch.isfinite(tensor).all():
            raise FloatingPointError(
                '{} contains non-finite values: {}'.format(
                    name,
                    self._tensor_stats_message(tensor),
                )
            )

    def _require_module_params_finite(self, module, module_name):
        if not self.debug_nan:
            return
        for name, param in module.named_parameters():
            self._require_finite(f'{module_name}.param.{name}', param)

    def _require_module_grads_finite(self, module, module_name):
        if not self.debug_nan:
            return
        for name, param in module.named_parameters():
            if param.grad is not None:
                self._require_finite(f'{module_name}.grad.{name}', param.grad)

    def _require_optimizer_state_finite(self, optimizer, optimizer_name):
        if not self.debug_nan:
            return
        for param_idx, state in enumerate(optimizer.state.values()):
            for state_name, value in state.items():
                if torch.is_tensor(value):
                    self._require_finite(
                        f'{optimizer_name}.state[{param_idx}].{state_name}',
                        value,
                    )

    @staticmethod
    def _transition_mean(transitions, key):
        if len(transitions) == 0 or key not in transitions[0]:
            return None
        values = [transition.get(key, None) for transition in transitions]
        if any(value is None for value in values):
            return None
        return torch.stack(values, dim=0).mean().item()

    @staticmethod
    def _transition_stats(transitions, key):
        if len(transitions) == 0 or key not in transitions[0]:
            return None
        values = [transition.get(key, None) for transition in transitions]
        if any(value is None for value in values):
            return None
        stacked = torch.stack(values, dim=0)
        abs_stacked = stacked.abs()
        return {
            'mean': stacked.mean().item(),
            'std': stacked.std(unbiased=False).item(),
            'min': stacked.min().item(),
            'max': stacked.max().item(),
            'abs_mean': abs_stacked.mean().item(),
            'abs_max': abs_stacked.max().item(),
        }

    def _compute_gae(self, rewards, values, bootstrap_value):
        advantages = torch.zeros_like(values)
        gae = torch.zeros_like(bootstrap_value)
        next_value = bootstrap_value

        for step_idx in range(rewards.shape[0] - 1, -1, -1):
            delta = rewards[step_idx] + self.gamma * next_value - values[step_idx]
            gae = delta + self.gamma * self.gae_lambda * gae
            advantages[step_idx] = gae
            next_value = values[step_idx]

        returns = advantages + values
        return returns, advantages

    def train_one_epoch(self, current_epoch, tbar):
        start_epoch = self.last_epoch + 1
        logger_iter_interval = self.cfgs.TRAINER.LOGGER_ITER_INTERVAL
        total_loss = 0.0
        model_ref = self._get_model_ref()

        train_loader_iter = iter(self.train_loader)
        for i in range(0, len(self.train_loader)):
            total_iter = current_epoch * len(self.train_loader) + i
            if total_iter >= self.max_iter:
                break

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            actor_lr = self.actor_optimizer.param_groups[0]['lr']
            critic_lr = self.critic_optimizer.param_groups[0]['lr']
            if self.debug_nan:
                self._require_module_params_finite(model_ref.actor, f'iter{total_iter}.actor.before')
                self._require_module_params_finite(model_ref.critic, f'iter{total_iter}.critic.before')

            start_timer = time.time()
            data = next(train_loader_iter)
            data = self._move_batch_to_device(data)
            data_timer = time.time()

            with torch.cuda.amp.autocast(enabled=self.cfgs.OPTIMIZATION.AMP):
                rollout = model_ref.rollout_episode(
                    batch=data,
                    steps=self.rollout_steps,
                )
                transitions = rollout['transitions']
                rewards, values, log_probs, entropies = self._collect_transition_tensors(transitions)
                bootstrap_value = rollout['bootstrap_value']
                raw_rewards = rewards
                self._require_finite(f'iter{total_iter}.rewards_raw', raw_rewards)
                self._require_finite(f'iter{total_iter}.values', values)
                self._require_finite(f'iter{total_iter}.log_probs', log_probs)
                self._require_finite(f'iter{total_iter}.entropies', entropies)
                self._require_finite(f'iter{total_iter}.bootstrap_value', bootstrap_value)
                
                # Standardize rewards to stabilize value function training
                if self.normalize_rewards:
                    reward_mean = raw_rewards.mean()
                    reward_std = raw_rewards.std(unbiased=False).clamp_min(self.reward_norm_eps)
                    rewards = (raw_rewards - reward_mean) / reward_std
                else:
                    rewards = raw_rewards
                
                ### TODO:  change gae function
                returns, advantages = self._compute_gae(rewards, values, bootstrap_value)
                self._require_finite(f'iter{total_iter}.rewards_used', rewards)
                self._require_finite(f'iter{total_iter}.returns', returns)
                self._require_finite(f'iter{total_iter}.advantages', advantages)

                # Standardize advantages to stabilize policy gradient
                adv_mean = advantages.mean()
                adv_std = advantages.std(unbiased=False).clamp_min(self.reward_norm_eps)
                normalized_advantages = (advantages - adv_mean) / adv_std
                self._require_finite(f'iter{total_iter}.normalized_advantages', normalized_advantages)

                actor_loss = -(log_probs * normalized_advantages.detach()).mean() - self.entropy_weight * entropies.mean()
                critic_loss = torch.mean((values - returns.detach()) ** 2)
                total_obj = self.actor_loss_weight * actor_loss + self.critic_loss_weight * critic_loss
                self._require_finite(f'iter{total_iter}.actor_loss', actor_loss)
                self._require_finite(f'iter{total_iter}.critic_loss', critic_loss)
                self._require_finite(f'iter{total_iter}.total_obj', total_obj)

                tb_info = {}
                infer_timer = time.time()

            self.scaler.scale(total_obj).backward()
            self.scaler.unscale_(self.actor_optimizer)
            self.scaler.unscale_(self.critic_optimizer)
            if self.debug_nan:
                self._require_module_grads_finite(model_ref.actor, f'iter{total_iter}.actor.before_clip')
                self._require_module_grads_finite(model_ref.critic, f'iter{total_iter}.critic.before_clip')

            if self.clip_grad is not None:
                self.clip_grad(self.model)
            if self.debug_nan:
                self._require_module_grads_finite(model_ref.actor, f'iter{total_iter}.actor.after_clip')
                self._require_module_grads_finite(model_ref.critic, f'iter{total_iter}.critic.after_clip')

            self.scaler.step(self.actor_optimizer)
            self.scaler.step(self.critic_optimizer)
            self.scaler.update()
            if self.debug_nan:
                self._require_module_params_finite(model_ref.actor, f'iter{total_iter}.actor.after_step')
                self._require_module_params_finite(model_ref.critic, f'iter{total_iter}.critic.after_step')
                self._require_optimizer_state_finite(self.actor_optimizer, f'iter{total_iter}.actor_optimizer')
                self._require_optimizer_state_finite(self.critic_optimizer, f'iter{total_iter}.critic_optimizer')

            if not self._schedulers_on_epoch():
                with self.actor_warmup_scheduler.dampening():
                    self.actor_scheduler.step()
                with self.critic_warmup_scheduler.dampening():
                    self.critic_scheduler.step()

            total_loss += total_obj.item()

            seed_loss_tensor = transitions[0]['stereo_loss_before']
            final_loss_tensor = transitions[-1]['stereo_loss_after']
            seed_loss = seed_loss_tensor.mean().item()
            final_loss = final_loss_tensor.mean().item()
            delta_loss = seed_loss - final_loss
            worsened_ratio = (final_loss_tensor > seed_loss_tensor).to(torch.float32).mean().item()

            reward_raw_mean = raw_rewards.mean().item()
            reward_raw_std = raw_rewards.std(unbiased=False).item()
            reward_used_mean = rewards.mean().item()
            reward_used_std = rewards.std(unbiased=False).item()
            exposure_abs_mean = torch.stack(
                [transition['action'].detach().abs().mean(dim=1) for transition in transitions],
                dim=0,
            ).mean().item()
            action_mean_mean = self._transition_mean(transitions, 'action_mean_mean')
            action_mean_std = self._transition_mean(transitions, 'action_mean_std')
            action_mean_min = self._transition_mean(transitions, 'action_mean_min')
            action_mean_max = self._transition_mean(transitions, 'action_mean_max')
            action_log_std_mean = self._transition_mean(transitions, 'action_log_std_mean')
            action_log_std_min = self._transition_mean(transitions, 'action_log_std_min')
            action_log_std_max = self._transition_mean(transitions, 'action_log_std_max')
            action_saturation = self._transition_mean(transitions, 'action_saturation')
            exposure_time_stats = self._transition_stats(transitions, 'exposure_time')
            gain_stats = self._transition_stats(transitions, 'gain')
            disp_finite_ratio = self._transition_mean(transitions, 'disp_finite_ratio')
            disp_valid_ratio = self._transition_mean(transitions, 'disp_valid_ratio')

            policy_message = ''
            if action_mean_mean is not None:
                policy_message += ' AMean:{:.4f}/{:.4f}[{:.4f},{:.4f}]'.format(
                    action_mean_mean,
                    action_mean_std if action_mean_std is not None else 0.0,
                    action_mean_min if action_mean_min is not None else 0.0,
                    action_mean_max if action_mean_max is not None else 0.0,
                )
            if action_log_std_mean is not None:
                policy_message += ' LogStd:{:.4f}[{:.4f},{:.4f}]'.format(
                    action_log_std_mean,
                    action_log_std_min if action_log_std_min is not None else 0.0,
                    action_log_std_max if action_log_std_max is not None else 0.0,
                )
            if action_saturation is not None:
                policy_message += ' Sat:{:.3f}'.format(action_saturation)
            if exposure_time_stats is not None and gain_stats is not None:
                policy_message += ' T:{:.2f}-{:.2f} G:{:.2f}-{:.2f}'.format(
                    exposure_time_stats['min'],
                    exposure_time_stats['max'],
                    gain_stats['min'],
                    gain_stats['max'],
                )
            if disp_finite_ratio is not None and disp_valid_ratio is not None:
                policy_message += ' DFinite:{:.3f} DValid:{:.3f}'.format(
                    disp_finite_ratio,
                    disp_valid_ratio,
                )
            if self.debug_nan:
                policy_message += ' Finite:1'

            trained_time_past_all = tbar.format_dict['elapsed']
            single_iter_second = trained_time_past_all / (total_iter + 1 - start_epoch * len(self.train_loader))
            remaining_second_all = single_iter_second * (self.total_epochs * len(self.train_loader) - total_iter - 1)
            if total_iter % logger_iter_interval == 0:
                message = (
                    'Training Epoch:{:>2d}/{} Iter:{:>4d}/{} '
                    'Loss:{:#.6g}({:#.6g}) '
                    'ALR:{:.4e} CLR:{:.4e} '
                    'ALoss:{:.4f} CLoss:{:.4f} '
                    'Reward:{:.4f} Ent:{:.4f} ExpAbs:{:.4f}{} Worse:{:.3f} '
                    'SeedLoss:{:.4f} FinalLoss:{:.4f} DeltaLoss:{:.4f} '
                    'DataTime:{:.2f} InferTime:{:.2f}ms '
                    'Time cost: {}/{}'
                ).format(
                    current_epoch,
                    self.total_epochs,
                    i,
                    len(self.train_loader),
                    total_obj.item(),
                    total_loss / (i + 1),
                    actor_lr,
                    critic_lr,
                    actor_loss.item(),
                    critic_loss.item(),
                    reward_used_mean,
                    entropies.mean().item(),
                    exposure_abs_mean,
                    policy_message,
                    worsened_ratio,
                    seed_loss,
                    final_loss,
                    delta_loss,
                    data_timer - start_timer,
                    (infer_timer - data_timer) * 1000,
                    tbar.format_interval(trained_time_past_all),
                    tbar.format_interval(remaining_second_all),
                )
                self.logger.info(message)

            tb_info.update({
                'scalar/train/loss_total': total_obj.item(),
                'scalar/train/loss_actor': actor_loss.item(),
                'scalar/train/loss_critic': critic_loss.item(),
                'scalar/train/return': returns.mean().item(),
                'scalar/train/advantage': advantages.mean().item(),
                'scalar/train/advantage_normalized': normalized_advantages.mean().item(),
                'scalar/train/advantage_std': advantages.std(unbiased=False).item(),
                'scalar/train/reward': reward_used_mean,
                'scalar/train/reward_std': reward_used_std,
                'scalar/train/reward_raw': reward_raw_mean,
                'scalar/train/reward_raw_std': reward_raw_std,
                'scalar/train/worsened_ratio': worsened_ratio,
                'scalar/train/entropy': entropies.mean().item(),
                'scalar/train/action_abs_mean': exposure_abs_mean,
                'scalar/train/exposure_abs_mean': exposure_abs_mean,
                'scalar/train/seed_stereo_loss': seed_loss,
                'scalar/train/final_stereo_loss': final_loss,
                'scalar/train/delta_stereo_loss': delta_loss,
                'scalar/train/lr_actor': actor_lr,
                'scalar/train/lr_critic': critic_lr,
            })
            if self.debug_nan:
                tb_info['scalar/train/finite_ok'] = 1.0
            if action_mean_mean is not None:
                tb_info['scalar/train/action_mean_mean'] = action_mean_mean
                tb_info['scalar/train/action_mean_std'] = action_mean_std
                tb_info['scalar/train/action_mean_min'] = action_mean_min
                tb_info['scalar/train/action_mean_max'] = action_mean_max
            if action_log_std_mean is not None:
                tb_info['scalar/train/action_log_std_mean'] = action_log_std_mean
                tb_info['scalar/train/action_log_std_min'] = action_log_std_min
                tb_info['scalar/train/action_log_std_max'] = action_log_std_max
            if action_saturation is not None:
                tb_info['scalar/train/action_saturation_ratio'] = action_saturation
            if exposure_time_stats is not None:
                tb_info['scalar/train/exposure_time_mean'] = exposure_time_stats['mean']
                tb_info['scalar/train/exposure_time_min'] = exposure_time_stats['min']
                tb_info['scalar/train/exposure_time_max'] = exposure_time_stats['max']
            if gain_stats is not None:
                tb_info['scalar/train/gain_mean'] = gain_stats['mean']
                tb_info['scalar/train/gain_min'] = gain_stats['min']
                tb_info['scalar/train/gain_max'] = gain_stats['max']
            if disp_finite_ratio is not None:
                tb_info['scalar/train/disp_finite_ratio'] = disp_finite_ratio
            if disp_valid_ratio is not None:
                tb_info['scalar/train/disp_valid_ratio'] = disp_valid_ratio

            if self.cfgs.TRAINER.TRAIN_VISUALIZATION:
                final_left, final_right = rollout['final_images']
                tb_info['image/train/image'] = torch.cat([final_left[0], final_right[0]], dim=1)
                tb_info['image/train/disp'] = color_map_tensorboard(
                    data['disp'][0],
                    rollout['final_pred']['disp_pred'].squeeze(1)[0],
                )

            if total_iter % logger_iter_interval == 0 and self.local_rank == 0 and self.tb_writer is not None:
                write_tensorboard(self.tb_writer, tb_info, total_iter)

    @torch.no_grad()
    def eval_one_epoch(self, current_epoch):
        metric_func_dict = {
            'epe': epe_metric,
            'd1_all': d1_metric,
            'thres_1': partial(threshold_metric, threshold=1),
            'thres_2': partial(threshold_metric, threshold=2),
            'thres_3': partial(threshold_metric, threshold=3),
        }

        evaluator_cfgs = self.cfgs.EVALUATOR
        local_rank = self.local_rank
        model_ref = self._get_model_ref()

        epoch_metrics = {}
        for metric_name in list(evaluator_cfgs.METRIC):
            epoch_metrics[metric_name] = {'indexes': [], 'values': []}

        for i, data in enumerate(self.eval_loader):
            for key, value in data.items():
                data[key] = value.to(local_rank) if torch.is_tensor(value) else value

            infer_start = time.time()
            rollout = model_ref.rollout_episode(
                batch=data,
                steps=self.rollout_steps,
                deterministic=self.deterministic_eval,
            )
            infer_time = time.time() - infer_start

            model_pred = rollout['final_pred']
            disp_pred = model_pred['disp_pred']
            disp_gt = data['disp']
            mask = torch.isfinite(disp_gt) & (disp_gt < evaluator_cfgs.MAX_DISP) & (disp_gt > 0)
            if 'occ_mask' in data and evaluator_cfgs.get('APPLY_OCC_MASK', False):
                mask = mask & ~data['occ_mask'].to(torch.bool)
            disp_gt = torch.nan_to_num(disp_gt, nan=0.0, posinf=0.0, neginf=0.0)

            indexes = data['index'].tolist()
            for metric_name in evaluator_cfgs.METRIC:
                if metric_name not in metric_func_dict:
                    raise ValueError("Unknown metric: {}".format(metric_name))
                metric_func = metric_func_dict[metric_name]
                result = metric_func(disp_pred.squeeze(1), disp_gt, mask)
                epoch_metrics[metric_name]['indexes'].extend(indexes)
                epoch_metrics[metric_name]['values'].extend(result.tolist())

            if i % self.cfgs.TRAINER.LOGGER_ITER_INTERVAL == 0:
                message = (
                    'Evaluating Epoch:{:>2d} Iter:{:>4d}/{} '
                    'InferTime: {:.2f}ms'
                ).format(
                    current_epoch,
                    i,
                    len(self.eval_loader),
                    infer_time * 1000,
                )
                self.logger.info(message)

                if self.cfgs.TRAINER.EVAL_VISUALIZATION and self.tb_writer is not None:
                    final_left, final_right = rollout['final_images']
                    tb_info = {
                        'image/eval/image': torch.cat([final_left[0], final_right[0]], dim=1),
                        'image/eval/disp': color_map_tensorboard(data['disp'][0], model_pred['disp_pred'].squeeze(1)[0]),
                    }
                    write_tensorboard(self.tb_writer, tb_info, current_epoch * len(self.eval_loader) + i)

        if self.args.dist_mode:
            dist.barrier()
            self.logger.info("Start reduce metrics.")
            for metric_name in epoch_metrics.keys():
                indexes = torch.tensor(epoch_metrics[metric_name]['indexes']).to(local_rank)
                values = torch.tensor(epoch_metrics[metric_name]['values']).to(local_rank)
                gathered_indexes = [torch.zeros_like(indexes) for _ in range(dist.get_world_size())]
                gathered_values = [torch.zeros_like(values) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_indexes, indexes)
                dist.all_gather(gathered_values, values)
                unique_dict = {}
                for key, value in zip(torch.cat(gathered_indexes, dim=0).tolist(), torch.cat(gathered_values, dim=0).tolist()):
                    if key not in unique_dict:
                        unique_dict[key] = value
                epoch_metrics[metric_name]['indexes'] = list(unique_dict.keys())
                epoch_metrics[metric_name]['values'] = list(unique_dict.values())

        results = {}
        for metric_name in epoch_metrics.keys():
            results[metric_name] = torch.tensor(epoch_metrics[metric_name]['values']).mean()

        if local_rank == 0 and self.tb_writer is not None:
            tb_info = {}
            for metric_name, metric_value in results.items():
                tb_info[f'scalar/val/{metric_name}'] = metric_value.item()
            write_tensorboard(self.tb_writer, tb_info, current_epoch)

        self.logger.info(f"Epoch {current_epoch} metrics: {results}")

    def save_ckpt(self, current_epoch):
        should_save = (
            current_epoch % self.cfgs.TRAINER.CKPT_SAVE_INTERVAL == 0
            or current_epoch == self.total_epochs - 1
        )
        if should_save and self.global_rank == 0:
            ckpt_list = glob.glob(os.path.join(self.args.ckpt_dir, 'checkpoint_epoch_*.pth'))
            ckpt_list.sort(key=os.path.getmtime)
            if len(ckpt_list) >= self.cfgs.TRAINER.MAX_CKPT_SAVE_NUM:
                remove_count = len(ckpt_list) - self.cfgs.TRAINER.MAX_CKPT_SAVE_NUM + 1
                for cur_file_idx in range(0, remove_count):
                    os.remove(ckpt_list[cur_file_idx])

            model_ref = self._get_model_ref()
            state = {
                'epoch': current_epoch,
                'model_state': model_ref.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'scheduler_state': self.scheduler.state_dict(),
                'actor_optimizer_state': self.actor_optimizer.state_dict(),
                'actor_scheduler_state': self.actor_scheduler.state_dict(),
                'critic_optimizer_state': self.critic_optimizer.state_dict(),
                'critic_scheduler_state': self.critic_scheduler.state_dict(),
                'scaler_state': self.scaler.state_dict(),
            }
            if hasattr(model_ref, 'get_component_state_dicts'):
                state.update(model_ref.get_component_state_dicts())

            ckpt_name = os.path.join(self.args.ckpt_dir, 'checkpoint_epoch_%d.pth' % current_epoch)
            torch.save(state, ckpt_name)

        if self.args.dist_mode:
            dist.barrier()
