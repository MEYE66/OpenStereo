# @Time    : 2026/3/31
# @Author  : OpenAI Codex
# @Modified: 2026/4/1 - Adaptive exposure reward trainer
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


class A2CTrainerTemplateReward(TrainerTemplate):
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer, model):
        self.rl_cfg = self._resolve_rl_cfg(cfgs)
        self.rollout_steps = int(self._cfg_get(self.rl_cfg, 'ROLLOUT_STEPS', 3))
        self.gamma = float(self._cfg_get(self.rl_cfg, 'GAMMA', 0.99))
        self.gae_lambda = float(self._cfg_get(self.rl_cfg, 'GAE_LAMBDA', 0.95))
        self.action_penalty_weight = float(self._cfg_get(self.rl_cfg, 'ACTION_PENALTY_WEIGHT', 0.0))
        self.entropy_weight = float(self._cfg_get(self.rl_cfg, 'ENTROPY_WEIGHT', 0.001))
        self.actor_loss_weight = float(self._cfg_get(self.rl_cfg, 'ACTOR_LOSS_WEIGHT', 1.0))
        self.critic_loss_weight = float(self._cfg_get(self.rl_cfg, 'CRITIC_LOSS_WEIGHT', 0.5))
        self.disp_loss_weight = float(self._cfg_get(self.rl_cfg, 'DISP_LOSS_WEIGHT', 0.0))
        self.normalize_rewards = bool(self._cfg_get(self.rl_cfg, 'NORMALIZE_REWARDS', False))
        self.reward_norm_eps = float(self._cfg_get(self.rl_cfg, 'REWARD_NORM_EPS', 1e-6))
        self.deterministic_eval = bool(self._cfg_get(self.rl_cfg, 'DETERMINISTIC_EVAL', True))

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
        stereo_on_epoch = bool(self._cfg_get(self.stereo_scheduler_cfg, 'ON_EPOCH', False))
        if actor_on_epoch != critic_on_epoch or actor_on_epoch != stereo_on_epoch:
            raise ValueError('ACTOR/CRITIC/STEREO scheduler ON_EPOCH must be consistent.')
        return actor_on_epoch

    def _build_component_warmup(self, optimizer, scheduler_cfg):
        last_step = (self.last_epoch + 1) * len(self.train_loader) - 1
        warmup_cfg = self._cfg_get(scheduler_cfg, 'WARMUP', None)
        warmup_steps = self._cfg_get(warmup_cfg, 'WARM_STEPS', 1) if warmup_cfg is not None else 1
        return LinearWarmup(optimizer, warmup_period=warmup_steps, last_step=last_step)

    def build_optimizer_and_scheduler(self):
        model_ref = self._get_model_ref()
        if not hasattr(model_ref, 'get_actor_parameters'):
            raise AttributeError('A2CTrainerTemplateReward requires model.get_actor_parameters().')
        if not hasattr(model_ref, 'get_critic_parameters'):
            raise AttributeError('A2CTrainerTemplateReward requires model.get_critic_parameters().')
        if not hasattr(model_ref, 'get_stereo_parameters'):
            raise AttributeError('A2CTrainerTemplateReward requires model.get_stereo_parameters().')

        model_ref.set_stereo_requires_grad(False)

        actor_optimizer_cfg = self._get_component_cfg('ACTOR_OPTIMIZER')
        actor_scheduler_cfg = self._get_component_cfg('ACTOR_SCHEDULER')
        critic_optimizer_cfg = self._get_component_cfg('CRITIC_OPTIMIZER')
        critic_scheduler_cfg = self._get_component_cfg('CRITIC_SCHEDULER')
        stereo_optimizer_cfg = self._get_component_cfg('STEREO_OPTIMIZER')
        stereo_scheduler_cfg = self._get_component_cfg('STEREO_SCHEDULER')

        self.actor_scheduler_cfg = copy.deepcopy(actor_scheduler_cfg)
        self.critic_scheduler_cfg = copy.deepcopy(critic_scheduler_cfg)
        self.stereo_scheduler_cfg = copy.deepcopy(stereo_scheduler_cfg)

        self.actor_optimizer = self._build_optimizer(model_ref.get_actor_parameters(), actor_optimizer_cfg)
        self.actor_scheduler = self._build_scheduler(self.actor_optimizer, actor_scheduler_cfg)

        self.critic_optimizer = self._build_optimizer(model_ref.get_critic_parameters(), critic_optimizer_cfg)
        self.critic_scheduler = self._build_scheduler(self.critic_optimizer, critic_scheduler_cfg)

        stereo_optimizer = self._build_optimizer(model_ref.get_stereo_parameters(), stereo_optimizer_cfg)
        stereo_scheduler = self._build_scheduler(stereo_optimizer, stereo_scheduler_cfg)
        return stereo_optimizer, stereo_scheduler

    def build_warmup(self):
        self.actor_warmup_scheduler = self._build_component_warmup(self.actor_optimizer, self.actor_scheduler_cfg)
        self.critic_warmup_scheduler = self._build_component_warmup(self.critic_optimizer, self.critic_scheduler_cfg)
        return self._build_component_warmup(self.optimizer, self.stereo_scheduler_cfg)

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

    def _prepare_train_modes(self):
        model_ref = self._get_model_ref()
        self.model.eval()
        model_ref.actor.train()
        model_ref.critic.train()
        model_ref.env.train()
        model_ref.set_stereo_train_mode(False)
        model_ref.set_stereo_requires_grad(False)
        if self.cfgs.OPTIMIZATION.get('FREEZE_BN', False):
            self.model = common_utils.freeze_bn(self.model)
            model_ref.set_stereo_train_mode(False)

    def train(self, current_epoch, tbar):
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

            start_timer = time.time()
            data = next(train_loader_iter)
            data = self._move_batch_to_device(data)
            data_timer = time.time()

            with torch.cuda.amp.autocast(enabled=self.cfgs.OPTIMIZATION.AMP):
                rollout = model_ref.rollout_episode(
                    batch=data,
                    steps=self.rollout_steps,
                    action_penalty_weight=self.action_penalty_weight,
                )
                transitions = rollout['transitions']
                rewards, values, log_probs, entropies = self._collect_transition_tensors(transitions)
                bootstrap_value = rollout['bootstrap_value']

                if self.normalize_rewards:
                    reward_mean = rewards.mean()
                    reward_std = rewards.std(unbiased=False).clamp_min(self.reward_norm_eps)
                    rewards = (rewards - reward_mean) / reward_std

                returns, advantages = self._compute_gae(rewards, values, bootstrap_value)

                adv_mean = advantages.mean()
                adv_std = advantages.std(unbiased=False).clamp_min(self.reward_norm_eps)
                normalized_advantages = (advantages - adv_mean) / adv_std

                actor_loss = -(log_probs * normalized_advantages.detach()).mean() - self.entropy_weight * entropies.mean()
                critic_loss = torch.mean((values - returns.detach()) ** 2)
                total_obj = self.actor_loss_weight * actor_loss + self.critic_loss_weight * critic_loss

                tb_info = {}
                infer_timer = time.time()

            self.scaler.scale(total_obj).backward()
            self.scaler.unscale_(self.actor_optimizer)
            self.scaler.unscale_(self.critic_optimizer)

            if self.clip_grad is not None:
                self.clip_grad(self.model)

            self.scaler.step(self.actor_optimizer)
            self.scaler.step(self.critic_optimizer)
            self.scaler.update()

            if not self._schedulers_on_epoch():
                with self.actor_warmup_scheduler.dampening():
                    self.actor_scheduler.step()
                with self.critic_warmup_scheduler.dampening():
                    self.critic_scheduler.step()

            total_loss += total_obj.item()

            seed_epe = transitions[0]['epe_before'].mean().item()
            final_epe = transitions[-1]['epe_after'].mean().item()
            reward_mean = rewards.mean().item()
            action_mean = torch.stack(
                [transition['action'].detach().abs().mean(dim=1) for transition in transitions],
                dim=0,
            ).mean().item()

            reward_disp_mean = torch.stack([transition['reward_disp'] for transition in transitions], dim=0).mean().item()
            reward_adapt_mean = torch.stack([transition['reward_adapt'] for transition in transitions], dim=0).mean().item()
            scene_dr_mean = torch.stack([transition['scene_dr'] for transition in transitions], dim=0).mean().item()
            ev_gap_before_mean = torch.stack([transition['ev_gap_before'] for transition in transitions], dim=0).mean().item()
            ev_gap_after_mean = torch.stack([transition['ev_gap_after'] for transition in transitions], dim=0).mean().item()
            ev_gap_target_mean = torch.stack([transition['ev_gap_target'] for transition in transitions], dim=0).mean().item()

            trained_time_past_all = tbar.format_dict['elapsed']
            single_iter_second = trained_time_past_all / (total_iter + 1 - start_epoch * len(self.train_loader))
            remaining_second_all = single_iter_second * (self.total_epochs * len(self.train_loader) - total_iter - 1)
            if total_iter % logger_iter_interval == 0:
                message = (
                    'Training Epoch:{:>2d}/{} Iter:{:>4d}/{} '
                    'Loss:{:#.6g}({:#.6g}) '
                    'ALR:{:.4e} CLR:{:.4e} '
                    'ALoss:{:.4f} CLoss:{:.4f} '
                    'RDisp:{:.4f} RAdapt:{:.4f} '
                    'GapB:{:.4f} GapA:{:.4f} GapT:{:.4f} DR:{:.4f} '
                    'SeedEPE:{:.4f} FinalEPE:{:.4f} '
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
                    reward_disp_mean,
                    reward_adapt_mean,
                    ev_gap_before_mean,
                    ev_gap_after_mean,
                    ev_gap_target_mean,
                    scene_dr_mean,
                    seed_epe,
                    final_epe,
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
                'scalar/train/advantage_std': advantages.std().item(),
                'scalar/train/reward': rewards.mean().item(),
                'scalar/train/reward_std': rewards.std().item(),
                'scalar/train/reward_disp': reward_disp_mean,
                'scalar/train/reward_adapt': reward_adapt_mean,
                'scalar/train/scene_dr': scene_dr_mean,
                'scalar/train/ev_gap_before': ev_gap_before_mean,
                'scalar/train/ev_gap_after': ev_gap_after_mean,
                'scalar/train/ev_gap_target': ev_gap_target_mean,
                'scalar/train/entropy': entropies.mean().item(),
                'scalar/train/action_abs_mean': action_mean,
                'scalar/train/seed_epe': seed_epe,
                'scalar/train/final_epe': final_epe,
                'scalar/train/delta_epe': seed_epe - final_epe,
                'scalar/train/lr_actor': actor_lr,
                'scalar/train/lr_critic': critic_lr,
            })

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
                action_penalty_weight=self.action_penalty_weight,
            )
            infer_time = time.time() - infer_start

            model_pred = rollout['final_pred']
            disp_pred = model_pred['disp_pred']
            disp_gt = data['disp']
            mask = (disp_gt < evaluator_cfgs.MAX_DISP) & (disp_gt > 0)
            if 'occ_mask' in data and evaluator_cfgs.get('APPLY_OCC_MASK', False):
                mask = mask & ~data['occ_mask'].to(torch.bool)

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
