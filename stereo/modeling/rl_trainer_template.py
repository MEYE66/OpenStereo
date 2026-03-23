# @Time    : 2026/3/20
# @Author  : GitHub Copilot
import time
import torch

from stereo.modeling.trainer_template import TrainerTemplate
from stereo.utils import common_utils
from stereo.utils.common_utils import color_map_tensorboard, write_tensorboard


class ActorCriticTrainerTemplate(TrainerTemplate):
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer, model):
        self.rl_cfg = getattr(cfgs.OPTIMIZATION, 'RL', {})
        super().__init__(args, cfgs, local_rank, global_rank, logger, tb_writer, model)

    def _get_rl_model(self):
        return self.model.module if self.args.dist_mode else self.model

    def build_optimizer_and_scheduler(self):
        rl_model = self._get_rl_model()
        if hasattr(rl_model, 'set_stereo_requires_grad'):
            train_stereo = bool(self.rl_cfg.get('TRAIN_STEREO', True))
            rl_model.set_stereo_requires_grad(train_stereo)
            if not train_stereo:
                self.logger.info('Freeze stereo matching network parameters for Actor-Critic training')

        if hasattr(rl_model, 'get_rl_parameters'):
            params = rl_model.get_rl_parameters(self.rl_cfg)
        else:
            params = [p for p in self.model.parameters() if p.requires_grad]

        if self.cfgs.OPTIMIZATION.OPTIMIZER.NAME == 'Lamb':
            from stereo.utils.lamb import Lamb
            optimizer_cls = Lamb
        else:
            optimizer_cls = getattr(torch.optim, self.cfgs.OPTIMIZATION.OPTIMIZER.NAME)
        valid_arg = common_utils.get_valid_args(optimizer_cls, self.cfgs.OPTIMIZATION.OPTIMIZER, ['name'])
        optimizer = optimizer_cls(params=params, **valid_arg)

        self.cfgs.OPTIMIZATION.SCHEDULER.TOTAL_STEPS = self.max_iter
        scheduler_cls = getattr(torch.optim.lr_scheduler, self.cfgs.OPTIMIZATION.SCHEDULER.NAME)
        valid_arg = common_utils.get_valid_args(scheduler_cls, self.cfgs.OPTIMIZATION.SCHEDULER, ['name', 'on_epoch'])
        scheduler = scheduler_cls(optimizer, **valid_arg)
        return optimizer, scheduler

    def _get_forward_rl_func(self):
        model = self._get_rl_model()
        if not hasattr(model, 'forward_rl'):
            raise AttributeError('Model does not implement forward_rl for Actor-Critic training.')
        return model.forward_rl

    def _get_rl_loss_func(self):
        model = self._get_rl_model()
        if not hasattr(model, 'get_rl_loss'):
            raise AttributeError('Model does not implement get_rl_loss for Actor-Critic training.')
        return model.get_rl_loss

    def train_one_epoch(self, current_epoch, tbar):
        start_epoch = self.last_epoch + 1
        logger_iter_interval = self.cfgs.TRAINER.LOGGER_ITER_INTERVAL
        total_loss = 0.0
        forward_rl_func = self._get_forward_rl_func()
        rl_loss_func = self._get_rl_loss_func()

        train_loader_iter = iter(self.train_loader)
        for i in range(0, len(self.train_loader)):
            total_iter = current_epoch * len(self.train_loader) + i
            if total_iter >= self.max_iter:
                break

            self.optimizer.zero_grad()
            lr = self.optimizer.param_groups[0]['lr']

            start_timer = time.time()
            data = next(train_loader_iter)
            for k, v in data.items():
                data[k] = v.to(self.local_rank) if torch.is_tensor(v) else v
            data_timer = time.time()

            with torch.cuda.amp.autocast(enabled=self.cfgs.OPTIMIZATION.AMP):
                model_pred, rl_info = forward_rl_func(data, deterministic=False)
                infer_timer = time.time()
                loss, tb_info = rl_loss_func(model_pred, data, rl_info, self.rl_cfg)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            if self.clip_grad is not None:
                self.clip_grad(self.model)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            with self.warmup_scheduler.dampening():
                if not self.cfgs.OPTIMIZATION.SCHEDULER.ON_EPOCH:
                    self.scheduler.step()

            total_loss += loss.item()

            trained_time_past_all = tbar.format_dict['elapsed']
            single_iter_second = trained_time_past_all / (total_iter + 1 - start_epoch * len(self.train_loader))
            remaining_second_all = single_iter_second * (self.total_epochs * len(self.train_loader) - total_iter - 1)
            if total_iter % logger_iter_interval == 0:
                message = ('Training Epoch:{:>2d}/{} Iter:{:>4d}/{} '
                           'Loss:{:#.6g}({:#.6g}) LR:{:.4e} '
                           'DataTime:{:.2f} InferTime:{:.2f}ms '
                           'Time cost: {}/{}'
                           ).format(current_epoch, self.total_epochs, i, len(self.train_loader),
                                    loss.item(), total_loss / (i + 1), lr,
                                    data_timer - start_timer, (infer_timer - data_timer) * 1000,
                                    tbar.format_interval(trained_time_past_all),
                                    tbar.format_interval(remaining_second_all))
                self.logger.info(message)

            if self.cfgs.TRAINER.TRAIN_VISUALIZATION:
                tb_info['image/train/image'] = torch.cat([data['left'][0], data['right'][0]], dim=1) / 256
                tb_info['image/train/disp'] = color_map_tensorboard(data['disp'][0], model_pred['disp_pred'].squeeze(1)[0])

            tb_info.update({'scalar/train/lr': lr})
            if total_iter % logger_iter_interval == 0 and self.local_rank == 0 and self.tb_writer is not None:
                write_tensorboard(self.tb_writer, tb_info, total_iter)
