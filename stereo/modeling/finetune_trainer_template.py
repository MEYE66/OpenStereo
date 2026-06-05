# @Time    : 2024/1/20 03:13
# @Author  : zhangchenming
import torch

from stereo.modeling.trainer_template import TrainerTemplate
from stereo.utils import common_utils
from stereo.utils.lamb import Lamb


class FinetuneTrainerTemplate(TrainerTemplate):
    def _get_base_model(self):
        return self.model.module if self.args.dist_mode else self.model

    def _validate_and_freeze_for_exposure_finetune(self):
        model = self._get_base_model()
        if not hasattr(model, 'exposure_controller'):
            raise AttributeError(
                'ExposureControllerFinetuneTrainerTemplate only supports exposure-controller '
                'stereo cascade models with attribute: exposure_controller.'
            )

        for _, param in model.named_parameters():
            param.requires_grad = False

        trainable_count = 0
        for _, param in model.exposure_controller.named_parameters():
            param.requires_grad = True
            trainable_count += 1

        if trainable_count == 0:
            raise ValueError(
                'The exposure_controller has no trainable parameters. '
                'Please check whether it is a learnable controller.'
            )

        self.logger.info('Freeze stereo matching network parameters and only optimize exposure_controller')

    def build_optimizer_and_scheduler(self):
        self._validate_and_freeze_for_exposure_finetune()

        model = self._get_base_model()
        params = [p for p in model.exposure_controller.parameters() if p.requires_grad]

        if self.cfgs.OPTIMIZATION.OPTIMIZER.NAME == 'Lamb':
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



