from stereo.modeling.trainer_template import TrainerTemplate
from stereo.modeling.finetune_trainer_template import FinetuneTrainerTemplate
from .grad_ae import GradientAEGwcNet, GradientAELidarGwcNet



__all__ = {
    'GradientAEGwcNet': GradientAEGwcNet,
    'GradientAELidarGwcNet': GradientAELidarGwcNet,
    'GradientAEGwcNetFinetune': GradientAEGwcNet,
    'GradientAELidarGwcNetFinetune': GradientAELidarGwcNet,
}


class Trainer(TrainerTemplate):
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer):
        model = __all__[cfgs.MODEL.NAME](cfgs.MODEL)
        super().__init__(args, cfgs, local_rank, global_rank, logger, tb_writer, model)




class FinetuneTrainer(FinetuneTrainerTemplate):
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer):
        model = __all__[cfgs.MODEL.NAME](cfgs.MODEL)
        super().__init__(args, cfgs, local_rank, global_rank, logger, tb_writer, model)
