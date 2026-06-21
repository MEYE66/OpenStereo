from stereo.modeling.trainer_template import TrainerTemplate
from ...rl_trainer_template import A2CTrainerTemplate

from .adp_ae import AbsAdpAENet


__all__ = {
    'AbsAdpAENet': AbsAdpAENet,
}


class Trainer(TrainerTemplate):
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer):
        model = __all__[cfgs.MODEL.NAME](cfgs.MODEL)
        super().__init__(args, cfgs, local_rank, global_rank, logger, tb_writer, model)


class RLTrainer(A2CTrainerTemplate):
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer):
        model = __all__[cfgs.MODEL.NAME](cfgs.MODEL)
        super().__init__(args, cfgs, local_rank, global_rank, logger, tb_writer, model)
