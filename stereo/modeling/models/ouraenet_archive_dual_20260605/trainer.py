from stereo.modeling.trainer_template import TrainerTemplate
from ...rl_trainer_template import A2CTrainerTemplate

from .our_ae import OurAENet, OurAENetRAFTStereoDual

__all__ = {
    'OurAENet': OurAENet,
    'OurAENetRAFTStereoDual': OurAENetRAFTStereoDual,
}


class Trainer(TrainerTemplate):
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer):
        model = __all__[cfgs.MODEL.NAME](cfgs.MODEL)
        super().__init__(args, cfgs, local_rank, global_rank, logger, tb_writer, model)


class RLTrainer(A2CTrainerTemplate):
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer):
        model = __all__[cfgs.MODEL.NAME](cfgs.MODEL)
        super().__init__(args, cfgs, local_rank, global_rank, logger, tb_writer, model)
