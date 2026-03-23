from stereo.modeling.trainer_template import TrainerTemplate
from stereo.modeling.rl_trainer_template import ActorCriticTrainerTemplate
from .grad_ae import GardientAEPSMNet
from .grad_ae_actor_critic import RLGardientAEPSMNet



__all__ = {
    'GardientAEPSMNet': GardientAEPSMNet,
    'RLGardientAEPSMNet': RLGardientAEPSMNet,
}


class Trainer(TrainerTemplate):
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer):
        model = __all__[cfgs.MODEL.NAME](cfgs.MODEL)
        super().__init__(args, cfgs, local_rank, global_rank, logger, tb_writer, model)


class RLTrainer(ActorCriticTrainerTemplate):
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer):
        model = __all__[cfgs.MODEL.NAME](cfgs.MODEL)
        super().__init__(args, cfgs, local_rank, global_rank, logger, tb_writer, model)
