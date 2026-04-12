from stereo.modeling.trainer_template import TrainerTemplate
from ...rl_trainer_template import A2CTrainerTemplate

# from .our_rl_actor_critic import RLOurAEGwcNet
from .our_rl_actor_critic_act import RLOurAEGwcNet

# try:
#     from .our_ae import StereoAEGwcNet
# except ImportError:
#     StereoAEGwcNet = None


__all__ = {
    'RLOurAEGwcNet': RLOurAEGwcNet,
}

# if StereoAEGwcNet is not None:
    # __all__['StereoAEGwcNet'] = StereoAEGwcNet


class Trainer(TrainerTemplate):
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer):
        model = __all__[cfgs.MODEL.NAME](cfgs.MODEL)
        super().__init__(args, cfgs, local_rank, global_rank, logger, tb_writer, model)


class RLTrainer(A2CTrainerTemplate):
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer):
        model = __all__[cfgs.MODEL.NAME](cfgs.MODEL)
        super().__init__(args, cfgs, local_rank, global_rank, logger, tb_writer, model)
