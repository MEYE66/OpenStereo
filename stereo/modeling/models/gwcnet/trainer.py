# @Time    : 2024/2/9 11:39
# @Author  : zhangchenming
from stereo.modeling.trainer_template import TrainerTemplate
from stereo.modeling.carla_trainer import TrainerSequenceTemplate

from .gwcnet import GwcNet
from .gwcnet_tonemapping import IAGwcNet, RAODGwcNet, SANGwcNet
from .gwcnet_sequence import GwcSequenceNet

__all__ = {
    'GwcNet': GwcNet,
    'IAGwcNet': IAGwcNet,
    'RAODGwcNet': RAODGwcNet,
    'SANGwcNet': SANGwcNet,
    'GwcSequenceNet': GwcSequenceNet,
}

class Trainer(TrainerTemplate):
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer):
        model = __all__[cfgs.MODEL.NAME](cfgs.MODEL)
        super().__init__(args, cfgs, local_rank, global_rank, logger, tb_writer, model)



class SequenceTrainer(TrainerSequenceTemplate):
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer):
        model = __all__[cfgs.MODEL.NAME](cfgs.MODEL)
        super().__init__(args, cfgs, local_rank, global_rank, logger, tb_writer, model)