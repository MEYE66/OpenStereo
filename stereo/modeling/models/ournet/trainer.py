from stereo.modeling.trainer_template import TrainerTemplate
# from .average_ae import AverageAEGwcNet
from .our_ae import StereoAEGwcNet


__all__ = {
    'StereoAEGwcNet': StereoAEGwcNet,
}


class Trainer(TrainerTemplate):
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer):
        model = __all__[cfgs.MODEL.NAME](cfgs.MODEL)
        super().__init__(args, cfgs, local_rank, global_rank, logger, tb_writer, model)
