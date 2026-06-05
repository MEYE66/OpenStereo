from stereo.modeling.trainer_template import TrainerTemplate
from stereo.modeling.finetune_trainer_template import FinetuneTrainerTemplate
from .neural_ae import NeuralAERAFTStereoDual


__all__ = {
    'NeuralAERAFTStereoDual': NeuralAERAFTStereoDual,
    'NeuralAERAFTStereoDualFinetune': NeuralAERAFTStereoDual,
}




class Trainer(TrainerTemplate):
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer):
        model = __all__[cfgs.MODEL.NAME](cfgs.MODEL)
        super().__init__(args, cfgs, local_rank, global_rank, logger, tb_writer, model)


class FinetuneTrainer(FinetuneTrainerTemplate):
    def __init__(self, args, cfgs, local_rank, global_rank, logger, tb_writer):
        model = __all__[cfgs.MODEL.NAME](cfgs.MODEL)
        super().__init__(args, cfgs, local_rank, global_rank, logger, tb_writer, model)
