# @Time    : 2023/8/26 13:02
# @Author  : zhangchenming

# from ref_code.averageae_arch import AverageAEStereoNet

from .models.casnet.trainer import Trainer as CasStereoTrainer
from .models.gwcnet.trainer import Trainer as GwcNetTrainer
from .models.gwcnet.trainer import SequenceTrainer as GwcNetSequenceTrainer



from .models.msnet.trainer import Trainer as MSNetTrainer
from .models.psmnet.trainer import Trainer as PSMNetTrainer
from .models.sttr.trainer import Trainer as STTRTrainer
# from .models.lightstereo.trainer import Trainer as LightStereoTrainer
# from .models.stereobase.trainer import Trainer as StereoBaseGRUTrainer
# from .models.iinet.trainer import Trainer as IINetTrainer


from .models.avenet.trainer_gwcnet import Trainer as AverageAEGwcNetTrainer
from .models.avenet.trainer_psmnet import Trainer as AverageAEPSMNetTrainer
from .models.gradnet.trainer_gwcnet import Trainer as GradientAEGwcNetTrainer
from .models.gradnet.trainer_gwcnet import RLTrainer as RLGradientAEGwcNetTrainer
from .models.gradnet.trainer_psmnet import Trainer as GardientAEPSMNetTrainer
from .models.gradnet.trainer_psmnet import RLTrainer as RLGardientAEPSMNetTrainer
from .models.neuralnet.trainer_gwcnet import Trainer as NeuralAEGwcNetTrainer
from .models.neuralnet.trainer_psmnet import Trainer as NeuralAEPSMNetTrainer
from .models.stereonet.trainer import Trainer as StereoAEGwcNetTrainer

# try:
# 'If you want to train/eval NMRF-Stereo, please refer to docs/prepare_foundationstereo.md
    # from .models.foundationstereo.trainer import Trainer as FoundationStereoTrainer
# except:
    # raise ValueError('If you want to train/eval NMRF-Stereo, please refer to docs/prepare_foundationstereo.md. Otherwise you can comment out this line of code')


# If you want to train/eval NMRF-Stereo, you need to build deformable attention and superpixel-guided disparity downsample operator: 'cd stereo/modeling/models/nmrf/ops && sh make.sh && cd ..'
# try:
#     from .models.nmrf.trainer import Trainer as NMRFTrainer
# except:
#     raise ValueError("If you want to train/eval NMRF-Stereo, you need to build deformable attention and superpixel-guided disparity downsample operator: 'cd stereo/modeling/models/nmrf/ops && sh make.sh && cd ..'")

__all__ = {
    'STTR': STTRTrainer,
    'PSMNet': PSMNetTrainer,
    'MSNet2D': MSNetTrainer,
    'MSNet3D': MSNetTrainer,
    'GwcNet': GwcNetTrainer,
    # 'AANet': AANetTrainer,
    'CasGwcNet': CasStereoTrainer,
    'CasPSMNet': CasStereoTrainer,
    # 'LightStereo': LightStereoTrainer,
    # 'StereoBaseGRU': StereoBaseGRUTrainer,
    # 'FoundationStereo': FoundationStereoTrainer,
    # 'IInet': IINetTrainer,
    # 'NMRF': NMRFTrainer


    # Tonemapping variants of GwcNet
    "IAGwcNet": GwcNetTrainer,
    "RAODGwcNet": GwcNetTrainer,
    "SANGwcNet": GwcNetTrainer,
    "GamutGwcNet": GwcNetTrainer,

    # Tonemapping variants of PSMNet
    "IAPSMNet": PSMNetTrainer,
    "RAODPSMNet": PSMNetTrainer,
    "SANPSMNet": PSMNetTrainer,

    # Stereo AE variants
    "AverageAEGwcNet": AverageAEGwcNetTrainer,
    "GradientAEGwcNet": GradientAEGwcNetTrainer,
    "NeuralAEGwcNet": NeuralAEGwcNetTrainer,
    "StereoAEGwcNet": StereoAEGwcNetTrainer,
    "AverageAEPSMNet": AverageAEPSMNetTrainer,
    "GardientAEPSMNet": GardientAEPSMNetTrainer,
    "RLGardientAEPSMNet": RLGardientAEPSMNetTrainer,
    "NeuralAEPSMNet": NeuralAEPSMNetTrainer,
    "RLGradientAEGwcNet": RLGradientAEGwcNetTrainer,

    # Sequence AE variants
    'GwcSequenceNet': GwcNetSequenceTrainer,

}   


def build_trainer(args, cfgs, local_rank, global_rank, logger, tb_writer):
    trainer = __all__[cfgs.MODEL.NAME](args, cfgs, local_rank, global_rank, logger, tb_writer)
    return trainer
