import sys
from pathlib import Path
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


repo_root = Path(__file__).resolve().parents[4]
# print(f"Adding repo root to sys.path: {repo_root}")
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from stereo.modeling.models.aegwcnet.ae_util import ImageFormationModel, exposure_value_equation
from stereo.modeling.models.gwcnet.gwcnet import GwcNet as BaseGwcNet



class RinToneMapper(nn.Module):
    def __init__(self, param=0.18):
        super().__init__()
        self.param = param

    def forward(self, img):
        img = (img - img.min()) / (img.max() - img.min() + 1e-6)
        Lw_ave = torch.exp(torch.mean(torch.log(1e-6 + img)))
        Lm = (self.param / Lw_ave) * img
        Lm_max = Lm.max()
        out = (Lm * (1 + (Lm / (Lm_max ** 2)))) / (1 + Lm)
        return out


class CNNToneMapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(32, 3, kernel_size=3, padding=1, bias=False)

    def forward(self, img):
        x = F.relu(self.conv1(img))
        x = F.relu(self.conv2(x))
        out = torch.sigmoid(self.conv3(x)) * img
        return out



class LTModule(nn.Module):
    def __init__(self, param=0.18):
        super().__init__()
        self.rin_tone_mapper = RinToneMapper(param=param)
    def forward(self, img):
        x = self.rin_tone_mapper(img)
        out = self.cnn_tone_mapper(x)
        return out



# stereo images, 
class SController(nn.Module):
    def __init__(self, ):
        super().__init__()
        self.conv1 = nn.Conv2d(6, 32, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(32, 1, kernel_size=3, padding=1, bias=False)

        self.expo_pred = nn.Linear(1, 1, bias=False)
        self.gain_pred = nn.Linear(1, 1, bias=False)

    def exposure_act(self, x, max_value, min_value):
        out = (1 - torch.sigmoid(x)) * torch.log(min_value) + torch.sigmoid(x) * torch.log(max_value)
        out = torch.exp(out)
        return out

    def forward(self, image):
        return image

    

if __name__ == '__main__':
    from types import SimpleNamespace
    cfgs = SimpleNamespace(
        MAX_DISP=int(192),
        USE_CONCAT_VOLUME=bool(True),
        CONCAT_CHANNELS=int(8),
        DOWNSAMPLE=int(4),
        NUM_GROUPS=int(8),
    )

    neural_aegwcnet_model = DRLExposureController(cfgs=cfgs)

    inputs = {
        'left': torch.randn(1, 3, 256, 512),
        'right': torch.randn(1, 3, 256, 512),
    }