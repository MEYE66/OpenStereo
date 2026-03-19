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





# stereo images, 



class DRLExposureController(nn.Module):
    def __init__(self, min_t=0.5, max_t=2.0, min_gain=1.0, max_gain=14.0):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False)
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, padding=1, bias=False)


        self.fc_out = nn.Linear(16, 2)  # 输出 (exp, gain)
        self.min_t, self.max_t = min_t, max_t
        self.min_gain, self.max_gain = min_gain, max_gain

    def forward(self, inputs):
        radiance_left, radiance_right = inputs['left'], inputs['right']
        x = (radiance_left + radiance_right) / 2

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.adaptive_avg_pool2d(self.conv3(x), (1, 1)).view(x.size(0), -1)
        out = self.fc_out(x)
        exp_time = torch.clamp(out[:, 0], self.min_t, self.max_t)
        gain = torch.clamp(out[:, 1], self.min_gain, self.max_gain)
        return exp_time, gain
    



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