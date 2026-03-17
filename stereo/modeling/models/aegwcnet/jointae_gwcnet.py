import torch
import torch.nn as nn
import torch.nn.functional as F

from stereo.modeling.models.gwcnet.gwcnet import GwcNet as BaseGwcNet



class AverageBasedAutoExposure(nn.Module):
    def __init__(self):
        super(AverageBasedAutoExposure, self).__init__()

    def forward(self, img):
        # img: [B, C, H, W]
        avg_intensity = torch.mean(img, dim=[1, 2, 3], keepdim=True)  # [B, 1, 1, 1]
        return avg_intensity.squeeze()  # [B]