import torch
import torch.nn as nn
import torch.nn.functional as F

# from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.ae_utils import ImageFormationModel, GradientExposureControl
from basicsr.archs.stereos.psmnet.psmnet import PSMNet
from basicsr.archs.stereos.gwcnet.gwcnet import GWCNet
from basicsr.utils.registry import ARCH_REGISTRY


def exposure_value_equation(ev_vaules, time_limits, gain_limits):
    gain = torch.clamp(ev_vaules / time_limits[1], 1., None)
    expo = ev_vaules / gain
    gain = torch.clamp(gain, gain_limits[0], gain_limits[1])
    expo = torch.clamp(expo, time_limits[0], time_limits[1])
    return expo, gain


class GradientAEStereoNet(nn.Module):
    def __init__(self, max_disp=192, time_limits=[1., 20.], gain_limits=[1., 14.]):
        super(GradientAEStereoNet, self).__init__()
        self.image_formation_model = ImageFormationModel().cuda()
        self.exposure_controller = GradientExposureControl().cuda()
        self.disp_model = GWCNet(max_disp=max_disp).cuda()

        self.time_limits = torch.tensor(time_limits, requires_grad=True).cuda()
        self.gain_limits = torch.tensor(gain_limits, requires_grad=True).cuda()

        self.init_exp = torch.tensor((time_limits[0] + time_limits[1]) / 2,  requires_grad=True).cuda()
        self.init_gain = torch.tensor((gain_limits[0] + gain_limits[1]) / 2,  requires_grad=True).cuda()

    def forward(self,radiance_left, radiance_right ):
        # print(inputs.keys())
        # 1.simulation
        # print(radiance_left.shape, radiance_right.shape)
        # print(f"value range:{radiance_left.min().item()} ~ {radiance_left.max().item()}, {radiance_right.min().item()} ~ {radiance_right.max().item()}")
        img_left = self.image_formation_model(radiance_left, self.init_exp, self.init_gain)
        img_right = self.image_formation_model(radiance_right, self.init_exp, self.init_gain)

        # 2.auto exposure control
        expo_values = self.exposure_controller((img_left+img_right)/2)
        expo_update, gain_update = exposure_value_equation(expo_values, self.time_limits, self.gain_limits)

        # 3.update simulation
        img_left_updated = self.image_formation_model(radiance_left, expo_update, gain_update)
        img_right_updated = self.image_formation_model(radiance_right, expo_update, gain_update)

        # 4.disparity estimation
        disp_pred = self.disp_model(img_left_updated, img_right_updated)
        return disp_pred



# class GradientAEMonoNet(nn.Module):
#     def __init__(self):
#         super(GradientAEMonoNet, self).__init__()



if __name__ == '__main__':

    cuda_available = torch.cuda.is_available()
    model = GradientAEStereoNet()
    radiance_left = torch.rand(1, 3, 256, 512)
    radiance_right = torch.rand(1, 3, 256, 512)

    if cuda_available:
        model = model.cuda()
        radiance_left = radiance_left.cuda()
        radiance_right = radiance_right.cuda()


    disp_pred = model(radiance_left, radiance_right)
    print(disp_pred)