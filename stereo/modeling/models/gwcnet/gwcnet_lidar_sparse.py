import torch
import torch.nn.functional as F

from .gwcnet import GwcNet


class LidarGwcNet(GwcNet):
    def __init__(self, cfgs):
        super().__init__(cfgs)

    def get_loss(self, model_preds, input_data):
        disp_gt = input_data["disp"]  # [bz, h, w]

        if "valid" in input_data:
            mask = input_data["valid"].to(torch.bool)
        else:
            mask = (disp_gt < self.maxdisp) & (disp_gt > 0)

        valid_count = int(mask.sum().item())
        if valid_count == 0:
            zero_loss = sum((pred.sum() * 0.0) for pred in model_preds["disp_preds"])
            loss_info = {
                "scalar/train/loss_disp": 0.0,
                "scalar/train/sparse_valid_count": 0,
                "scalar/train/sparse_empty_batch": 1,
            }
            return zero_loss, loss_info

        weights = [0.5, 0.5, 0.7, 1.0]

        loss = 0.0
        for disp_est, weight in zip(model_preds["disp_preds"], weights):
            loss += weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True)

        loss_info = {
            "scalar/train/loss_disp": loss.item(),
            "scalar/train/sparse_valid_count": valid_count,
            "scalar/train/sparse_empty_batch": 0,
        }
        return loss, loss_info
