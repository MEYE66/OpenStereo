import torch
import torch.nn as nn
import torch.nn.functional as F



import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[4]
# print(f"Adding repo root to sys.path: {repo_root}")
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    # Prefer absolute imports so this file can be run directly.
    from stereo.modeling.models.gwcnet.gwcnet_backbone import GwcNet as GwcNetBackbone
    from stereo.modeling.models.gwcnet.gwcnet_cost_processor import GwcVolumeCostProcessor
    from stereo.modeling.models.gwcnet.gwcnet_temporal_disp_processor import TemporalGwcDispProcessor
except ModuleNotFoundError:

    from .gwcnet_backbone import GwcNet as GwcNetBackbone
    from .gwcnet_cost_processor import GwcVolumeCostProcessor
    from .gwcnet_temporal_disp_processor import TemporalGwcDispProcessor




# try:
#     # Prefer absolute imports so this file can be run directly.
#     from stereo.modeling.models.ae_util import ImageFormationModel, exposure_value_equation
#     from stereo.modeling.models.gwcnet.gwcnet import GwcNet as BaseGwcNet
#     from stereo.modeling.models.psmnet.psmnet import PSMNet as BasePSMNet
# except ModuleNotFoundError:
#     # Fallback for environments where package root is preconfigured.
#     from ..ae_util import ImageFormationModel, exposure_value_equation
#     from ..gwcnet.gwcnet import GwcNet as BaseGwcNet
#     from ..psmnet.psmnet import PSMNet as BasePSMNet


class GwcSequenceNet(nn.Module):
    """GwcNet with temporal cost-volume aggregation via ConvGRU.

    The model reads stereo pairs from two timestamps. Features from t-1 are
    first propagated through the hourglass stack to obtain hidden states, then
    injected into the t path through ConvGRU layers.
    """

    def __init__(self, cfgs):
        super().__init__()
        self.maxdisp = cfgs.MAX_DISP

        use_concat_volume = cfgs.USE_CONCAT_VOLUME
        concat_channels = cfgs.CONCAT_CHANNELS
        downsample = cfgs.DOWNSAMPLE
        num_groups = cfgs.NUM_GROUPS

        self.Backbone = GwcNetBackbone(use_concat_volume=use_concat_volume, concat_channels=concat_channels)
        self.CostProcessor = GwcVolumeCostProcessor(maxdisp=self.maxdisp, downsample=downsample, num_groups=num_groups,
                                                    use_concat_volume=use_concat_volume)
        self.DispProcessor = TemporalGwcDispProcessor(maxdisp=self.maxdisp, downsample=downsample,
                                                      num_groups=num_groups, use_concat_volume=use_concat_volume,
                                                      concat_channels=concat_channels)


    def _run_single_step(self, frame_inputs, prev_hidden_states=None):
        backbone_out = self.Backbone(frame_inputs)
        step_inputs = dict(frame_inputs)
        step_inputs.update(backbone_out)
        cost_out = self.CostProcessor(step_inputs)
        step_inputs.update(cost_out)
        disp_out, hidden_states = self.DispProcessor.forward_with_hidden(step_inputs, prev_hidden_states)
        return disp_out, hidden_states

    def forward(self, inputs):
        prev_inputs = {'left': inputs['left_1'], 'right': inputs['right_1']}
        _, prev_hidden_states = self._run_single_step(prev_inputs, prev_hidden_states=None)

        curr_inputs = {'left': inputs['left_2'], 'right': inputs['right_2']}
        disp_out, _ = self._run_single_step(curr_inputs, prev_hidden_states=prev_hidden_states)

        if self.training:
            disp_ests = disp_out['training_disp']['disp']['disp_ests']
            return {
                'disp_preds': disp_ests,
                'disp_pred': disp_ests[-1]
            }

        return {'disp_pred': disp_out['inference_disp']['disp_est']}

    def get_loss(self, model_preds, input_data):
        if 'disp_1' in input_data:
            disp_gt = input_data['disp_1']
        elif 'disp2' in input_data:
            disp_gt = input_data['disp2']
        else:
            disp_gt = input_data['disp']
        mask = (disp_gt < self.maxdisp) & (disp_gt > 0)

        weights = [0.5, 0.5, 0.7, 1.0]

        loss = 0.0
        for disp_est, weight in zip(model_preds['disp_preds'], weights):
            loss += weight * F.smooth_l1_loss(disp_est[mask], disp_gt[mask], size_average=True)

        loss_info = {'scalar/train/loss_disp': loss.item()}
        return loss, loss_info



if __name__ == '__main__':
    from types import SimpleNamespace
    cfgs = SimpleNamespace(
        MAX_DISP=int(192),
        USE_CONCAT_VOLUME=bool(True),
        CONCAT_CHANNELS=int(8),
        DOWNSAMPLE=int(4),
        NUM_GROUPS=int(8),
    )

    gwc_sequence_model = GwcSequenceNet(cfgs=cfgs)
    # print(gwc_sequence_model)   


    inputs = {
        'left_1': torch.randn(1, 3, 256, 512),
        'right_1': torch.randn(1, 3, 256, 512),
        'left_2': torch.randn(1, 3, 256, 512),
        'right_2': torch.randn(1, 3, 256, 512),
    }


    outputs = gwc_sequence_model(inputs)
    print('Model outputs:', outputs)
    