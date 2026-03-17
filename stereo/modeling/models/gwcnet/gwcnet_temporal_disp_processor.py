import torch
import torch.nn as nn
import torch.nn.functional as F

from .hourglass import Hourglass
from .gwcnet_disp_processor import convbn_3d, disparity_regression


class ConvGRU3DCell(nn.Module):
    """3D ConvGRU cell for temporal aggregation on cost volume features."""

    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        gate_in_channels = input_channels + hidden_channels

        self.gate_conv = nn.Conv3d(gate_in_channels, hidden_channels * 2, kernel_size=kernel_size,
                                   padding=padding, bias=True)
        self.candidate_conv = nn.Conv3d(gate_in_channels, hidden_channels, kernel_size=kernel_size,
                                        padding=padding, bias=True)

        self.hidden_channels = hidden_channels

    def forward(self, x, hidden_state=None):
        if hidden_state is None:
            hidden_state = torch.zeros(
                x.size(0), self.hidden_channels, x.size(2), x.size(3), x.size(4),
                device=x.device, dtype=x.dtype
            )

        gates = self.gate_conv(torch.cat([x, hidden_state], dim=1))
        update_gate, reset_gate = torch.chunk(gates, 2, dim=1)
        update_gate = torch.sigmoid(update_gate)
        reset_gate = torch.sigmoid(reset_gate)

        candidate = self.candidate_conv(torch.cat([x, reset_gate * hidden_state], dim=1))
        candidate = torch.tanh(candidate)

        new_hidden_state = (1.0 - update_gate) * hidden_state + update_gate * candidate
        return new_hidden_state, new_hidden_state


class TemporalGwcDispProcessor(nn.Module):
    """Temporal variant of GwcNet disparity processor.

    ConvGRU cells are inserted after each hourglass stage. Hidden states from
    t-1 are fed into t to aggregate temporal cues.
    """

    def __init__(self, maxdisp=192, downsample=4, num_groups=40, use_concat_volume=True, concat_channels=12, *args,
                 **kwargs):
        super().__init__()

        self.maxdisp = maxdisp
        self.downsample = downsample
        self.num_groups = num_groups
        self.use_concat_volume = use_concat_volume
        self.concat_channels = concat_channels if use_concat_volume else 0

        self.dres0 = nn.Sequential(
            convbn_3d(self.num_groups + self.concat_channels * 2, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True)
        )

        self.dres1 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            convbn_3d(32, 32, 3, 1, 1)
        )

        self.dres2 = Hourglass(32)
        self.dres3 = Hourglass(32)
        self.dres4 = Hourglass(32)

        self.temporal_gru1 = ConvGRU3DCell(32, 32)
        self.temporal_gru2 = ConvGRU3DCell(32, 32)
        self.temporal_gru3 = ConvGRU3DCell(32, 32)

        self.classif0 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False)
        )

        self.classif1 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False)
        )

        self.classif2 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False)
        )

        self.classif3 = nn.Sequential(
            convbn_3d(32, 32, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1, stride=1, bias=False)
        )

    def _run_aggregation(self, volume, prev_hidden_states=None):
        if prev_hidden_states is None:
            prev_hidden_states = [None, None, None]

        cost0 = self.dres0(volume)
        cost0 = self.dres1(cost0) + cost0

        out1 = self.dres2(cost0)
        out1, hidden1 = self.temporal_gru1(out1, prev_hidden_states[0])

        out2 = self.dres3(out1)
        out2, hidden2 = self.temporal_gru2(out2, prev_hidden_states[1])

        out3 = self.dres4(out2)
        out3, hidden3 = self.temporal_gru3(out3, prev_hidden_states[2])

        return cost0, out1, out2, out3, [hidden1, hidden2, hidden3]

    def _build_predictions(self, cost0, out1, out2, out3, h, w):
        if self.training:
            cost0 = self.classif0(cost0)
            cost1 = self.classif1(out1)
            cost2 = self.classif2(out2)
            cost3 = self.classif3(out3)

            cost0 = F.interpolate(cost0, [self.maxdisp, h, w], mode='trilinear', align_corners=False)
            cost0 = torch.squeeze(cost0, 1)
            pred0 = F.softmax(cost0, dim=1)
            pred0 = disparity_regression(pred0, self.maxdisp)

            cost1 = F.interpolate(cost1, [self.maxdisp, h, w], mode='trilinear', align_corners=False)
            cost1 = torch.squeeze(cost1, 1)
            pred1 = F.softmax(cost1, dim=1)
            pred1 = disparity_regression(pred1, self.maxdisp)

            cost2 = F.interpolate(cost2, [self.maxdisp, h, w], mode='trilinear', align_corners=False)
            cost2 = torch.squeeze(cost2, 1)
            pred2 = F.softmax(cost2, dim=1)
            pred2 = disparity_regression(pred2, self.maxdisp)

            cost3 = F.interpolate(cost3, [self.maxdisp, h, w], mode='trilinear', align_corners=False)
            cost3 = torch.squeeze(cost3, 1)
            pred3 = F.softmax(cost3, dim=1)
            pred3 = disparity_regression(pred3, self.maxdisp)

            return {
                "training_disp": {
                    "disp": {
                        "disp_ests": [pred0, pred1, pred2, pred3],
                    },
                },
            }

        cost3 = self.classif3(out3)
        cost3 = F.interpolate(cost3, [self.maxdisp, h, w], mode='trilinear', align_corners=False)
        cost3 = torch.squeeze(cost3, 1)
        pred3 = F.softmax(cost3, dim=1)
        pred3 = disparity_regression(pred3, self.maxdisp)

        return {
            "inference_disp": {
                "disp_est": pred3,
            },
        }

    def forward_with_hidden(self, inputs, prev_hidden_states=None):
        volume = inputs['cost_volume']
        h, w = inputs['left'].shape[2:]
        cost0, out1, out2, out3, hidden_states = self._run_aggregation(volume, prev_hidden_states)
        output = self._build_predictions(cost0, out1, out2, out3, h, w)
        return output, hidden_states

    def forward(self, inputs):
        output, _ = self.forward_with_hidden(inputs, prev_hidden_states=None)
        return output

    def input_output(self):
        return {
            "inputs": ["cost_volume", "disp_shape"],
            "outputs": ["training_disp", "inference_disp", "visual_summary"]
        }
