import torch
import torch.nn as nn
import torch.nn.functional as F
from types import SimpleNamespace



import sys
from pathlib import Path
repo_root = Path(__file__).resolve().parents[4]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))


# from .submodule import AlternateCorrBlock
# from .submodule import BasicEncoder, MultiBasicEncoder, ResidualBlock
# from .submodule import BasicMultiUpdateBlock
# from .submodule import CorrBlock1D, PytorchAlternateCorrBlock1D, CorrBlockFast1D
# from .utils.utils import coords_grid, upflow8


from stereo.modeling.models.raftstereo.submodule import AlternateCorrBlock
from stereo.modeling.models.raftstereo.submodule import BasicEncoder, MultiBasicEncoder, ResidualBlock
from stereo.modeling.models.raftstereo.submodule import BasicMultiUpdateBlock
from stereo.modeling.models.raftstereo.submodule import CorrBlock1D, PytorchAlternateCorrBlock1D, CorrBlockFast1D
from stereo.modeling.models.raftstereo.fusion import LinearExposureFusion
from stereo.modeling.models.raftstereo.utils.utils import coords_grid, upflow8




try:
    autocast = torch.cuda.amp.autocast
except AttributeError:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass

class RAFTStereo(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        self.cfgs = cfgs
        # self.num
        self.max_disp = cfgs.MAX_DISP
        self.loss_gamma = cfgs.get('LOSS_GAMMA', 0.9)
        self.train_iters = cfgs.get('TRAIN_ITERS', 12)
        self.eval_iters = cfgs.get('EVAL_ITERS', self.train_iters)

        self.corr_implementation = cfgs.get('CORR_IMPLEMENTATION', 'reg')
        self.corr_levels = cfgs.CORR_LEVELS
        self.corr_radius = cfgs.CORR_RADIUS
        self.context_norm = cfgs.get('CONTEXT_NORM', 'batch')
        self.hidden_dims = cfgs.HIDDEN_DIMS
        self.mixed_precision = cfgs.get('MIXED_PRECISION', False)
        self.n_downsample = cfgs.N_DOWNSAMPLE
        self.n_gru_layers = cfgs.N_GRU_LAYERS
        self.shared_backbone = cfgs.get('SHARED_BACKBONE', False)
        self.slow_fast_gru = cfgs.get('SLOW_FAST_GRU', False)

        self.update_args = SimpleNamespace(
            corr_levels=self.corr_levels,
            corr_radius=self.corr_radius,
            n_downsample=self.n_downsample,
            n_gru_layers=self.n_gru_layers,
        )

        context_dims = self.hidden_dims
        self.cnet = MultiBasicEncoder(
            output_dim=[context_dims, context_dims],
            norm_fn=self.context_norm,
            downsample=self.n_downsample,
        )
        self.update_block = BasicMultiUpdateBlock(self.update_args, hidden_dims=self.hidden_dims)

        self.context_zqr_convs = nn.ModuleList([
            nn.Conv2d(context_dims[i], self.hidden_dims[i] * 3, 3, padding=3 // 2)
            for i in range(self.n_gru_layers)
        ])

        if self.shared_backbone:
            self.conv2 = nn.Sequential(
                ResidualBlock(128, 128, 'instance', stride=1),
                nn.Conv2d(128, 256, 3, padding=1),
            )
        else:
            self.fnet = BasicEncoder(
                output_dim=256,
                norm_fn='instance',
                downsample=self.n_downsample,
            )

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()

    def initialize_flow(self, img):
        """Flow is represented as difference between two coordinate grids."""
        batch_size, _, height, width = img.shape
        coords0 = coords_grid(batch_size, height, width).to(img.device)
        coords1 = coords_grid(batch_size, height, width).to(img.device)
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination."""
        batch_size, dims, height, width = flow.shape
        factor = 2 ** self.n_downsample
        mask = mask.view(batch_size, 1, 9, factor, factor, height, width)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * flow, [3, 3], padding=1)
        up_flow = up_flow.view(batch_size, dims, 9, 1, 1, height, width)
        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(batch_size, dims, factor * height, factor * width)

    def _normalize_inputs(self, image1, image2):
        max_value = max(image1.detach().amax().item(), image2.detach().amax().item())
        input_scale = 255.0 if max_value > 1.5 else 1.0
        image1 = (2 * (image1 / input_scale) - 1.0).contiguous()
        image2 = (2 * (image2 / input_scale) - 1.0).contiguous()
        return image1, image2

    def _predict_disparities(self, fmap1, fmap2, cnet_list, iters, flow_init=None, test_mode=False):
        net_list = [torch.tanh(item[0]) for item in cnet_list]
        inp_list = [torch.relu(item[1]) for item in cnet_list]
        inp_list = [
            list(conv(inp).split(split_size=conv.out_channels // 3, dim=1))
            for inp, conv in zip(inp_list, self.context_zqr_convs)
        ]

        if self.corr_implementation == 'reg':
            corr_block = CorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        elif self.corr_implementation == 'alt':
            corr_block = PytorchAlternateCorrBlock1D
            fmap1, fmap2 = fmap1.float(), fmap2.float()
        elif self.corr_implementation == 'reg_cuda':
            corr_block = CorrBlockFast1D
        elif self.corr_implementation == 'alt_cuda':
            corr_block = AlternateCorrBlock
        else:
            raise ValueError(f'Unsupported corr implementation: {self.corr_implementation}')

        corr_fn = corr_block(
            fmap1,
            fmap2,
            radius=self.corr_radius,
            num_levels=self.corr_levels,
        )

        coords0, coords1 = self.initialize_flow(net_list[0])
        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1)
            flow = coords1 - coords0

            with autocast(enabled=self.mixed_precision):
                if self.n_gru_layers == 3 and self.slow_fast_gru:
                    net_list = self.update_block(
                        net_list,
                        inp_list,
                        iter32=True,
                        iter16=False,
                        iter08=False,
                        update=False,
                    )

                if self.n_gru_layers >= 2 and self.slow_fast_gru:
                    net_list = self.update_block(
                        net_list,
                        inp_list,
                        iter32=self.n_gru_layers == 3,
                        iter16=True,
                        iter08=False,
                        update=False,
                    )

                net_list, up_mask, delta_flow = self.update_block(
                    net_list,
                    inp_list,
                    corr,
                    flow,
                    iter32=self.n_gru_layers == 3,
                    iter16=self.n_gru_layers >= 2,
                )

            delta_flow[:, 1] = 0.0
            coords1 = coords1 + delta_flow

            if test_mode and itr < iters - 1:
                continue

            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_predictions.append(flow_up[:, :1])

        if test_mode:
            return coords1 - coords0, flow_predictions[-1]

        return flow_predictions

    def _forward_impl(self, image1, image2, iters, flow_init=None, test_mode=False):
        image1, image2 = self._normalize_inputs(image1, image2)

        with autocast(enabled=self.mixed_precision):
            if self.shared_backbone:
                *cnet_list, x = self.cnet(
                    torch.cat((image1, image2), dim=0),
                    dual_inp=True,
                    num_layers=self.n_gru_layers,
                )
                fmap1, fmap2 = self.conv2(x).split(dim=0, split_size=x.shape[0] // 2)
            else:
                cnet_list = self.cnet(image1, num_layers=self.n_gru_layers)
                fmap1, fmap2 = self.fnet([image1, image2])

        return self._predict_disparities(
            fmap1,
            fmap2,
            cnet_list,
            iters,
            flow_init=flow_init,
            test_mode=test_mode,
        )

    def forward(self, inputs, image2=None, iters=None, flow_init=None, test_mode=False):
        if torch.is_tensor(inputs):
            if image2 is None:
                raise ValueError('image2 is required when calling RAFTStereo with raw tensors')
            if iters is None:
                iters = self.eval_iters if test_mode else self.train_iters
            return self._forward_impl(inputs, image2, iters, flow_init=flow_init, test_mode=test_mode)

        image1 = inputs.get('left', inputs.get('img1'))
        image2 = inputs.get('right', inputs.get('img2'))
        # if image1 is None or image2 is None:
            # raise KeyError('RAFTStereo expects input keys left/right or img1/img2')

        if iters is None:
            iters = self.train_iters if self.training and not test_mode else self.eval_iters

        if self.training and not test_mode:
            disp_preds = self._forward_impl(image1, image2, iters, flow_init=flow_init)
            return {
                'disp_preds': disp_preds,
                'disp_pred': disp_preds[-1],
            }

        _, disp_pred = self._forward_impl(image1, image2, iters, flow_init=flow_init, test_mode=True)
        return {'disp_pred': disp_pred}

    def get_loss(self, model_preds, input_data):
        disp_gt = input_data['disp']
        if disp_gt.dim() == 4:
            disp_gt = disp_gt.squeeze(1)

        valid = (disp_gt > 0) & (disp_gt < self.max_disp)
        input_valid = input_data.get('valid')
        if torch.is_tensor(input_valid):
            if input_valid.dim() == 4:
                input_valid = input_valid.squeeze(1)
            valid = valid & input_valid.bool()

        disp_preds = model_preds['disp_preds']
        if valid.sum().item() == 0:
            zero_loss = sum(pred.sum() * 0.0 for pred in disp_preds)
            loss_info = {
                'scalar/train/loss_disp': 0.0,
                'scalar/train/epe': 0.0,
                'scalar/train/1px': 0.0,
                'scalar/train/3px': 0.0,
                'scalar/train/5px': 0.0,
            }
            return zero_loss, loss_info

        disp_gt = disp_gt.unsqueeze(1)
        valid_mask = valid.unsqueeze(1)
        n_predictions = len(disp_preds)
        adjusted_loss_gamma = 1.0
        if n_predictions > 1:
            adjusted_loss_gamma = self.loss_gamma ** (15 / (n_predictions - 1))

        flow_loss = 0.0
        for idx, disp_pred in enumerate(disp_preds):
            weight = adjusted_loss_gamma ** (n_predictions - idx - 1)
            flow_loss = flow_loss + weight * (disp_pred - disp_gt).abs()[valid_mask].mean()

        epe = (disp_preds[-1] - disp_gt).abs()[valid_mask]
        loss_info = {
            'scalar/train/loss_disp': flow_loss.item(),
            'scalar/train/epe': epe.mean().item(),
            'scalar/train/1px': (epe < 1).float().mean().item(),
            'scalar/train/3px': (epe < 3).float().mean().item(),
            'scalar/train/5px': (epe < 5).float().mean().item(),
        }
        return flow_loss, loss_info


class RAFTStereoDual(RAFTStereo):
    def __init__(self, cfgs):
        super().__init__(cfgs)

        context_dims = self.hidden_dims
        self.cnet = MultiBasicEncoder(
            output_dim=[context_dims, context_dims],
            norm_fn=self.context_norm,
            downsample=self.n_downsample,
            num_input=6,
        )

        if not self.shared_backbone:
            self.fnet = BasicEncoder(
                output_dim=256,
                norm_fn='instance',
                downsample=self.n_downsample,
                num_input=6,
            )

    def _forward_dual_impl(self, image1, image2, image1_next, image2_next, iters, flow_init=None, test_mode=False):
        image1, image2 = self._normalize_inputs(image1, image2)
        image1_next, image2_next = self._normalize_inputs(image1_next, image2_next)

        left_pair = torch.cat((image1, image1_next), dim=1)
        right_pair = torch.cat((image2, image2_next), dim=1)

        with autocast(enabled=self.mixed_precision):
            if self.shared_backbone:
                *cnet_list, x = self.cnet(
                    torch.cat((left_pair, right_pair), dim=0),
                    dual_inp=True,
                    num_layers=self.n_gru_layers,
                )
                fmap1, fmap2 = self.conv2(x).split(dim=0, split_size=x.shape[0] // 2)
            else:
                cnet_list = self.cnet(left_pair, num_layers=self.n_gru_layers)
                fmap1, fmap2 = self.fnet([left_pair, right_pair])

        return self._predict_disparities(
            fmap1,
            fmap2,
            cnet_list,
            iters,
            flow_init=flow_init,
            test_mode=test_mode,
        )

    def forward(
        self,
        inputs,
        image2=None,
        image1_next=None,
        image2_next=None,
        iters=None,
        flow_init=None,
        test_mode=False,
    ):
        if torch.is_tensor(inputs):
            if image2 is None or image1_next is None or image2_next is None:
                raise ValueError(
                    'image2, image1_next, and image2_next are required when calling RAFTStereoDual with raw tensors'
                )
            if iters is None:
                iters = self.eval_iters if test_mode else self.train_iters
            return self._forward_dual_impl(
                inputs,
                image2,
                image1_next,
                image2_next,
                iters,
                flow_init=flow_init,
                test_mode=test_mode,
            )

        image1 = inputs['left_1']
        image2 = inputs['right_1']
        image1_next = inputs['left_2']
        image2_next = inputs['right_2']

        if iters is None:
            iters = self.train_iters if self.training and not test_mode else self.eval_iters

        if self.training and not test_mode:
            disp_preds = self._forward_dual_impl(
                image1,
                image2,
                image1_next,
                image2_next,
                iters,
                flow_init=flow_init,
            )
            return {
                'disp_preds': disp_preds,
                'disp_pred': disp_preds[-1],
            }

        _, disp_pred = self._forward_dual_impl(
            image1,
            image2,
            image1_next,
            image2_next,
            iters,
            flow_init=flow_init,
            test_mode=True,
        )
        return {'disp_pred': disp_pred}


class RAFTStereoFusionDual(RAFTStereo):
    def __init__(self, cfgs):
        super().__init__(cfgs)
        if self.shared_backbone:
            raise ValueError('RAFTStereoFusionDual currently requires SHARED_BACKBONE=False')

        self.fusion_num_exposures = int(cfgs.get('FUSION_NUM_EXPOSURES', 2))
        self.return_fusion_alpha = bool(cfgs.get('RETURN_FUSION_ALPHA', False))
        self.feature_fusion = LinearExposureFusion(
            dim=256,
            num_exposures=self.fusion_num_exposures,
            num_heads=int(cfgs.get('FUSION_NUM_HEADS', 4)),
            tau=float(cfgs.get('FUSION_TAU', 1.0)),
        )

    def _normalize_frame_pairs(self, left_images, right_images):
        norm_left = []
        norm_right = []
        for left_image, right_image in zip(left_images, right_images):
            left_image, right_image = self._normalize_inputs(left_image, right_image)
            norm_left.append(left_image)
            norm_right.append(right_image)
        return norm_left, norm_right

    def _extract_fused_matching_features(self, left_images, right_images):
        stereo_inputs = []
        for left_image, right_image in zip(left_images, right_images):
            stereo_inputs.extend((left_image, right_image))

        feature_maps = self.fnet(stereo_inputs)
        left_features = []
        right_features = []
        for idx in range(self.fusion_num_exposures):
            left_features.append(feature_maps[2 * idx])
            right_features.append(feature_maps[2 * idx + 1])

        left_features = torch.stack(left_features, dim=1)
        right_features = torch.stack(right_features, dim=1)
        fused_left, alpha_left = self.feature_fusion(left_features)
        fused_right, alpha_right = self.feature_fusion(right_features)
        return fused_left, fused_right, alpha_left, alpha_right

    def _forward_fusion_impl(self, left_images, right_images, iters, flow_init=None, test_mode=False):
        left_images, right_images = self._normalize_frame_pairs(left_images, right_images)

        with autocast(enabled=self.mixed_precision):
            cnet_list = self.cnet(left_images[0], num_layers=self.n_gru_layers)
            fused_left, fused_right, alpha_left, alpha_right = self._extract_fused_matching_features(
                left_images,
                right_images,
            )

        disp_outputs = self._predict_disparities(
            fused_left,
            fused_right,
            cnet_list,
            iters,
            flow_init=flow_init,
            test_mode=test_mode,
        )
        return disp_outputs, alpha_left, alpha_right

    def _collect_multiframe_inputs(self, inputs):
        left_images = []
        right_images = []
        for idx in range(1, self.fusion_num_exposures + 1):
            left_key = f'left_{idx}'
            right_key = f'right_{idx}'
            if left_key not in inputs or right_key not in inputs:
                raise KeyError(
                    f'RAFTStereoFusionDual expects input keys {left_key} and {right_key}'
                )
            left_images.append(inputs[left_key])
            right_images.append(inputs[right_key])
        return left_images, right_images

    def _attach_fusion_info(self, outputs, alpha_left, alpha_right):
        if self.return_fusion_alpha:
            outputs['fusion_alpha_left'] = alpha_left
            outputs['fusion_alpha_right'] = alpha_right
        return outputs

    def forward(
        self,
        inputs,
        image2=None,
        image1_next=None,
        image2_next=None,
        iters=None,
        flow_init=None,
        test_mode=False,
    ):
        if torch.is_tensor(inputs):
            if self.fusion_num_exposures != 2:
                raise ValueError('Raw tensor mode only supports FUSION_NUM_EXPOSURES=2')
            if image2 is None or image1_next is None or image2_next is None:
                raise ValueError(
                    'image2, image1_next, and image2_next are required when calling RAFTStereoFusionDual with raw tensors'
                )
            if iters is None:
                iters = self.eval_iters if test_mode else self.train_iters
            disp_outputs, _, _ = self._forward_fusion_impl(
                [inputs, image1_next],
                [image2, image2_next],
                iters,
                flow_init=flow_init,
                test_mode=test_mode,
            )
            return disp_outputs

        left_images, right_images = self._collect_multiframe_inputs(inputs)
        if iters is None:
            iters = self.train_iters if self.training and not test_mode else self.eval_iters

        if self.training and not test_mode:
            disp_preds, alpha_left, alpha_right = self._forward_fusion_impl(
                left_images,
                right_images,
                iters,
                flow_init=flow_init,
            )
            outputs = {
                'disp_preds': disp_preds,
                'disp_pred': disp_preds[-1],
            }
            return self._attach_fusion_info(outputs, alpha_left, alpha_right)

        (_, disp_pred), alpha_left, alpha_right = self._forward_fusion_impl(
            left_images,
            right_images,
            iters,
            flow_init=flow_init,
            test_mode=True,
        )
        outputs = {'disp_pred': disp_pred}
        return self._attach_fusion_info(outputs, alpha_left, alpha_right)




if __name__ == "__main__":
    from types import SimpleNamespace

    class Cfg(SimpleNamespace):
        def get(self, key, default=None):
            return getattr(self, key, default)

    cfgs = Cfg(
        MAX_DISP=192,
        LOSS_GAMMA=0.9,
        TRAIN_ITERS=1,
        EVAL_ITERS=1,
        CORR_IMPLEMENTATION="reg",
        CORR_LEVELS=4,
        CORR_RADIUS=4,
        CONTEXT_NORM="batch",
        HIDDEN_DIMS=[128, 128, 128],
        MIXED_PRECISION=False,
        N_DOWNSAMPLE=2,
        N_GRU_LAYERS=3,
        SHARED_BACKBONE=False,
        SLOW_FAST_GRU=False,
    )

    raft_model = RAFTStereo(cfgs).eval()
    dual_model = RAFTStereoDual(cfgs).eval()
    fusion_dual_model = RAFTStereoFusionDual(cfgs).eval()
    dummy_left = torch.randn(1, 3, 64, 96)
    dummy_right = torch.randn(1, 3, 64, 96)
    dummy_left_next = torch.randn(1, 3, 64, 96)
    dummy_right_next = torch.randn(1, 3, 64, 96)

    with torch.no_grad():
        raft_out = raft_model({'left': dummy_left, 'right': dummy_right})
        dual_out = dual_model({
            'left_1': dummy_left,
            'right_1': dummy_right,
            'left_2': dummy_left_next,
            'right_2': dummy_right_next,
        })
        fusion_dual_out = fusion_dual_model({
            'left_1': dummy_left,
            'right_1': dummy_right,
            'left_2': dummy_left_next,
            'right_2': dummy_right_next,
        })

    print('single:', raft_out['disp_pred'].shape)
    print('dual:', dual_out['disp_pred'].shape)
    print('fusion_dual:', fusion_dual_out['disp_pred'].shape)