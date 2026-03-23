import torch
import torch.nn as nn


class GwcVolumeCostProcessor(nn.Module):
    def __init__(self, maxdisp=192, downsample=4, num_groups=40, use_concat_volume=True, *args, **kwargs):
        super().__init__()
        self.maxdisp = maxdisp
        self.downsample = downsample
        self.num_groups = num_groups
        self.use_concat_volume = use_concat_volume

    def groupwise_correlation(self, fea1, fea2, similarity_type='None'):
        B, C, H, W = fea1.shape
        num_groups = self.num_groups
        assert C % num_groups == 0
        channels_per_group = C // num_groups
        
        f1 = fea1.view([B, num_groups, channels_per_group, H, W])
        f2 = fea2.view([B, num_groups, channels_per_group, H, W])
        
        if similarity_type == 'cosine':
            # Cosine Similarity
            norm1 = torch.norm(f1, dim=2, keepdim=True)
            norm2 = torch.norm(f2, dim=2, keepdim=True)
            eps = 1e-5
            cost = (f1 * f2).sum(dim=2) / (norm1.squeeze(2) * norm2.squeeze(2) + eps)
        elif similarity_type == 'ncc':
            # Normalized Cross Correlation (zero-centered)
            f1_mean = f1.mean(dim=2, keepdim=True)
            f2_mean = f2.mean(dim=2, keepdim=True)
            f1_centered = f1 - f1_mean
            f2_centered = f2 - f2_mean
            norm1 = torch.norm(f1_centered, dim=2, keepdim=True)
            norm2 = torch.norm(f2_centered, dim=2, keepdim=True)
            eps = 1e-5
            cost = (f1_centered * f2_centered).sum(dim=2) / (norm1.squeeze(2) * norm2.squeeze(2) + eps)
        else:
            # Original inner product
            cost = (f1 * f2).mean(dim=2)

        assert cost.shape == (B, num_groups, H, W)
        return cost

    def build_gwc_volume(self, refimg_fea, targetimg_fea):
        B, C, H, W = refimg_fea.shape
        maxdisp = self.maxdisp // self.downsample
        num_groups = self.num_groups
        volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W], requires_grad=False)
        for i in range(maxdisp):
            if i > 0:
                volume[:, :, i, :, i:] = self.groupwise_correlation(
                    refimg_fea[:, :, :, i:],
                    targetimg_fea[:, :, :, :-i]
                )
            else:
                volume[:, :, i, :, :] = self.groupwise_correlation(
                    refimg_fea,
                    targetimg_fea
                )
        volume = volume.contiguous()
        return volume

    def build_concat_volume(self, refimg_fea, targetimg_fea):
        B, C, H, W = refimg_fea.shape
        maxdisp = self.maxdisp // self.downsample
        volume = refimg_fea.new_zeros([B, 2 * C, maxdisp, H, W], requires_grad=False)
        for i in range(maxdisp):
            if i > 0:
                volume[:, :C, i, :, i:] = refimg_fea[:, :, :, i:]
                volume[:, C:, i, :, i:] = targetimg_fea[:, :, :, :-i]
            else:
                volume[:, :C, i, :, :] = refimg_fea
                volume[:, C:, i, :, :] = targetimg_fea
        volume = volume.contiguous()
        return volume

    def forward(self, inputs):
        left_feature = inputs['ref_feature']
        right_feature = inputs['tgt_feature']
        gwc_volume = self.build_gwc_volume(
            left_feature['gwc_feature'], right_feature['gwc_feature']
        )
        if self.use_concat_volume:
            concat_volume = self.build_concat_volume(
                left_feature['concat_feature'], right_feature['concat_feature']
            )
            volume = torch.cat((gwc_volume, concat_volume), 1)
        else:
            volume = gwc_volume
        return {"cost_volume": volume}

    def input_output(self):
        return {
            "inputs": ["ref_feature", "tgt_feature"],
            "outputs": ["cost_volume"]
        }
