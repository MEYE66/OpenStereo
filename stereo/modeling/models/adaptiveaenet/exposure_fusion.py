import torch
import torch.nn as nn
import torch.nn.functional as F


class MertensExposureFusion(nn.Module):
    """Batch-aware Mertens exposure fusion for two LDR subframes."""

    def __init__(
        self,
        n_levels=4,
        w_sat=1.0,
        w_cont=1.0,
        w_exp=1.0,
        well_exposed_sigma=0.2,
        eps=1e-8,
        clamp_input=True,
        clamp_output=True,
    ):
        super().__init__()
        self.n_levels = max(int(n_levels), 1)
        self.w_sat = float(w_sat)
        self.w_cont = float(w_cont)
        self.w_exp = float(w_exp)
        self.well_exposed_sigma = float(well_exposed_sigma)
        self.eps = float(eps)
        self.clamp_input = bool(clamp_input)
        self.clamp_output = bool(clamp_output)
        self.register_buffer(
            'downsample_kernel',
            torch.tensor(
                [
                    [1.0, 2.0, 1.0],
                    [2.0, 4.0, 2.0],
                    [1.0, 2.0, 1.0],
                ],
                dtype=torch.float32,
            ).view(1, 1, 3, 3) / 16.0,
        )
        self.register_buffer(
            'laplacian_kernel',
            torch.tensor(
                [
                    [0.0, 1.0, 0.0],
                    [1.0, -4.0, 1.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=torch.float32,
            ).view(1, 1, 3, 3),
        )

    @staticmethod
    def _flatten_exposure_dim(tensor):
        batch_size, num_frames, channels, height, width = tensor.shape
        return tensor.reshape(batch_size * num_frames, channels, height, width)

    @staticmethod
    def _restore_exposure_dim(tensor, batch_size, num_frames):
        _, channels, height, width = tensor.shape
        return tensor.reshape(batch_size, num_frames, channels, height, width)

    @staticmethod
    def _pad_stage_to_downsample(stage):
        _, _, height, width = stage.shape
        pad_vertical = (1, 0) if height % 2 == 0 else (1, 1)
        pad_horizontal = (1, 0) if width % 2 == 0 else (1, 1)
        return F.pad(stage, (*pad_horizontal, *pad_vertical), mode='replicate')

    @staticmethod
    def _expand_stage(stage, target_shape):
        batch_size, num_frames, channels, height, width = stage.shape
        flat = stage.reshape(batch_size * num_frames, channels, height, width)
        expanded_height = 2 * height - 1
        expanded_width = 2 * width - 1
        expanded = F.interpolate(
            flat,
            size=(expanded_height, expanded_width),
            mode='bilinear',
            align_corners=True,
        )
        pad = (
            0,
            target_shape[1] - expanded_width,
            0,
            target_shape[0] - expanded_height,
        )
        if any(value > 0 for value in pad):
            expanded = F.pad(expanded, pad=pad, mode='replicate')
        return expanded.reshape(
            batch_size,
            num_frames,
            channels,
            target_shape[0],
            target_shape[1],
        )

    def _depthwise_kernel(self, kernel, channels, dtype, device):
        return kernel.to(dtype=dtype, device=device).expand(channels, 1, -1, -1)

    def _depthwise_conv2d(self, tensor, kernel, padding=0, stride=1):
        channels = tensor.shape[1]
        depthwise_kernel = self._depthwise_kernel(
            kernel,
            channels=channels,
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return F.conv2d(
            tensor,
            depthwise_kernel,
            padding=padding,
            stride=stride,
            groups=channels,
        )

    def _downsample(self, stage):
        batch_size, num_frames, _, _, _ = stage.shape
        flat = self._flatten_exposure_dim(stage)
        padded = self._pad_stage_to_downsample(flat)
        downsampled = self._depthwise_conv2d(
            padded,
            self.downsample_kernel,
            padding=0,
            stride=2,
        )
        return self._restore_exposure_dim(downsampled, batch_size, num_frames)

    def _compute_gaussian_pyramid(self, tensor):
        pyramid = [tensor]
        for _ in range(1, self.n_levels):
            pyramid.append(self._downsample(pyramid[-1]))
        return pyramid

    def _compute_laplacian_pyramid(self, gaussian_pyramid):
        laplacian_pyramid = [None] * len(gaussian_pyramid)
        laplacian_pyramid[-1] = gaussian_pyramid[-1]
        for level in range(len(gaussian_pyramid) - 2, -1, -1):
            target_shape = gaussian_pyramid[level].shape[-2:]
            expanded = self._expand_stage(gaussian_pyramid[level + 1], target_shape)
            laplacian_pyramid[level] = gaussian_pyramid[level] - expanded
        return laplacian_pyramid

    def _collapse_pyramid(self, laplacian_pyramid):
        current = laplacian_pyramid[-1]
        for stage in laplacian_pyramid[-2::-1]:
            current = self._expand_stage(current, stage.shape[-2:]) + stage
        return current

    def _compute_contrast(self, gray_burst):
        batch_size, num_frames, _, _, _ = gray_burst.shape
        flat_gray = self._flatten_exposure_dim(gray_burst)
        contrast = torch.abs(
            self._depthwise_conv2d(flat_gray, self.laplacian_kernel, padding=1)
        )
        return self._restore_exposure_dim(contrast, batch_size, num_frames)

    def _compute_saturation(self, burst, gray_burst):
        variance = torch.mean((burst - gray_burst) ** 2, dim=2, keepdim=True)
        return torch.sqrt(torch.clamp(variance, min=self.eps))

    def _compute_well_exposedness(self, burst):
        denominator = 2.0 * max(self.well_exposed_sigma, self.eps)
        return torch.exp(-torch.sum((burst - 0.5) ** 2, dim=2, keepdim=True) / denominator)

    def _compute_weights(self, burst):
        gray_burst = torch.mean(burst, dim=2, keepdim=True)
        contrast = self._compute_contrast(gray_burst)
        saturation = self._compute_saturation(burst, gray_burst)
        well_exposedness = self._compute_well_exposedness(burst)

        weights = (
            (contrast ** self.w_cont)
            * (saturation ** self.w_sat)
            * (well_exposedness ** self.w_exp)
        )
        weight_sum = weights.sum(dim=1, keepdim=True)
        uniform = torch.full_like(weights, 1.0 / burst.shape[1])
        normalized = weights / weight_sum.clamp_min(self.eps)
        return torch.where(weight_sum > self.eps, normalized, uniform)

    def _fuse_burst(self, burst):
        weights = self._compute_weights(burst)
        image_pyramid = self._compute_laplacian_pyramid(
            self._compute_gaussian_pyramid(burst)
        )
        weight_pyramid = self._compute_gaussian_pyramid(weights)
        fused_pyramid = [
            torch.sum(weight_stage * image_stage, dim=1, keepdim=True)
            for weight_stage, image_stage in zip(weight_pyramid, image_pyramid)
        ]
        fused = self._collapse_pyramid(fused_pyramid).squeeze(1)
        if self.clamp_output:
            fused = fused.clamp(0.0, 1.0)
        return fused

    def forward(self, frame_1, frame_2):
        if frame_1.shape != frame_2.shape:
            raise ValueError('MertensExposureFusion expects frames with identical shapes.')
        if frame_1.ndim != 4:
            raise ValueError('MertensExposureFusion expects inputs shaped [B, C, H, W].')

        burst = torch.stack([frame_1, frame_2], dim=1)
        if self.clamp_input:
            burst = burst.clamp(0.0, 1.0)
        return self._fuse_burst(burst)
