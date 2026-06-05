import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn


def cfg_get(cfgs, key, default=None):
    if isinstance(cfgs, dict):
        return cfgs.get(key, default)
    return getattr(cfgs, key, default)


def _as_bool(value):
    if isinstance(value, str):
        return value.lower() in {'1', 'true', 'yes', 'y', 'on'}
    return bool(value)


def _as_hw(value, default=(256, 256)):
    if value is None:
        return default
    if isinstance(value, (tuple, list)):
        if len(value) == 1:
            return int(value[0]), int(value[0])
        return int(value[0]), int(value[1])
    value = str(value).lower().replace('x', ',')
    h, w = value.split(',', maxsplit=1)
    return int(h), int(w)


def _load_torch_checkpoint(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class DeepHDRFusion(nn.Module):
    """Fuse two LDR frames into one HDR-like LDR image with the local DeepHDR model."""

    def __init__(self, cfgs):
        super().__init__()
        self.enabled = _as_bool(cfg_get(cfgs, 'USE_DEEPHDR', True))
        self.trainable = _as_bool(cfg_get(cfgs, 'DEEPHDR_TRAINABLE', False))
        self.gamma = float(cfg_get(cfgs, 'DEEPHDR_GAMMA', 2.2))
        self.exposure_eps = float(cfg_get(cfgs, 'DEEPHDR_EXPOSURE_EPS', 1e-6))
        self.output_clamp = _as_bool(cfg_get(cfgs, 'DEEPHDR_OUTPUT_CLAMP', True))
        self.strict_load = _as_bool(cfg_get(cfgs, 'DEEPHDR_STRICT_LOAD', False))

        if not self.enabled:
            self.model = None
            return

        repo_root = Path(__file__).resolve().parents[3]
        deephdr_root = Path(cfg_get(cfgs, 'DEEPHDR_ROOT', repo_root / 'DeepHDR-pytorch')).expanduser()
        if not deephdr_root.is_absolute():
            deephdr_root = repo_root / deephdr_root
        if str(deephdr_root) not in sys.path:
            sys.path.insert(0, str(deephdr_root))

        from models.DeepHDR import DeepHDR

        deephdr_cfg = SimpleNamespace(
            image_size=_as_hw(cfg_get(cfgs, 'DEEPHDR_IMAGE_SIZE', (256, 256))),
            c_dim=int(cfg_get(cfgs, 'DEEPHDR_C_DIM', 3)),
            num_shots=2,
        )
        self.model = DeepHDR(deephdr_cfg)
        self._load_checkpoint(cfg_get(cfgs, 'DEEPHDR_CKPT', ''))

        for param in self.model.parameters():
            param.requires_grad = self.trainable
        if not self.trainable:
            self.model.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.enabled and not self.trainable:
            self.model.eval()
        return self

    def _load_checkpoint(self, checkpoint_path):
        if checkpoint_path is None or str(checkpoint_path).strip() == '':
            return
        checkpoint_path = Path(str(checkpoint_path)).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path(__file__).resolve().parents[3] / checkpoint_path
        checkpoint = _load_torch_checkpoint(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
        self.model.load_state_dict(state_dict, strict=self.strict_load)

    @staticmethod
    def _to_ldr01(image):
        image = image.to(dtype=torch.float32)
        max_value = image.detach().amax()
        if max_value > 1.5:
            image = image / 255.0
        return image.clamp(0.0, 1.0)

    def _state_to_exposures(self, state):
        if state.ndim != 2 or state.shape[1] not in (2, 4):
            raise ValueError('DeepHDRFusion expects exposure state shape [B,2] or [B,4].')

        if state.shape[1] == 2:
            exposure = (state[:, 0] * state[:, 1]).clamp_min(self.exposure_eps)
            return exposure, exposure

        exposure_1 = (state[:, 0] * state[:, 1]).clamp_min(self.exposure_eps)
        exposure_2 = (state[:, 2] * state[:, 3]).clamp_min(self.exposure_eps)
        return exposure_1, exposure_2

    def _ldr_to_deephdr_hdr(self, ldr01, exposure):
        exposure = exposure.to(dtype=ldr01.dtype, device=ldr01.device).view(-1, 1, 1, 1)
        hdr01 = ldr01.clamp(0.0, 1.0).pow(self.gamma) / exposure.clamp_min(self.exposure_eps)
        return hdr01 * 2.0 - 1.0

    def fuse_pair(self, frame_1, frame_2, state):
        frame_1 = self._to_ldr01(frame_1)
        frame_2 = self._to_ldr01(frame_2)
        if not self.enabled:
            return 0.5 * (frame_1 + frame_2)

        exposure_1, exposure_2 = self._state_to_exposures(state)
        in_ldr = torch.cat([frame_1 * 2.0 - 1.0, frame_2 * 2.0 - 1.0], dim=1)
        in_hdr = torch.cat([
            self._ldr_to_deephdr_hdr(frame_1, exposure_1),
            self._ldr_to_deephdr_hdr(frame_2, exposure_2),
        ], dim=1)

        if self.trainable:
            fused = self.model(in_ldr, in_hdr)
        else:
            with torch.no_grad():
                fused = self.model(in_ldr, in_hdr)

        fused = (fused + 1.0) * 0.5
        if self.output_clamp:
            fused = fused.clamp(0.0, 1.0)
        return fused

    def forward(self, left_1, right_1, left_2, right_2, state):
        left_hdr = self.fuse_pair(left_1, left_2, state)
        right_hdr = self.fuse_pair(right_1, right_2, state)
        return left_hdr, right_hdr
