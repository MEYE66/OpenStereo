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


def _as_float(value, default):
    if value is None:
        return float(default)
    return float(value)


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
        self.input_mode = str(cfg_get(cfgs, 'DEEPHDR_INPUT_MODE', 'normalized')).lower()
        self.exposure_mode = str(cfg_get(cfgs, 'DEEPHDR_EXPOSURE_MODE', 'linear_gain')).lower()
        self.output_mode = str(cfg_get(cfgs, 'DEEPHDR_OUTPUT_MODE', 'linear')).lower()
        default_input_scale = 65535.0 / 255.0 if self.input_mode in {
            'kalantari',
            'kalantari_16bit',
            'uint16_div255',
        } else 1.0
        self.ldr_input_scale = _as_float(
            cfg_get(cfgs, 'DEEPHDR_LDR_INPUT_SCALE', default_input_scale),
            default_input_scale,
        )
        self.output_eps = float(cfg_get(cfgs, 'DEEPHDR_OUTPUT_EPS', 1e-6))
        self.output_quantile = float(cfg_get(cfgs, 'DEEPHDR_OUTPUT_QUANTILE', 0.995))
        self.tonemap_mu = float(cfg_get(cfgs, 'DEEPHDR_TONEMAP_MU', 5000.0))

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

    def _compute_exposure(self, exposure_time, gain):
        if self.exposure_mode in {'linear', 'linear_gain', 'gain_linear'}:
            gain_scale = gain
        elif self.exposure_mode in {'db', 'db_gain', 'gain_db'}:
            gain_scale = torch.pow(torch.full_like(gain, 10.0), gain / 20.0)
        elif self.exposure_mode in {'time', 'time_only'}:
            gain_scale = torch.ones_like(gain)
        else:
            raise ValueError(f'Unsupported DEEPHDR_EXPOSURE_MODE: {self.exposure_mode}')
        return (exposure_time * gain_scale).clamp_min(self.exposure_eps)

    def _state_to_exposures(self, state):
        if state.ndim != 2 or state.shape[1] not in (2, 4):
            raise ValueError('DeepHDRFusion expects exposure state shape [B,2] or [B,4].')

        if state.shape[1] == 2:
            exposure = self._compute_exposure(state[:, 0], state[:, 1])
            return exposure, exposure

        exposure_1 = self._compute_exposure(state[:, 0], state[:, 1])
        exposure_2 = self._compute_exposure(state[:, 2], state[:, 3])
        return exposure_1, exposure_2

    def _to_deephdr_ldr(self, ldr01):
        if self.input_mode in {'normalized', 'zero_one', '01'}:
            return ldr01.clamp(0.0, 1.0) * 2.0 - 1.0
        if self.input_mode in {'kalantari', 'kalantari_16bit', 'uint16_div255'}:
            return ldr01.clamp(0.0, 1.0) * self.ldr_input_scale * 2.0 - 1.0
        raise ValueError(f'Unsupported DEEPHDR_INPUT_MODE: {self.input_mode}')

    def _deephdr_ldr_to_hdr(self, deephdr_ldr, exposure):
        exposure = exposure.to(dtype=deephdr_ldr.dtype, device=deephdr_ldr.device).view(-1, 1, 1, 1)
        hdr01 = ((deephdr_ldr + 1.0) * 0.5).clamp_min(0.0).pow(self.gamma)
        hdr01 = hdr01 / exposure.clamp_min(self.exposure_eps)
        return hdr01 * 2.0 - 1.0

    def _postprocess_output(self, fused):
        fused = (fused + 1.0) * 0.5
        if self.output_clamp:
            fused = fused.clamp(0.0, 1.0)

        if self.output_mode in {'linear', 'none', 'raw'}:
            return fused

        denom = fused.detach().amax(dim=(1, 2, 3), keepdim=True).clamp_min(self.output_eps)
        fused_norm = (fused / denom).clamp(0.0, 1.0)
        if self.output_mode in {'max_norm', 'maxnorm', 'normalize'}:
            return fused_norm
        if self.output_mode in {'percentile_norm', 'quantile_norm', 'robust_norm'}:
            quantile = min(max(self.output_quantile, 0.0), 1.0)
            denom = torch.quantile(
                fused.detach().flatten(1),
                quantile,
                dim=1,
            ).view(-1, 1, 1, 1).clamp_min(self.output_eps)
            return (fused / denom).clamp(0.0, 1.0)
        if self.output_mode in {'reinhard', 'reinhard_norm'}:
            return fused_norm / (1.0 + fused_norm)
        if self.output_mode in {'log', 'log_tonemap'}:
            return torch.log1p(self.tonemap_mu * fused_norm) / torch.log1p(
                fused_norm.new_tensor(self.tonemap_mu)
            )
        raise ValueError(f'Unsupported DEEPHDR_OUTPUT_MODE: {self.output_mode}')

    def fuse_pair(self, frame_1, frame_2, state):
        frame_1 = self._to_ldr01(frame_1)
        frame_2 = self._to_ldr01(frame_2)
        if not self.enabled:
            return 0.5 * (frame_1 + frame_2)

        exposure_1, exposure_2 = self._state_to_exposures(state)
        deephdr_ldr_1 = self._to_deephdr_ldr(frame_1)
        deephdr_ldr_2 = self._to_deephdr_ldr(frame_2)
        in_ldr = torch.cat([deephdr_ldr_1, deephdr_ldr_2], dim=1)
        in_hdr = torch.cat([
            self._deephdr_ldr_to_hdr(deephdr_ldr_1, exposure_1),
            self._deephdr_ldr_to_hdr(deephdr_ldr_2, exposure_2),
        ], dim=1)

        if self.trainable:
            fused = self.model(in_ldr, in_hdr)
        else:
            with torch.no_grad():
                fused = self.model(in_ldr, in_hdr)

        return self._postprocess_output(fused)

    def forward(self, left_1, right_1, left_2, right_2, state):
        left_hdr = self.fuse_pair(left_1, left_2, state)
        right_hdr = self.fuse_pair(right_1, right_2, state)
        return left_hdr, right_hdr
