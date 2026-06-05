import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


def _tensor_stats_message(tensor):
    detached = tensor.detach()
    finite_mask = torch.isfinite(detached)
    finite_count = int(finite_mask.sum().item())
    total_count = detached.numel()
    message = (
        'shape={} dtype={} device={} finite={}/{}'
    ).format(tuple(detached.shape), detached.dtype, detached.device, finite_count, total_count)
    if finite_count > 0:
        finite_values = detached[finite_mask].to(dtype=torch.float32)
        message += ' min={:.6g} max={:.6g} mean={:.6g} std={:.6g}'.format(
            finite_values.min().item(),
            finite_values.max().item(),
            finite_values.mean().item(),
            finite_values.std(unbiased=False).item(),
        )
    return message


def require_finite_tensor(name, tensor):
    if not torch.is_tensor(tensor):
        return
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(
            '{} contains non-finite values: {}'.format(name, _tensor_stats_message(tensor))
        )


class QuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inputs, nbits=8):
        max_val = 2 ** int(nbits) - 1
        outputs = torch.clamp(torch.floor(inputs + 0.5), min=0, max=max_val)
        return outputs / max_val

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class ImageFormationModel(nn.Module):
    def __init__(self, nbits=8):
        super().__init__()
        self.nbits = int(nbits)
        self.register_buffer('gaussian_var', torch.tensor(3e-5, dtype=torch.float32))
        self.register_buffer('poisson_scale', torch.tensor(3.3e-4, dtype=torch.float32))

    def forward(self, radiance, exposure_time, gain):
        exposure_time = exposure_time.view(-1, 1, 1, 1)
        gain = gain.view(-1, 1, 1, 1)

        gauss_std = torch.sqrt(self.gaussian_var) * exposure_time
        poisson_scale = torch.clamp(self.poisson_scale * exposure_time, min=1e-8)

        radiance = radiance * exposure_time
        shot_noise = torch.poisson(radiance / poisson_scale) * poisson_scale * gain
        readout_noise = gauss_std * torch.randn_like(radiance) * gain
        adc_noise = gauss_std * torch.randn_like(radiance)

        noisy_radiance = shot_noise + readout_noise + adc_noise
        noisy_radiance = torch.clamp(noisy_radiance, min=0.0)
        return QuantizeSTE.apply(noisy_radiance, self.nbits)


class AbsoluteExposureEnv(nn.Module):
    def __init__(
        self,
        time_limits=(1.0, 20.0),
        gain_limits=(1.0, 20.0),
        init_time=10.0,
        init_gain=10.0,
        nbits=10,
    ):
        super().__init__()
        self.image_formation = ImageFormationModel(nbits=nbits)
        self.register_buffer('time_limits', torch.tensor(time_limits, dtype=torch.float32))
        self.register_buffer('gain_limits', torch.tensor(gain_limits, dtype=torch.float32))
        init_time = self._to_pair_tensor(init_time, 'init_time')
        init_gain = self._to_pair_tensor(init_gain, 'init_gain')
        init_state = torch.stack([init_time[0], init_gain[0], init_time[1], init_gain[1]])
        self.register_buffer('init_state', init_state)

    @staticmethod
    def _to_pair_tensor(value, name):
        tensor = torch.as_tensor(value, dtype=torch.float32).flatten()
        if tensor.numel() == 1:
            return tensor.expand(2)
        if tensor.numel() == 2:
            return tensor
        raise ValueError(f'AbsoluteExposureEnv expects one or two {name} values.')

    def get_initial_state(self, reference_image):
        batch_size = reference_image.shape[0]
        return self.init_state.to(
            dtype=reference_image.dtype,
            device=reference_image.device,
        ).view(1, 4).expand(batch_size, 4)

    def clamp_action(self, action):
        exposure_time_1 = torch.clamp(action[:, 0], self.time_limits[0], self.time_limits[1])
        gain_1 = torch.clamp(action[:, 1], self.gain_limits[0], self.gain_limits[1])
        exposure_time_2 = torch.clamp(action[:, 2], self.time_limits[0], self.time_limits[1])
        gain_2 = torch.clamp(action[:, 3], self.gain_limits[0], self.gain_limits[1])
        return torch.stack([exposure_time_1, gain_1, exposure_time_2, gain_2], dim=1)

    def apply_action(self, action):
        return self.clamp_action(action)

    def render_image(self, radiance, exposure_time, gain):
        return self.image_formation(radiance, exposure_time, gain)

    def render_stereo_pair(self, left_radiance, right_radiance, exposure_time, gain):
        left_image = self.render_image(left_radiance, exposure_time, gain)
        right_image = self.render_image(right_radiance, exposure_time, gain)
        return left_image, right_image

    def render_dual_frames(
        self,
        left_radiance_1,
        right_radiance_1,
        left_radiance_2,
        right_radiance_2,
        state,
    ):
        left_1, right_1 = self.render_stereo_pair(
            left_radiance_1,
            right_radiance_1,
            state[:, 0],
            state[:, 1],
        )
        left_2, right_2 = self.render_stereo_pair(
            left_radiance_2,
            right_radiance_2,
            state[:, 2],
            state[:, 3],
        )
        return left_1, right_1, left_2, right_2


class ConvEncoder(nn.Module):
    def __init__(self, in_channels, feature_dim=32, output_dim=32, input_size=128):
        super().__init__()
        self.input_size = int(input_size)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, feature_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(feature_dim, feature_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(feature_dim * 2, output_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )
        self.fc = nn.Linear((self.input_size // 8) ** 2 * output_dim, output_dim, bias=False)

    def forward(self, inputs):
        inputs = F.interpolate(
            inputs,
            size=(self.input_size, self.input_size),
            mode='bilinear',
            align_corners=False,
        )
        features = self.net(inputs)
        return self.fc(features.reshape(features.shape[0], -1))


class A2CActor(nn.Module):
    def __init__(
        self,
        image_channels=12,
        feature_dim=32,
        hidden_dim=32,
        input_size=128,
        time_limits=(1.0, 20.0),
        gain_limits=(1.0, 20.0),
        time_classes=(1, 2, 3, 5, 8, 10, 15, 20),
        gain_classes=(1, 2, 4, 8, 16),
        debug_checks=True,
    ):
        super().__init__()
        self.debug_checks = bool(debug_checks)
        self.last_policy_stats = {}
        self.encoder = ConvEncoder(
            in_channels=image_channels + 4,
            feature_dim=feature_dim,
            output_dim=hidden_dim,
            input_size=input_size,
        )
        self.time_1_logits_head = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, len(time_classes), bias=False),
        )
        self.gain_1_logits_head = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, len(gain_classes), bias=False),
        )
        self.time_2_logits_head = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, len(time_classes), bias=False),
        )
        self.gain_2_logits_head = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, len(gain_classes), bias=False),
        )
        self.register_buffer('time_limits', torch.tensor(time_limits, dtype=torch.float32))
        self.register_buffer('gain_limits', torch.tensor(gain_limits, dtype=torch.float32))
        self.register_buffer('time_values', torch.tensor(time_classes, dtype=torch.float32))
        self.register_buffer('gain_values', torch.tensor(gain_classes, dtype=torch.float32))
        self.register_buffer('time_indices', torch.arange(len(time_classes), dtype=torch.float32))
        self.register_buffer('gain_indices', torch.arange(len(gain_classes), dtype=torch.float32))

    @staticmethod
    def _tile_state(state_vector, height, width):
        return state_vector.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)

    def _normalize_state(self, state):
        time_range = (self.time_limits[1] - self.time_limits[0]).clamp_min(1e-6)
        gain_range = (self.gain_limits[1] - self.gain_limits[0]).clamp_min(1e-6)
        norm_time = (state[:, 0:1] - self.time_limits[0]) / time_range
        norm_gain = (state[:, 1:2] - self.gain_limits[0]) / gain_range
        norm_time_2 = (state[:, 2:3] - self.time_limits[0]) / time_range
        norm_gain_2 = (state[:, 3:4] - self.gain_limits[0]) / gain_range
        return torch.cat([norm_time, norm_gain, norm_time_2, norm_gain_2], dim=1).clamp(0.0, 1.0)

    def _build_observation(self, left_image_1, right_image_1, left_image_2, right_image_2, state):
        _, _, height, width = left_image_1.shape
        state_tensor = self._tile_state(self._normalize_state(state), height, width)
        return torch.cat([left_image_1, right_image_1, left_image_2, right_image_2, state_tensor], dim=1)

    @staticmethod
    def _build_discrete_action(logits, values, indices, deterministic):
        dist = Categorical(logits=logits)
        probs = torch.softmax(logits, dim=-1)
        indices = indices.to(dtype=probs.dtype, device=probs.device)
        soft_index = (probs * indices.view(1, -1)).sum(dim=1)
        if deterministic:
            hard_index = torch.round(soft_index).clamp(0, values.numel() - 1).to(torch.long)
        else:
            hard_index = dist.sample()
        hard_one_hot = F.one_hot(hard_index, num_classes=values.numel()).to(dtype=probs.dtype)
        ste_one_hot = probs + (hard_one_hot - probs).detach()
        values = values.to(dtype=probs.dtype, device=probs.device)
        action_value = (ste_one_hot * values.view(1, -1)).sum(dim=1)
        soft_value = (probs * values.view(1, -1)).sum(dim=1)
        return {
            'value': action_value,
            'dist': dist,
            'class': hard_index,
            'soft_index': soft_index,
            'soft_value': soft_value,
            'prob_max': probs.max(dim=1).values,
            'entropy': dist.entropy(),
        }

    def _check_finite(self, name, tensor):
        if self.debug_checks:
            require_finite_tensor(name, tensor)

    @staticmethod
    def _per_sample_stats(tensor, prefix):
        detached = tensor.detach()
        return {
            f'{prefix}_mean': detached.mean(dim=1),
            f'{prefix}_std': detached.std(dim=1, unbiased=False),
            f'{prefix}_min': detached.min(dim=1).values,
            f'{prefix}_max': detached.max(dim=1).values,
        }

    def _record_policy_stats(self, head_outputs):
        stats = {}
        soft_values = torch.stack([output['soft_value'] for output in head_outputs.values()], dim=1)
        prob_maxes = torch.stack([output['prob_max'] for output in head_outputs.values()], dim=1)
        stats.update(self._per_sample_stats(soft_values, 'action_mean'))
        stats['action_saturation'] = (prob_maxes.detach() > 0.98).to(torch.float32).mean(dim=1)
        for name, output in head_outputs.items():
            stats[f'{name}_class'] = output['class'].detach().to(torch.float32)
            stats[f'{name}_soft_index'] = output['soft_index'].detach()
            stats[f'{name}_prob_max'] = output['prob_max'].detach()
            stats[f'{name}_entropy'] = output['entropy'].detach()
        self.last_policy_stats = stats

    def forward(self, left_image_1, right_image_1, left_image_2, right_image_2, state, deterministic=False):
        self._check_finite('actor.left_image_1', left_image_1)
        self._check_finite('actor.right_image_1', right_image_1)
        self._check_finite('actor.left_image_2', left_image_2)
        self._check_finite('actor.right_image_2', right_image_2)
        self._check_finite('actor.state', state)
        observation = self._build_observation(left_image_1, right_image_1, left_image_2, right_image_2, state)
        self._check_finite('actor.observation', observation)
        features = self.encoder(observation)
        self._check_finite('actor.features', features)
        logits = {
            'time_1': self.time_1_logits_head(features),
            'gain_1': self.gain_1_logits_head(features),
            'time_2': self.time_2_logits_head(features),
            'gain_2': self.gain_2_logits_head(features),
        }
        for name, value in logits.items():
            self._check_finite(f'actor.{name}_logits', value)

        head_outputs = {
            'time_1': self._build_discrete_action(
                logits['time_1'],
                self.time_values,
                self.time_indices,
                deterministic=deterministic,
            ),
            'gain_1': self._build_discrete_action(
                logits['gain_1'],
                self.gain_values,
                self.gain_indices,
                deterministic=deterministic,
            ),
            'time_2': self._build_discrete_action(
                logits['time_2'],
                self.time_values,
                self.time_indices,
                deterministic=deterministic,
            ),
            'gain_2': self._build_discrete_action(
                logits['gain_2'],
                self.gain_values,
                self.gain_indices,
                deterministic=deterministic,
            ),
        }
        action = torch.stack(
            [
                head_outputs['time_1']['value'],
                head_outputs['gain_1']['value'],
                head_outputs['time_2']['value'],
                head_outputs['gain_2']['value'],
            ],
            dim=1,
        )
        self._check_finite('actor.action', action)

        log_prob = action.new_zeros(action.shape[0])
        entropy = action.new_zeros(action.shape[0])
        for output in head_outputs.values():
            log_prob = log_prob + output['dist'].log_prob(output['class'].detach())
            entropy = entropy + output['entropy']
        self._check_finite('actor.log_prob', log_prob)
        self._check_finite('actor.entropy', entropy)
        self._record_policy_stats(head_outputs)
        return action, log_prob, entropy


class DispCritic(nn.Module):
    def __init__(self, feature_dim=32, hidden_dim=32, input_size=128):
        super().__init__()
        self.encoder = ConvEncoder(
            in_channels=1,
            feature_dim=feature_dim,
            output_dim=hidden_dim,
            input_size=input_size,
        )
        self.value_head = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    @staticmethod
    def _prepare_disparity(disparity):
        if disparity.ndim == 3:
            return disparity.unsqueeze(1)
        return disparity

    def forward(self, disparity):
        disparity = self._prepare_disparity(disparity)
        features = self.encoder(disparity)
        return self.value_head(features).squeeze(-1)
