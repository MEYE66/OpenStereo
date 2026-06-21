import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


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
        self.register_buffer('init_state', torch.tensor([init_time, init_gain], dtype=torch.float32))

    def get_initial_state(self, reference_image):
        batch_size = reference_image.shape[0]
        return self.init_state.to(
            dtype=reference_image.dtype,
            device=reference_image.device,
        ).view(1, 2).expand(batch_size, 2)

    def clamp_action(self, action):
        exposure_time = torch.clamp(action[:, 0], self.time_limits[0], self.time_limits[1])
        gain = torch.clamp(action[:, 1], self.gain_limits[0], self.gain_limits[1])
        return torch.stack([exposure_time, gain], dim=1)

    def apply_action(self, action):
        return self.clamp_action(action)

    def render_image(self, radiance, state):
        return self.image_formation(radiance, state[:, 0], state[:, 1])

    def render_stereo_pair(self, left_radiance, right_radiance, state):
        left_image = self.render_image(left_radiance, state)
        right_image = self.render_image(right_radiance, state)
        return left_image, right_image

    def render_dual_frames(
        self,
        left_radiance_1,
        right_radiance_1,
        left_radiance_2,
        right_radiance_2,
        state,
    ):
        left_1, right_1 = self.render_stereo_pair(left_radiance_1, right_radiance_1, state)
        left_2, right_2 = self.render_stereo_pair(left_radiance_2, right_radiance_2, state)
        return left_1, right_1, left_2, right_2


class DualExposureEnv(nn.Module):
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
        self.register_buffer('init_state', torch.tensor(
            [init_time, init_gain, init_time, init_gain, init_time, init_gain],
            dtype=torch.float32,
        ))

    def get_initial_state(self, reference_image):
        batch_size = reference_image.shape[0]
        return self.init_state.to(
            dtype=reference_image.dtype,
            device=reference_image.device,
        ).view(1, 6).expand(batch_size, 6)

    def clamp_exp1_action(self, action):
        exposure_time = torch.clamp(action[:, 0], self.time_limits[0], self.time_limits[1])
        gain = torch.clamp(action[:, 1], self.gain_limits[0], self.gain_limits[1])
        return torch.stack([exposure_time, gain], dim=1)

    def clamp_exp2_action(self, action):
        left_time = torch.clamp(action[:, 0], self.time_limits[0], self.time_limits[1])
        left_gain = torch.clamp(action[:, 1], self.gain_limits[0], self.gain_limits[1])
        right_time = torch.clamp(action[:, 2], self.time_limits[0], self.time_limits[1])
        right_gain = torch.clamp(action[:, 3], self.gain_limits[0], self.gain_limits[1])
        return torch.stack([left_time, left_gain, right_time, right_gain], dim=1)

    def clamp_action(self, action):
        exp1_action = self.clamp_exp1_action(action[:, 0:2])
        exp2_action = self.clamp_exp2_action(action[:, 2:6])
        return torch.cat([exp1_action, exp2_action], dim=1)

    def apply_action(self, action):
        return self.clamp_action(action)

    def render_image(self, radiance, exposure_time, gain):
        return self.image_formation(radiance, exposure_time, gain)

    def render_first_pair(self, left_radiance, right_radiance, exp1_state):
        exp1_state = self.clamp_exp1_action(exp1_state)
        left_image = self.render_image(left_radiance, exp1_state[:, 0], exp1_state[:, 1])
        right_image = self.render_image(right_radiance, exp1_state[:, 0], exp1_state[:, 1])
        return left_image, right_image

    def render_second_pair(self, left_radiance, right_radiance, state):
        state = self.clamp_action(state)
        left_image = self.render_image(left_radiance, state[:, 2], state[:, 3])
        right_image = self.render_image(right_radiance, state[:, 4], state[:, 5])
        return left_image, right_image

    def render_dual_frames(
        self,
        left_radiance_1,
        right_radiance_1,
        left_radiance_2,
        right_radiance_2,
        state,
    ):
        state = self.clamp_action(state)
        left_1, right_1 = self.render_first_pair(left_radiance_1, right_radiance_1, state[:, 0:2])
        left_2, right_2 = self.render_second_pair(left_radiance_2, right_radiance_2, state)
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
        image_channels=6,
        feature_dim=32,
        hidden_dim=32,
        input_size=256,
        init_log_std=-1.0,
        log_std_min=-5.0,
        log_std_max=2.0,
        time_limits=(1.0, 20.0),
        gain_limits=(1.0, 20.0),
        debug_checks=True,
    ):
        super().__init__()
        self.debug_checks = bool(debug_checks)
        self.last_policy_stats = {}
        self.encoder = ConvEncoder(
            in_channels=image_channels + 2,
            feature_dim=feature_dim,
            output_dim=hidden_dim,
            input_size=input_size,
        )
        self.action_mean_head = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 2, bias=False),
        )
        self.action_log_std_head = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 2, bias=False),
        )
        self.register_buffer('time_limits', torch.tensor(time_limits, dtype=torch.float32))
        self.register_buffer('gain_limits', torch.tensor(gain_limits, dtype=torch.float32))
        self.register_buffer('init_log_std', torch.tensor(float(init_log_std), dtype=torch.float32))
        self.register_buffer('log_std_min', torch.tensor(float(log_std_min), dtype=torch.float32))
        self.register_buffer('log_std_max', torch.tensor(float(log_std_max), dtype=torch.float32))
        self.register_buffer('squash_eps', torch.tensor(1e-6, dtype=torch.float32))

    @staticmethod
    def _tile_state(state_vector, height, width):
        return state_vector.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)

    def _normalize_state(self, state):
        time_range = (self.time_limits[1] - self.time_limits[0]).clamp_min(1e-6)
        gain_range = (self.gain_limits[1] - self.gain_limits[0]).clamp_min(1e-6)
        norm_time = (state[:, 0:1] - self.time_limits[0]) / time_range
        norm_gain = (state[:, 1:2] - self.gain_limits[0]) / gain_range
        return torch.cat([norm_time, norm_gain], dim=1).clamp(0.0, 1.0)

    def _action_center(self, reference):
        return torch.tensor(
            [
                0.5 * (float(self.time_limits[0]) + float(self.time_limits[1])),
                0.5 * (float(self.gain_limits[0]) + float(self.gain_limits[1])),
            ],
            dtype=reference.dtype,
            device=reference.device,
        )

    def _action_scale(self, reference):
        return torch.tensor(
            [
                0.5 * (float(self.time_limits[1]) - float(self.time_limits[0])),
                0.5 * (float(self.gain_limits[1]) - float(self.gain_limits[0])),
            ],
            dtype=reference.dtype,
            device=reference.device,
        )

    def _build_observation(self, left_image, right_image, state):
        _, _, height, width = left_image.shape
        state_tensor = self._tile_state(self._normalize_state(state), height, width)
        return torch.cat([left_image, right_image, state_tensor], dim=1)

    def _build_action(self, squashed_action):
        center = self._action_center(squashed_action).view(1, 2)
        scale = self._action_scale(squashed_action).view(1, 2)
        return center + squashed_action * scale

    def _compute_log_prob(self, dist, pre_tanh_action, squashed_action):
        base_log_prob = dist.log_prob(pre_tanh_action).sum(dim=-1)
        tanh_jacobian = torch.log(
            (1.0 - squashed_action.pow(2)).clamp_min(float(self.squash_eps))
        ).sum(dim=-1)
        scale = self._action_scale(pre_tanh_action)
        scale_jacobian = torch.log(scale.clamp_min(float(self.squash_eps))).sum()
        return base_log_prob - tanh_jacobian - scale_jacobian

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

    def _record_policy_stats(self, action_mean, action_log_std, squashed_action):
        stats = {}
        stats.update(self._per_sample_stats(action_mean, 'action_mean'))
        stats.update(self._per_sample_stats(action_log_std, 'action_log_std'))
        stats['action_saturation'] = (squashed_action.detach().abs() > 0.98).to(torch.float32).mean(dim=1)
        self.last_policy_stats = stats

    def forward(self, left_image, right_image, state, deterministic=False):
        self._check_finite('actor.left_image', left_image)
        self._check_finite('actor.right_image', right_image)
        self._check_finite('actor.state', state)
        observation = self._build_observation(left_image, right_image, state)
        self._check_finite('actor.observation', observation)
        features = self.encoder(observation)
        self._check_finite('actor.features', features)
        action_mean = self.action_mean_head(features)
        self._check_finite('actor.action_mean', action_mean)
        action_log_std = torch.clamp(
            self.action_log_std_head(features) + self.init_log_std,
            min=float(self.log_std_min),
            max=float(self.log_std_max),
        )
        self._check_finite('actor.action_log_std', action_log_std)
        dist = Normal(action_mean, torch.exp(action_log_std))
        if deterministic:
            pre_tanh_action = action_mean
        else:
            pre_tanh_action = dist.sample().detach()
        self._check_finite('actor.pre_tanh_action', pre_tanh_action)
        squashed_action = torch.tanh(pre_tanh_action)
        self._check_finite('actor.squashed_action', squashed_action)
        action = self._build_action(squashed_action)
        self._check_finite('actor.action', action)
        log_prob = self._compute_log_prob(
            dist,
            pre_tanh_action.detach(),
            squashed_action.detach(),
        )
        self._check_finite('actor.log_prob', log_prob)
        entropy = dist.entropy().sum(dim=-1)
        self._check_finite('actor.entropy', entropy)
        self._record_policy_stats(action_mean, action_log_std, squashed_action)
        return action, log_prob, entropy


class DualExposureA2CActor(nn.Module):
    def __init__(
        self,
        image_channels=6,
        feature_dim=32,
        hidden_dim=32,
        input_size=256,
        init_log_std=-1.0,
        log_std_min=-5.0,
        log_std_max=2.0,
        time_limits=(1.0, 20.0),
        gain_limits=(1.0, 20.0),
        debug_checks=True,
    ):
        super().__init__()
        self.debug_checks = bool(debug_checks)
        self.last_policy_stats = {}
        self.encoder = ConvEncoder(
            in_channels=image_channels * 2 + 6,
            feature_dim=feature_dim,
            output_dim=hidden_dim,
            input_size=input_size,
        )
        self.action_mean_head = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 6, bias=False),
        )
        self.action_log_std_head = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 6, bias=False),
        )
        self.register_buffer('time_limits', torch.tensor(time_limits, dtype=torch.float32))
        self.register_buffer('gain_limits', torch.tensor(gain_limits, dtype=torch.float32))
        self.register_buffer('init_log_std', torch.tensor(float(init_log_std), dtype=torch.float32))
        self.register_buffer('log_std_min', torch.tensor(float(log_std_min), dtype=torch.float32))
        self.register_buffer('log_std_max', torch.tensor(float(log_std_max), dtype=torch.float32))
        self.register_buffer('squash_eps', torch.tensor(1e-6, dtype=torch.float32))

    @staticmethod
    def _tile_state(state_vector, height, width):
        return state_vector.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)

    def _normalize_time_gain_pairs(self, state):
        time_range = (self.time_limits[1] - self.time_limits[0]).clamp_min(1e-6)
        gain_range = (self.gain_limits[1] - self.gain_limits[0]).clamp_min(1e-6)
        normalized = state.new_empty(state.shape)
        normalized[:, 0::2] = (state[:, 0::2] - self.time_limits[0]) / time_range
        normalized[:, 1::2] = (state[:, 1::2] - self.gain_limits[0]) / gain_range
        return normalized.clamp(0.0, 1.0)

    def _build_observation(
        self,
        left_radiance_1,
        right_radiance_1,
        left_radiance_2,
        right_radiance_2,
        state,
    ):
        _, _, height, width = left_radiance_1.shape
        state_tensor = self._tile_state(self._normalize_time_gain_pairs(state), height, width)
        return torch.cat(
            [
                left_radiance_1,
                right_radiance_1,
                left_radiance_2,
                right_radiance_2,
                state_tensor,
            ],
            dim=1,
        )

    def _action_center(self, dims, reference):
        base = [
            0.5 * (float(self.time_limits[0]) + float(self.time_limits[1])),
            0.5 * (float(self.gain_limits[0]) + float(self.gain_limits[1])),
        ]
        values = (base * (dims // 2))[:dims]
        return torch.tensor(values, dtype=reference.dtype, device=reference.device)

    def _action_scale(self, dims, reference):
        base = [
            0.5 * (float(self.time_limits[1]) - float(self.time_limits[0])),
            0.5 * (float(self.gain_limits[1]) - float(self.gain_limits[0])),
        ]
        values = (base * (dims // 2))[:dims]
        return torch.tensor(values, dtype=reference.dtype, device=reference.device)

    def _build_action(self, squashed_action):
        dims = squashed_action.shape[1]
        center = self._action_center(dims, squashed_action).view(1, dims)
        scale = self._action_scale(dims, squashed_action).view(1, dims)
        return center + squashed_action * scale

    def _compute_log_prob(self, dist, pre_tanh_action, squashed_action):
        base_log_prob = dist.log_prob(pre_tanh_action).sum(dim=-1)
        tanh_jacobian = torch.log(
            (1.0 - squashed_action.pow(2)).clamp_min(float(self.squash_eps))
        ).sum(dim=-1)
        scale = self._action_scale(squashed_action.shape[1], pre_tanh_action)
        scale_jacobian = torch.log(scale.clamp_min(float(self.squash_eps))).sum()
        return base_log_prob - tanh_jacobian - scale_jacobian

    def _check_finite(self, name, tensor):
        if self.debug_checks:
            require_finite_tensor(name, tensor)

    def _build_dist(self, features):
        action_mean = self.action_mean_head(features)
        action_log_std = torch.clamp(
            self.action_log_std_head(features) + self.init_log_std,
            min=float(self.log_std_min),
            max=float(self.log_std_max),
        )
        self._check_finite('actor.action_mean', action_mean)
        self._check_finite('actor.action_log_std', action_log_std)
        return Normal(action_mean, torch.exp(action_log_std)), action_mean, action_log_std

    @staticmethod
    def _sample_pre_action(dist, mean, deterministic):
        if deterministic:
            return mean
        return dist.sample().detach()

    def _sample_action(self, dist, mean, deterministic):
        pre_tanh_action = self._sample_pre_action(dist, mean, deterministic)
        self._check_finite('actor.pre_tanh_action', pre_tanh_action)
        squashed_action = torch.tanh(pre_tanh_action)
        self._check_finite('actor.squashed_action', squashed_action)
        action = self._build_action(squashed_action)
        self._check_finite('actor.action', action)
        log_prob = self._compute_log_prob(dist, pre_tanh_action.detach(), squashed_action.detach())
        entropy = dist.entropy().sum(dim=-1)
        self._check_finite('actor.log_prob', log_prob)
        self._check_finite('actor.entropy', entropy)
        return {
            'action': action,
            'log_prob': log_prob,
            'entropy': entropy,
            'mean': mean,
            'log_std': dist.scale.log(),
            'squashed': squashed_action,
        }

    @staticmethod
    def _per_sample_stats(tensor, prefix):
        detached = tensor.detach()
        return {
            f'{prefix}_mean': detached.mean(dim=1),
            f'{prefix}_std': detached.std(dim=1, unbiased=False),
            f'{prefix}_min': detached.min(dim=1).values,
            f'{prefix}_max': detached.max(dim=1).values,
        }

    def _record_policy_stats(self, action_mean, action_log_std, squashed_action, action):
        stats = {}
        stats.update(self._per_sample_stats(action_mean, 'action_mean'))
        stats.update(self._per_sample_stats(action_log_std, 'action_log_std'))
        stats.update(self._per_sample_stats(action, 'action'))
        stats['action_saturation'] = (squashed_action.detach().abs() > 0.98).to(torch.float32).mean(dim=1)
        self.last_policy_stats = stats

    def forward(
        self,
        left_radiance_1,
        right_radiance_1,
        left_radiance_2,
        right_radiance_2,
        state,
        deterministic=False,
    ):
        self._check_finite('actor.left_radiance_1', left_radiance_1)
        self._check_finite('actor.right_radiance_1', right_radiance_1)
        self._check_finite('actor.left_radiance_2', left_radiance_2)
        self._check_finite('actor.right_radiance_2', right_radiance_2)
        self._check_finite('actor.state', state)
        observation = self._build_observation(
            left_radiance_1,
            right_radiance_1,
            left_radiance_2,
            right_radiance_2,
            state,
        )
        self._check_finite('actor.observation', observation)
        features = self.encoder(observation)
        self._check_finite('actor.features', features)
        dist, mean, log_std = self._build_dist(features)
        result = self._sample_action(dist, mean, deterministic)
        self._record_policy_stats(
            result['mean'],
            log_std,
            result['squashed'],
            result['action'],
        )
        return result['action'], result['log_prob'], result['entropy']


class DispCritic(nn.Module):
    def __init__(self, feature_dim=32, hidden_dim=32, input_size=256):
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
