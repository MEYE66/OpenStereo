import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class QuantizeSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, n=10):
        max_val = 2 ** n - 1
        output = torch.clamp(torch.floor(input + 0.5), min=0, max=max_val)
        # output = (output - output.min()) / (output.max() - output.min())
        output = output / max_val
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


class ImageFormationModel(nn.Module):
    def __init__(self, nbits=10):
        super(ImageFormationModel, self).__init__()
        self.nbits = nbits
        self.register_buffer('gaussian_var', torch.tensor(3e-5, dtype=torch.float32))
        self.register_buffer('poisson_scale', torch.tensor(3.3e-4, dtype=torch.float32))

    def forward(self, radiance, t_pred, g_pred):
        t_pred = t_pred.view(-1, 1, 1, 1)
        g_pred = g_pred.view(-1, 1, 1, 1)

        gauss_std = torch.sqrt(self.gaussian_var) * t_pred
        poisson_scale = self.poisson_scale * t_pred

        radiance = radiance * t_pred
        shot_noise = torch.poisson(radiance / poisson_scale) * poisson_scale * g_pred
        readout_noise = gauss_std * torch.randn_like(radiance) * g_pred
        adc_noise = gauss_std * torch.randn_like(radiance)

        noise_radiance = shot_noise + readout_noise + adc_noise
        noise_radiance = torch.clamp(noise_radiance, 0.0, None)
        noise_radiance = QuantizeSTE.apply(noise_radiance, self.nbits)
        return noise_radiance


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        return self.lrelu(self.bn(self.conv(x)))


class Encoder(nn.Module):
    def __init__(self, nums_in, nums_feat, nums_out, downsample_size=128):
        super().__init__()
        self.downsample_size = downsample_size
        self.conv1 = ConvBlock(nums_in, nums_feat)
        self.conv2 = ConvBlock(nums_feat, nums_feat * 2)
        self.conv3 = ConvBlock(nums_feat * 2, nums_out)
        self.fc = nn.Linear((downsample_size // 8) ** 2 * nums_out, nums_out, bias=False)

    def forward(self, images):
        x = F.interpolate(images, size=(self.downsample_size, self.downsample_size), mode='bilinear', align_corners=False)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.fc(x.reshape(x.size(0), -1))
        return x


class ObservationEncoder(nn.Module):
    def __init__(self, image_channels, state_channels, nums_feat=32, nums_out=32, downsample_size=128):
        super().__init__()
        self.encoder = Encoder(
            nums_in=image_channels + state_channels,
            nums_feat=nums_feat,
            nums_out=nums_out,
            downsample_size=downsample_size,
        )

    def forward(self, image_input, state_tensor):
        return self.encoder(torch.cat([image_input, state_tensor], dim=1))


class ExposureEnvModel(nn.Module):
    def __init__(
        self,
        time_limits=(5.0, 20.0),
        gain_limits=(1.0, 20.0),
        init_time=(12.0, 12.5),
        init_gain=(10.0, 10.0),
        nbits=8,
    ):
        super().__init__()
        self.image_formation = ImageFormationModel(nbits=nbits)
        self.register_buffer('time_limits', torch.tensor(time_limits, dtype=torch.float32))
        self.register_buffer('gain_limits', torch.tensor(gain_limits, dtype=torch.float32))
        self.register_buffer('init_time', torch.tensor(init_time, dtype=torch.float32))
        self.register_buffer('init_gain', torch.tensor(init_gain, dtype=torch.float32))
        self.register_buffer('ev_eps', torch.tensor(1e-6, dtype=torch.float32))

    def time_gain_to_ev(self, states_t, states_g):
        exposure = (states_t * states_g).clamp_min(float(self.ev_eps))
        return torch.log2(exposure)

    def get_initial_states(self, left_image, right_image):
        batch_size = left_image.size(0)
        dtype = left_image.dtype
        left_t = torch.full((batch_size,), float(self.init_time[0]), dtype=dtype, device=left_image.device)
        right_t = torch.full((batch_size,), float(self.init_time[1]), dtype=dtype, device=right_image.device)
        left_g = torch.full((batch_size,), float(self.init_gain[0]), dtype=dtype, device=left_image.device)
        right_g = torch.full((batch_size,), float(self.init_gain[1]), dtype=dtype, device=right_image.device)
        return torch.stack([left_t, right_t], dim=1), torch.stack([left_g, right_g], dim=1)

    def decode_states_t_g(self, states_t, states_g):
        return {
            'ev_lr': self.time_gain_to_ev(states_t, states_g),
            'time': states_t,
            'gain': states_g,
        }

    @staticmethod
    def _states_to_mean_delta(states):
        mean = 0.5 * (states[:, 0] + states[:, 1])
        delta = states[:, 0] - states[:, 1]
        return mean, delta

    @staticmethod
    def _mean_delta_to_states(mean, delta):
        left = mean + 0.5 * delta
        right = mean - 0.5 * delta
        return torch.stack([left, right], dim=1)

    def apply_action_to_states(self, states_t, states_g, action):
        time_mean, time_delta = self._states_to_mean_delta(states_t)
        gain_mean, gain_delta = self._states_to_mean_delta(states_g)

        next_time_mean = torch.clamp(time_mean + action[:, 0], self.time_limits[0], self.time_limits[1])
        next_time_delta = torch.clamp(time_delta + action[:, 1], -float(self.time_limits[1] - self.time_limits[0]), float(self.time_limits[1] - self.time_limits[0]))
        next_gain_mean = torch.clamp(gain_mean + action[:, 2], self.gain_limits[0], self.gain_limits[1])
        next_gain_delta = torch.clamp(gain_delta + action[:, 3], -float(self.gain_limits[1] - self.gain_limits[0]), float(self.gain_limits[1] - self.gain_limits[0]))

        next_t = self._mean_delta_to_states(next_time_mean, next_time_delta)
        next_g = self._mean_delta_to_states(next_gain_mean, next_gain_delta)
        next_t = torch.stack([
            torch.clamp(next_t[:, 0], self.time_limits[0], self.time_limits[1]),
            torch.clamp(next_t[:, 1], self.time_limits[0], self.time_limits[1]),
        ], dim=1)
        next_g = torch.stack([
            torch.clamp(next_g[:, 0], self.gain_limits[0], self.gain_limits[1]),
            torch.clamp(next_g[:, 1], self.gain_limits[0], self.gain_limits[1]),
        ], dim=1)
        return next_t, next_g

    def render_images(self, left_image, right_image, states_t, states_g):
        updated_left = self.image_formation(left_image, states_t[:, 0], states_g[:, 0])
        updated_right = self.image_formation(right_image, states_t[:, 1], states_g[:, 1])
        return updated_left, updated_right

    def reset(self, left_image, right_image):
        states_t, states_g = self.get_initial_states(left_image, right_image)
        updated_left, updated_right = self.render_images(left_image, right_image, states_t, states_g)
        return updated_left, updated_right, states_t, states_g

    def step(self, left_image, right_image, states_t, states_g, action):
        next_t, next_g = self.apply_action_to_states(states_t, states_g, action)
        updated_left, updated_right = self.render_images(left_image, right_image, next_t, next_g)
        return updated_left, updated_right, next_t, next_g


class _ActorCriticBase(nn.Module):
    def __init__(
        self,
        image_channels,
        nums_feat=32,
        nums_out=32,
        state_dim=5,
        downsample_size=256,
        time_limits=(5.0, 20.0),
        gain_limits=(1.0, 20.0),
    ):
        super().__init__()
        self.state_dim = state_dim
        self.observation_encoder = ObservationEncoder(
            image_channels=image_channels,
            state_channels=state_dim,
            nums_feat=nums_feat,
            nums_out=nums_out,
            downsample_size=downsample_size,
        )
        self.register_buffer('time_limits', torch.tensor(time_limits, dtype=torch.float32))
        self.register_buffer('gain_limits', torch.tensor(gain_limits, dtype=torch.float32))

    @staticmethod
    def _tile_state_vector(state_vector, height, width):
        return state_vector.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)

    @staticmethod
    def _prepare_disparity(disparity):
        if disparity.ndim == 3:
            return disparity.unsqueeze(1)
        return disparity

    def _normalize_state_vector(self, states_t, states_g, remaining_steps):
        time_range = (self.time_limits[1] - self.time_limits[0]).clamp_min(1e-6)
        gain_range = (self.gain_limits[1] - self.gain_limits[0]).clamp_min(1e-6)

        time_mean = 0.5 * (states_t[:, 0] + states_t[:, 1])
        time_delta = states_t[:, 0] - states_t[:, 1]
        gain_mean = 0.5 * (states_g[:, 0] + states_g[:, 1])
        gain_delta = states_g[:, 0] - states_g[:, 1]

        norm_time_mean = ((time_mean - self.time_limits[0]) / time_range).unsqueeze(1)
        norm_time_delta = (time_delta / time_range).unsqueeze(1)
        norm_gain_mean = ((gain_mean - self.gain_limits[0]) / gain_range).unsqueeze(1)
        norm_gain_delta = (gain_delta / gain_range).unsqueeze(1)

        if remaining_steps.ndim == 1:
            remaining_steps = remaining_steps.unsqueeze(1)

        return torch.cat([
            norm_time_mean,
            norm_time_delta,
            norm_gain_mean,
            norm_gain_delta,
            remaining_steps,
        ], dim=1)

    def _build_state_tensor(self, reference_image, states_t, states_g, remaining_steps):
        state_vector = self._normalize_state_vector(states_t, states_g, remaining_steps)
        _, _, height, width = reference_image.shape
        return self._tile_state_vector(state_vector, height, width)


class AdpAEActorModelDual(_ActorCriticBase):
    def __init__(
        self,
        nums_in=12,
        nums_feat=32,
        nums_out=32,
        action_dim=4,
        state_dim=5,
        downsample_size=128,
        init_log_std=-0.5,
        log_std_min=-5.0,
        log_std_max=2.0,
        time_limits=(5.0, 20.0),
        gain_limits=(1.0, 20.0),
        time_delta_limit=3.0,
        gain_delta_limit=2.0,
    ):
        super().__init__(
            image_channels=nums_in,
            nums_feat=nums_feat,
            nums_out=nums_out,
            state_dim=state_dim,
            downsample_size=downsample_size,
            time_limits=time_limits,
            gain_limits=gain_limits,
        )
        self.action_mean_head = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(nums_out, action_dim, bias=True),
        )
        self.action_log_std_head = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(nums_out, action_dim, bias=True),
        )
        self.register_buffer('init_log_std', torch.tensor(float(init_log_std), dtype=torch.float32))
        self.register_buffer('log_std_min', torch.tensor(float(log_std_min), dtype=torch.float32))
        self.register_buffer('log_std_max', torch.tensor(float(log_std_max), dtype=torch.float32))
        self.register_buffer('time_delta_limit', torch.tensor(float(time_delta_limit), dtype=torch.float32))
        self.register_buffer('gain_delta_limit', torch.tensor(float(gain_delta_limit), dtype=torch.float32))
        self.register_buffer('squash_eps', torch.tensor(1e-6, dtype=torch.float32))

    def _compute_action_log_std(self, action_log_std):
        return torch.clamp(
            action_log_std + self.init_log_std,
            min=float(self.log_std_min),
            max=float(self.log_std_max),
        )

    def _build_dist(self, action_mean, action_log_std):
        std = torch.exp(action_log_std)
        return Normal(action_mean, std)

    def _action_scale_vector(self, ref_tensor):
        return torch.tensor(
            [
                float(self.time_delta_limit),
                float(self.time_delta_limit),
                float(self.gain_delta_limit),
                float(self.gain_delta_limit),
            ],
            dtype=ref_tensor.dtype,
            device=ref_tensor.device,
        )

    def _build_action_delta(self, squashed_action):
        action_scale = self._action_scale_vector(squashed_action).view(1, -1)
        return squashed_action * action_scale

    def _compute_squashed_log_prob(self, dist, pre_tanh_action, squashed_action):
        base_log_prob = dist.log_prob(pre_tanh_action).sum(dim=-1)
        jacobian_tanh = torch.log((1.0 - squashed_action.pow(2)).clamp_min(float(self.squash_eps))).sum(dim=-1)
        action_scale = self._action_scale_vector(pre_tanh_action)
        jacobian_scale = torch.log(action_scale.clamp_min(float(self.squash_eps))).sum()
        return base_log_prob - jacobian_tanh - jacobian_scale

    def forward(
        self,
        left_image_1,
        right_image_1,
        left_image_2,
        right_image_2,
        states_t,
        states_g,
        remaining_steps,
        deterministic=False,
    ):
        image_input = torch.cat([left_image_1, right_image_1, left_image_2, right_image_2], dim=1)
        state_tensor = self._build_state_tensor(left_image_1, states_t, states_g, remaining_steps)
        feat = self.observation_encoder(image_input, state_tensor)
        action_mean = self.action_mean_head(feat)
        action_log_std = self._compute_action_log_std(self.action_log_std_head(feat))
        dist = self._build_dist(action_mean, action_log_std)

        pre_tanh_action = action_mean if deterministic else dist.rsample()
        squashed_action = torch.tanh(pre_tanh_action)
        action_delta = self._build_action_delta(squashed_action)
        action_log_prob = self._compute_squashed_log_prob(dist, pre_tanh_action, squashed_action)
        entropy = -action_log_prob
        return action_delta, action_log_prob, entropy

    @staticmethod
    def get_loss(action_log_prob, advantage, entropy, entropy_weight=0.01):
        if action_log_prob.ndim != 1:
            action_log_prob = action_log_prob.reshape(-1)
        if advantage.ndim != 1:
            advantage = advantage.reshape(-1)
        if entropy.ndim != 1:
            entropy = entropy.reshape(-1)

        actor_pg_loss = -torch.mean(action_log_prob * advantage.detach())
        entropy_reward = float(entropy_weight) * torch.mean(entropy)
        return actor_pg_loss - entropy_reward


class AdpAEValueModelDual(_ActorCriticBase):
    def __init__(
        self,
        nums_in=14,
        nums_feat=32,
        nums_out=32,
        state_dim=5,
        downsample_size=128,
        time_limits=(5.0, 20.0),
        gain_limits=(1.0, 20.0),
    ):
        super().__init__(
            image_channels=nums_in,
            nums_feat=nums_feat,
            nums_out=nums_out,
            state_dim=state_dim,
            downsample_size=downsample_size,
            time_limits=time_limits,
            gain_limits=gain_limits,
        )
        self.critic_head = nn.Sequential(
            nn.Linear(nums_out, 32, bias=False),
            nn.LeakyReLU(),
            nn.Linear(32, 1, bias=False),
        )

    def forward(
        self,
        left_image_1,
        right_image_1,
        left_image_2,
        right_image_2,
        disparity_1,
        disparity_2,
        states_t,
        states_g,
        remaining_steps,
    ):
        disparity_1 = self._prepare_disparity(disparity_1)
        disparity_2 = self._prepare_disparity(disparity_2)
        image_input = torch.cat(
            [left_image_1, right_image_1, left_image_2, right_image_2, disparity_1, disparity_2],
            dim=1,
        )
        state_tensor = self._build_state_tensor(left_image_1, states_t, states_g, remaining_steps)
        feature = self.observation_encoder(image_input, state_tensor)
        output = self.critic_head(feature)
        return output.squeeze(-1)

    @staticmethod
    def get_loss(value_pred, td_target):
        if value_pred.ndim != 1:
            value_pred = value_pred.reshape(-1)
        if td_target.ndim != 1:
            td_target = td_target.reshape(-1)
        return torch.mean((td_target.detach() - value_pred) ** 2)


class FusionExposureEnvModel(nn.Module):
    def __init__(
        self,
        time_limits=(1.0, 20.0),
        gain_limits=(1.0, 20.0),
        init_time=(7.5, 7.5),
        init_gain=(7.5, 7.5),
        nbits=8,
    ):
        super().__init__()
        self.image_formation = ImageFormationModel(nbits=nbits)
        self.register_buffer('time_limits', torch.tensor(time_limits, dtype=torch.float32))
        self.register_buffer('gain_limits', torch.tensor(gain_limits, dtype=torch.float32))
        self.register_buffer('init_time', torch.tensor(init_time, dtype=torch.float32))
        self.register_buffer('init_gain', torch.tensor(init_gain, dtype=torch.float32))

    def get_initial_state(self, reference_image):
        batch_size = reference_image.shape[0]
        dtype = reference_image.dtype
        device = reference_image.device
        time_1 = torch.full((batch_size,), float(self.init_time[0]), dtype=dtype, device=device)
        gain_1 = torch.full((batch_size,), float(self.init_gain[0]), dtype=dtype, device=device)
        time_2 = torch.full((batch_size,), float(self.init_time[1]), dtype=dtype, device=device)
        gain_2 = torch.full((batch_size,), float(self.init_gain[1]), dtype=dtype, device=device)
        return torch.stack([time_1, gain_1, time_2, gain_2], dim=1)

    def apply_action_to_state(self, state, action_delta):
        next_state = state + action_delta
        time_1 = torch.clamp(next_state[:, 0], self.time_limits[0], self.time_limits[1])
        gain_1 = torch.clamp(next_state[:, 1], self.gain_limits[0], self.gain_limits[1])
        time_2 = torch.clamp(next_state[:, 2], self.time_limits[0], self.time_limits[1])
        gain_2 = torch.clamp(next_state[:, 3], self.gain_limits[0], self.gain_limits[1])
        return torch.stack([time_1, gain_1, time_2, gain_2], dim=1)

    def _render_pair(self, left_image, right_image, exposure_time, gain):
        updated_left = self.image_formation(left_image, exposure_time, gain)
        updated_right = self.image_formation(right_image, exposure_time, gain)
        return updated_left, updated_right

    def render_dual_frames(self, left_1, right_1, left_2, right_2, state):
        frame_1 = self._render_pair(left_1, right_1, state[:, 0], state[:, 1])
        frame_2 = self._render_pair(left_2, right_2, state[:, 2], state[:, 3])
        return frame_1[0], frame_1[1], frame_2[0], frame_2[1]


class _FusionActorCriticBase(nn.Module):
    def __init__(
        self,
        image_channels,
        state_channels=4,
        nums_feat=32,
        nums_out=32,
        downsample_size=128,
        time_limits=(1.0, 20.0),
        gain_limits=(1.0, 20.0),
    ):
        super().__init__()
        self.observation_encoder = ObservationEncoder(
            image_channels=image_channels,
            state_channels=state_channels,
            nums_feat=nums_feat,
            nums_out=nums_out,
            downsample_size=downsample_size,
        )
        self.register_buffer('time_limits', torch.tensor(time_limits, dtype=torch.float32))
        self.register_buffer('gain_limits', torch.tensor(gain_limits, dtype=torch.float32))

    @staticmethod
    def _tile_state_vector(state_vector, height, width):
        return state_vector.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, height, width)

    @staticmethod
    def _prepare_disparity(disparity):
        if disparity.ndim == 3:
            return disparity.unsqueeze(1)
        return disparity

    def _normalize_exposure_state(self, state):
        time_range = (self.time_limits[1] - self.time_limits[0]).clamp_min(1e-6)
        gain_range = (self.gain_limits[1] - self.gain_limits[0]).clamp_min(1e-6)
        time_1 = (state[:, 0:1] - self.time_limits[0]) / time_range
        gain_1 = (state[:, 1:2] - self.gain_limits[0]) / gain_range
        time_2 = (state[:, 2:3] - self.time_limits[0]) / time_range
        gain_2 = (state[:, 3:4] - self.gain_limits[0]) / gain_range
        return torch.cat([time_1, gain_1, time_2, gain_2], dim=1).clamp(0.0, 1.0)

    def _build_state_tensor(self, reference_tensor, state):
        _, _, height, width = reference_tensor.shape
        state_vector = self._normalize_exposure_state(state)
        return self._tile_state_vector(state_vector, height, width)


class OurAEActorModel(_FusionActorCriticBase):
    def __init__(
        self,
        nums_in=12,
        nums_feat=32,
        nums_out=32,
        downsample_size=128,
        init_log_std=-1.0,
        log_std_min=-5.0,
        log_std_max=2.0,
        time_limits=(1.0, 20.0),
        gain_limits=(1.0, 20.0),
        time_delta_limit=1.0,
        gain_delta_limit=0.6,
    ):
        super().__init__(
            image_channels=nums_in,
            state_channels=4,
            nums_feat=nums_feat,
            nums_out=nums_out,
            downsample_size=downsample_size,
            time_limits=time_limits,
            gain_limits=gain_limits,
        )
        self.action_mean_head = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(nums_out, 4, bias=False),
        )
        self.action_log_std_head = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(nums_out, 4, bias=False),
        )
        self.register_buffer('init_log_std', torch.tensor(float(init_log_std), dtype=torch.float32))
        self.register_buffer('log_std_min', torch.tensor(float(log_std_min), dtype=torch.float32))
        self.register_buffer('log_std_max', torch.tensor(float(log_std_max), dtype=torch.float32))
        self.register_buffer('time_delta_limit', torch.tensor(float(time_delta_limit), dtype=torch.float32))
        self.register_buffer('gain_delta_limit', torch.tensor(float(gain_delta_limit), dtype=torch.float32))
        self.register_buffer('squash_eps', torch.tensor(1e-6, dtype=torch.float32))

    def _action_scale_vector(self, reference):
        return torch.tensor(
            [
                float(self.time_delta_limit),
                float(self.gain_delta_limit),
                float(self.time_delta_limit),
                float(self.gain_delta_limit),
            ],
            dtype=reference.dtype,
            device=reference.device,
        )

    def _compute_log_prob(self, dist, pre_tanh_action, squashed_action):
        base_log_prob = dist.log_prob(pre_tanh_action).sum(dim=-1)
        tanh_jacobian = torch.log(
            (1.0 - squashed_action.pow(2)).clamp_min(float(self.squash_eps))
        ).sum(dim=-1)
        action_scale = self._action_scale_vector(pre_tanh_action)
        scale_jacobian = torch.log(action_scale.clamp_min(float(self.squash_eps))).sum()
        return base_log_prob - tanh_jacobian - scale_jacobian

    def forward(
        self,
        left_image_1,
        right_image_1,
        left_image_2,
        right_image_2,
        exposure_state,
        deterministic=False,
    ):
        image_input = torch.cat([left_image_1, right_image_1, left_image_2, right_image_2], dim=1)
        state_tensor = self._build_state_tensor(left_image_1, exposure_state)
        feat = self.observation_encoder(image_input, state_tensor)

        action_mean = self.action_mean_head(feat)
        action_log_std = torch.clamp(
            self.action_log_std_head(feat) + self.init_log_std,
            min=float(self.log_std_min),
            max=float(self.log_std_max),
        )
        dist = Normal(action_mean, torch.exp(action_log_std))
        pre_tanh_action = action_mean if deterministic else dist.rsample()
        squashed_action = torch.tanh(pre_tanh_action)
        action_scale = self._action_scale_vector(squashed_action).view(1, 4)
        action_delta = squashed_action * action_scale
        log_prob = self._compute_log_prob(dist, pre_tanh_action, squashed_action)
        entropy = -log_prob
        return action_delta, log_prob, entropy


class OurAEValueModel(_FusionActorCriticBase):
    def __init__(
        self,
        nums_feat=32,
        nums_out=32,
        downsample_size=256,
        time_limits=(1.0, 20.0),
        gain_limits=(1.0, 20.0),
    ):
        super().__init__(
            image_channels=1,
            state_channels=4,
            nums_feat=nums_feat,
            nums_out=nums_out,
            downsample_size=downsample_size,
            time_limits=time_limits,
            gain_limits=gain_limits,
        )
        self.critic_head = nn.Sequential(
            nn.Linear(nums_out, 32, bias=False),
            nn.LeakyReLU(),
            nn.Linear(32, 1, bias=False),
            # nn.Sigmoid(),
        )

    def forward(self, disparity, exposure_state):
        disparity = self._prepare_disparity(disparity)
        state_tensor = self._build_state_tensor(disparity, exposure_state)
        feature = self.observation_encoder(disparity, state_tensor)
        return self.critic_head(feature).squeeze(-1)
