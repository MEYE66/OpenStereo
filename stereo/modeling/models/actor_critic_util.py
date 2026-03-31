import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import sys
from pathlib import Path


def _find_repo_root(start_path: Path) -> Path:
    for parent in [start_path] + list(start_path.parents):
        if (parent / 'stereo').is_dir():
            return parent
    raise RuntimeError('Could not locate repository root containing the stereo package.')


repo_root = _find_repo_root(Path(__file__).resolve())
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from stereo.modeling.models.ae_util import ImageFormationModel, exposure_value_equation
from stereo.modeling.models.gradnet.grad_ae import GradientExposureController




class FeatureExtractor(nn.Module):
    def __init__(self, shape=(14, 64, 64), mid_channels=32, output_dim=4096, dropout_prob=0.5):
        """
        Args:
            shape: (C, H, W) 输入特征的形状
        """
        super().__init__()
        
        in_channels, _, size = shape
        min_feature_map_size = 4
        
        # 校验维度，增加更清晰的错误提示
        assert output_dim % (min_feature_map_size ** 2) == 0, f'output_dim={output_dim} 必须能被 {min_feature_map_size**2} 整除'
        
        # 辅助函数：提取重复的卷积块定义 (DRY原则)
        def _make_conv_block(in_c, out_c):
            return [
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(negative_slope=0.2)
            ]
        
        layers = []
        channels = mid_channels

        # 动态构建卷积网络层
        while size > min_feature_map_size:
            assert size % 2 == 0, f"特征图尺寸 {size} 必须为偶数"
            
            # 区分是否为最后一次下采样阶段
            if size == min_feature_map_size * 2:
                out_channels = output_dim // (min_feature_map_size ** 2)
            else:
                out_channels = channels
            
            # 添加卷积块
            layers.extend(_make_conv_block(in_channels, out_channels))
            
            # 为下一层迭代准备变量
            in_channels = out_channels
            if size != min_feature_map_size * 2:
                channels *= 2
            size //= 2

        # 使用 nn.Sequential 整合所有层，包括展平(Flatten)和 Dropout
        self.extractor = nn.Sequential(
            *layers,
            nn.Flatten(),
            nn.Dropout(p=dropout_prob)
        )

    def forward(self, x):
        # forward 变得极其干净，直接全链路直通
        return self.extractor(x)


class PolicyModel(nn.Module):
    """Actor策略网络：同时估计双目四维曝光参数 [t_l, g_l, t_r, g_r]。"""

    def __init__(
        self,
        shape=(10, 64, 64),
        mid_channels=32,
        output_dim=4096,
        action_dim=4,
        entropy_coef=0.01,
        time_limits=(5.0, 20.0),
        gain_limits=(1.0, 20.0),
    ):
        super().__init__()
        if action_dim != 4:
            raise ValueError('PolicyModel requires action_dim=4 for [t_l, g_l, t_r, g_r].')

        self.state_dim = 4
        self.input_channels = int(shape[0])
        self.exposure_controller = GradientExposureController(
            target_grad=0.12,
            min_exposure=5.0,
            max_exposure=20.0,
        )
        self.action_selection = FeatureExtractor(shape=shape, mid_channels=mid_channels, output_dim=output_dim)
        self.actor_mean_head = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, action_dim)
        )
        self.actor_log_std = nn.Parameter(torch.full((action_dim,), -1.0, dtype=torch.float32))
        self.down_sample = nn.AdaptiveAvgPool2d((shape[1], shape[2]))

        self.image_formation = ImageFormationModel(nbits=8)
        self.entropy_coef = float(entropy_coef)

        self.register_buffer('time_limits', torch.tensor(time_limits, dtype=torch.float32))
        self.register_buffer('gain_limits', torch.tensor(gain_limits, dtype=torch.float32))

    def _expand_state_map(self, states, h, w):
        return states.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, w)

    def _get_initial_states(self, left_image, right_image):
        left_ev = self.exposure_controller(left_image)
        right_ev = self.exposure_controller(right_image)

        left_t, left_g = exposure_value_equation(left_ev, self.time_limits, self.gain_limits)
        right_t, right_g = exposure_value_equation(right_ev, self.time_limits, self.gain_limits)
        return torch.stack([left_t, left_g, right_t, right_g], dim=1)

    def _build_dist(self, mean):
        std = torch.exp(self.actor_log_std).clamp(min=1e-4)
        return Normal(mean, std.view(1, -1).expand_as(mean))

    def _apply_action_to_states(self, states, action):
        delta = torch.tanh(action)
        new_left_time = torch.clamp(states[:, 0] + delta[:, 0], self.time_limits[0], self.time_limits[1])
        new_left_gain = torch.clamp(states[:, 1] + delta[:, 1], self.gain_limits[0], self.gain_limits[1])
        new_right_time = torch.clamp(states[:, 2] + delta[:, 2], self.time_limits[0], self.time_limits[1])
        new_right_gain = torch.clamp(states[:, 3] + delta[:, 3], self.gain_limits[0], self.gain_limits[1])
        return torch.stack([new_left_time, new_left_gain, new_right_time, new_right_gain], dim=1)

    def _render_updated_images(self, left_image, right_image, updated_states):
        updated_left = self.image_formation(left_image, updated_states[:, 0], updated_states[:, 1])
        updated_right = self.image_formation(right_image, updated_states[:, 2], updated_states[:, 3])
        return updated_left, updated_right

    def forward(self, left_image, right_image, states=None, deterministic=False):
        if states is None:
            states = self._get_initial_states(left_image, right_image)
        if states.dim() != 2 or states.shape[1] != self.state_dim:
            raise ValueError('states must have shape (B, 4), representing [t_l, g_l, t_r, g_r].')

        state_map = self._expand_state_map(states, left_image.shape[-2], left_image.shape[-1])
        input_feature = torch.cat([left_image, right_image, state_map], dim=1)
        if input_feature.shape[1] != self.input_channels:
            raise ValueError(
                f'PolicyModel expected {self.input_channels} input channels, got {input_feature.shape[1]}. '
                'Please update shape=(C, H, W) to match [left, right, state].'
            )
        input_feature = self.down_sample(input_feature)

        features = self.action_selection(input_feature)
        action_mean = self.actor_mean_head(features)
        dist = self._build_dist(action_mean)

        action = action_mean if deterministic else dist.rsample()
        log_prob = dist.log_prob(action).sum(dim=1)
        entropy = dist.entropy().sum(dim=1)

        updated_states = self._apply_action_to_states(states, action)
        updated_left, updated_right = self._render_updated_images(left_image, right_image, updated_states)

        entrop_penalty = -self.entropy_coef * entropy

        return {
            'updated_images': (updated_left, updated_right),
            'updated_states': updated_states,
            'action': action,
            'action_mean': action_mean,
            'log_prob': log_prob,
            'entropy': entropy,
            'entrop_penalty': entrop_penalty,
        }



class ValueModel(nn.Module):
    def __init__(self, shape=(7, 64, 64)):
        super().__init__()
        self.encoder = FeatureExtractor(shape=shape, output_dim=128)
        self.critic_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 1)
        )
        self.down_sample = nn.AdaptiveAvgPool2d((shape[1], shape[2]))

    def forward(self, left_image, right_image, disparity):
        input_feature = torch.cat([left_image, right_image, disparity], dim=1)  # 假设输入是 (B, C, H, W)，这里 C=6
        input_feature = self.down_sample(input_feature)  # 下采样到指定大小
        feature = self.encoder(input_feature)
        out = self.critic_head(feature)
        return {
            "value": out.squeeze(1)  # 输出形状 (B,)
        }

if __name__ == "__main__":
    model = FeatureExtractor(shape=(14, 64, 64), output_dim=4096)
    dummy_input = torch.randn(2, 14, 64, 64)
    output = model(dummy_input)
    print(output.shape)  # 应该输出 torch.Size([2, 4096])



    value = ValueModel(shape=(7, 64, 64))
    dummy_left = torch.randn(2, 3, 64, 64)
    dummy_right = torch.randn(2, 3, 64, 64)
    dummy_disp = torch.randn(2, 1, 64, 64)
    value_output = value(dummy_left, dummy_right, dummy_disp)
    # print(value_output['value'].shape)  # 应该输出 torch.Size([2, 1])

    policy = PolicyModel(shape=(10, 64, 64), output_dim=4096, action_dim=4)
    radiance_left = torch.rand(2, 3, 64, 64)
    radiance_right = torch.rand(2, 3, 64, 64)
    state = torch.tensor([[10.0, 1.0, 11.0, 1.1], [12.0, 1.2, 13.0, 1.3]])
    policy_out = policy(radiance_left, radiance_right, state)
    print(policy_out['updated_states'].shape)
    print(policy_out['entrop_penalty'].shape)