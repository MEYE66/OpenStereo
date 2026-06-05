import random
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Optional, Tuple
import sys
from pathlib import Path

import numpy as np
import torch


def _find_repo_root(start_path: Path) -> Path:
    for parent in [start_path] + list(start_path.parents):
        if (parent / 'stereo').is_dir():
            return parent
    raise RuntimeError('Could not locate repository root containing the stereo package.')


repo_root = _find_repo_root(Path(__file__).resolve())
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from stereo.datasets.carla_stereo_dataset import CarlaStereoDataset


@dataclass
class ExposureStateConfig:
    time_limits: Tuple[float, float] = (5.0, 20.0)
    gain_limits: Tuple[float, float] = (1.0, 20.0)
    init_time: float = 12.5
    init_gain: float = 1.0

    def clamp_state(self, state: torch.Tensor) -> torch.Tensor:
        if state.ndim == 1:
            state = state.unsqueeze(0)
        state = state.clone()
        state[:, 0] = torch.clamp(state[:, 0], self.time_limits[0], self.time_limits[1])
        state[:, 1] = torch.clamp(state[:, 1], self.gain_limits[0], self.gain_limits[1])
        return state

    def build_initial_state(self, batch_size: int, device: Optional[torch.device] = None) -> torch.Tensor:
        state = torch.tensor(
            [[self.init_time, self.init_gain]] * batch_size,
            dtype=torch.float32,
            device=device,
        )
        return self.clamp_state(state)


class ActorCriticReplayBuffer:
    """Experience replay buffer for Actor-Critic exposure control.

    Each transition contains:
    - obs / next_obs: dict with at least keys left/right (CarlaStereoDataset style).
    - state / next_state: exposure state tensor [exposure_time, gain].
    - action: actor action tensor.
    - reward: scalar reward.
    - done: episode done flag.
    - entrop_penalty: optional entropy penalty term for reward regularization.
    """

    def __init__(
        self,
        capacity: int,
        state_cfg: Optional[ExposureStateConfig] = None,
        seed: int = 0,
    ):
        if capacity <= 0:
            raise ValueError('capacity must be > 0')
        self.capacity = int(capacity)
        self.state_cfg = state_cfg or ExposureStateConfig()
        self._rng = random.Random(seed)
        self._buffer: Deque[Dict[str, Any]] = deque(maxlen=self.capacity)

    @staticmethod
    def _to_tensor(value: Any, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.to(dtype=dtype) if dtype is not None else value
        if isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value)
            return tensor.to(dtype=dtype) if dtype is not None else tensor
        if isinstance(value, (int, float, bool)):
            return torch.tensor(value, dtype=dtype if dtype is not None else None)
        raise TypeError(f'Unsupported tensor value type: {type(value)}')

    def _normalize_obs(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        normalized: Dict[str, Any] = {}
        for key, value in obs.items():
            if key == 'name':
                normalized[key] = value
                continue
            if value is None:
                normalized[key] = None
                continue

            if key in ('left', 'right', 'disp'):
                normalized[key] = self._to_tensor(value, dtype=torch.float32)
            elif key in ('occ_mask', 'valid'):
                normalized[key] = self._to_tensor(value).to(torch.bool)
            elif key == 'index':
                normalized[key] = self._to_tensor(value).to(torch.long)
            else:
                if isinstance(value, (np.ndarray, torch.Tensor, int, float, bool)):
                    normalized[key] = self._to_tensor(value)
                else:
                    normalized[key] = value
        return normalized

    def _normalize_state(self, state: Any) -> torch.Tensor:
        state_tensor = self._to_tensor(state, dtype=torch.float32)
        if state_tensor.ndim == 0:
            raise ValueError('state must represent [exposure_time, gain], got scalar')
        if state_tensor.ndim > 1:
            state_tensor = state_tensor.reshape(-1)
        if state_tensor.numel() != 2:
            raise ValueError('state must have exactly 2 values: [exposure_time, gain]')
        state_tensor = state_tensor.unsqueeze(0)
        state_tensor = self.state_cfg.clamp_state(state_tensor)
        return state_tensor.squeeze(0)

    @staticmethod
    def _stack_items(items: List[Any]) -> Any:
        if len(items) == 0:
            return None
        if all(torch.is_tensor(x) for x in items):
            return torch.stack(items, dim=0)
        return items

    def _stack_obs_dict(self, obs_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        keys = set()
        for obs in obs_list:
            keys.update(obs.keys())

        out: Dict[str, Any] = {}
        for key in keys:
            values = [obs.get(key, None) for obs in obs_list]
            if all(v is None for v in values):
                out[key] = None
                continue
            non_none = [v for v in values if v is not None]
            if len(non_none) != len(values):
                out[key] = values
                continue
            out[key] = self._stack_items(values)
        return out

    def add(
        self,
        obs: Dict[str, Any],
        state: Any,
        action: Any,
        reward: Any,
        next_obs: Dict[str, Any],
        next_state: Any,
        done: Any,
        entrop_penalty: Optional[Any] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        transition = {
            'obs': self._normalize_obs(obs),
            'state': self._normalize_state(state),
            'action': self._to_tensor(action, dtype=torch.float32),
            'reward': self._to_tensor(reward, dtype=torch.float32).reshape(()),
            'next_obs': self._normalize_obs(next_obs),
            'next_state': self._normalize_state(next_state),
            'done': self._to_tensor(done).to(torch.bool).reshape(()),
            'entrop_penalty': None if entrop_penalty is None else self._to_tensor(entrop_penalty, dtype=torch.float32).reshape(()),
            'extra': extra if extra is not None else {},
        }
        self._buffer.append(transition)

    def sample(self, batch_size: int, device: Optional[torch.device] = None) -> Dict[str, Any]:
        if batch_size <= 0:
            raise ValueError('batch_size must be > 0')
        if len(self._buffer) < batch_size:
            raise ValueError(f'Not enough samples in replay buffer: need {batch_size}, got {len(self._buffer)}')

        transitions = self._rng.sample(list(self._buffer), k=batch_size)
        obs_batch = self._stack_obs_dict([t['obs'] for t in transitions])
        next_obs_batch = self._stack_obs_dict([t['next_obs'] for t in transitions])

        batch = {
            'obs': obs_batch,
            'state': torch.stack([t['state'] for t in transitions], dim=0),
            'action': torch.stack([t['action'] for t in transitions], dim=0),
            'reward': torch.stack([t['reward'] for t in transitions], dim=0),
            'next_obs': next_obs_batch,
            'next_state': torch.stack([t['next_state'] for t in transitions], dim=0),
            'done': torch.stack([t['done'] for t in transitions], dim=0),
            'entrop_penalty': None,
            'extra': [t['extra'] for t in transitions],
        }

        penalties = [t['entrop_penalty'] for t in transitions]
        if all(p is not None for p in penalties):
            batch['entrop_penalty'] = torch.stack(penalties, dim=0)

        if device is not None:
            batch['state'] = batch['state'].to(device)
            batch['action'] = batch['action'].to(device)
            batch['reward'] = batch['reward'].to(device)
            batch['next_state'] = batch['next_state'].to(device)
            batch['done'] = batch['done'].to(device)
            if batch['entrop_penalty'] is not None:
                batch['entrop_penalty'] = batch['entrop_penalty'].to(device)

            for container_key in ('obs', 'next_obs'):
                for key, value in batch[container_key].items():
                    if torch.is_tensor(value):
                        batch[container_key][key] = value.to(device)

        return batch

    def clear(self) -> None:
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)


class CarlaStereoExperienceReplayBuffer(ActorCriticReplayBuffer):
    """Replay buffer bound to CarlaStereoDataset format.

    This class provides initial-state generation with fixed exposure time/gain,
    while enforcing that states are always within configured exposure limits.
    """

    def __init__(
        self,
        capacity: int,
        data_info: Any,
        data_cfg: Any,
        mode: str,
        state_cfg: Optional[ExposureStateConfig] = None,
        seed: int = 0,
    ):
        super().__init__(capacity=capacity, state_cfg=state_cfg, seed=seed)
        self.dataset = CarlaStereoDataset(data_info=data_info, data_cfg=data_cfg, mode=mode)

    def sample_initial_batch(self, batch_size: int) -> Dict[str, Any]:
        if len(self.dataset) == 0:
            raise ValueError('CarlaStereoDataset is empty, cannot sample initial batch.')
        if batch_size <= 0:
            raise ValueError('batch_size must be > 0')

        indices = [self._rng.randrange(len(self.dataset)) for _ in range(batch_size)]
        samples = [self.dataset[idx] for idx in indices]

        def _stack_sample_value(key: str) -> Any:
            values = [s[key] for s in samples]
            if key == 'name':
                return values
            if key in ('left', 'right', 'disp'):
                tensors = [self._to_tensor(v, dtype=torch.float32) for v in values]
                return torch.stack(tensors, dim=0)
            if key in ('occ_mask', 'valid'):
                tensors = [self._to_tensor(v).to(torch.bool) for v in values]
                return torch.stack(tensors, dim=0)
            if key == 'index':
                tensors = [self._to_tensor(v).to(torch.long) for v in values]
                return torch.stack(tensors, dim=0)
            tensors = [self._to_tensor(v) for v in values]
            return torch.stack(tensors, dim=0)

        obs = {
            'left': _stack_sample_value('left'),
            'right': _stack_sample_value('right'),
            'disp': _stack_sample_value('disp'),
            'occ_mask': _stack_sample_value('occ_mask'),
            'valid': _stack_sample_value('valid'),
            'index': _stack_sample_value('index'),
            'name': _stack_sample_value('name'),
        }
        initial_state = self.state_cfg.build_initial_state(batch_size=batch_size, device=obs['left'].device)

        return {
            'obs': obs,
            'state': initial_state,
        }


if __name__ == '__main__':
    cfg = ExposureStateConfig(time_limits=(5.0, 20.0), gain_limits=(1.0, 20.0), init_time=12.5, init_gain=1.0)
    buffer = ActorCriticReplayBuffer(capacity=8, state_cfg=cfg, seed=2026)

    for i in range(4):
        left = torch.rand(3, 64, 64)
        right = torch.rand(3, 64, 64)
        next_left = torch.rand(3, 64, 64)
        next_right = torch.rand(3, 64, 64)

        s = cfg.build_initial_state(1).squeeze(0)
        ns = torch.tensor([13.0, 1.2], dtype=torch.float32)
        action = torch.tensor([0.2, -0.1], dtype=torch.float32)
        reward = torch.tensor(1.0 - i * 0.1, dtype=torch.float32)
        done = torch.tensor(i == 3)

        buffer.add(
            obs={'left': left, 'right': right, 'index': i, 'name': f'sample_{i}'},
            state=s,
            action=action,
            reward=reward,
            next_obs={'left': next_left, 'right': next_right, 'index': i, 'name': f'sample_{i}_next'},
            next_state=ns,
            done=done,
            entrop_penalty=torch.tensor(-0.01),
        )

    batch = buffer.sample(batch_size=2)
    print('state shape:', tuple(batch['state'].shape))
    print('action shape:', tuple(batch['action'].shape))
    print('reward shape:', tuple(batch['reward'].shape))
    print('entrop_penalty shape:', None if batch['entrop_penalty'] is None else tuple(batch['entrop_penalty'].shape))
