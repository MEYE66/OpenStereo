import argparse
import json
from typing import Dict, Tuple, Any

import torch
import thop


def _to_device(inputs: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
	return {k: v.to(device, non_blocking=True) for k, v in inputs.items()}


def build_stereo_inputs(
	batch_size: int = 1,
	channels: int = 3,
	height: int = 544,
	width: int = 960,
	device: str = 'cuda',
) -> Dict[str, torch.Tensor]:
	dev = torch.device(device)
	return {
		'left': torch.rand(batch_size, channels, height, width, device=dev),
		'right': torch.rand(batch_size, channels, height, width, device=dev),
	}


def count_parameters(model: torch.nn.Module, trainable_only: bool = False) -> int:
	if trainable_only:
		return sum(p.numel() for p in model.parameters() if p.requires_grad)
	return sum(p.numel() for p in model.parameters())


@torch.no_grad()
def compute_macs_and_flops(
	model: torch.nn.Module,
	inputs: Dict[str, torch.Tensor],
) -> Tuple[float, float, float]:
	# thop.profile returns MACs by convention for most ops.
	macs, _ = thop.profile(model, inputs=(inputs,), verbose=False)
	flops = 2.0 * float(macs)
	return float(macs), flops, flops / 1e9


@torch.no_grad()
def measure_gpu_latency_ms(
	model: torch.nn.Module,
	inputs: Dict[str, torch.Tensor],
	warmup: int = 20,
	iters: int = 100,
) -> float:
	if not torch.cuda.is_available():
		raise RuntimeError('CUDA is not available, cannot measure GPU latency.')

	starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

	for _ in range(max(warmup, 0)):
		_ = model(inputs)

	elapsed = []
	for _ in range(max(iters, 1)):
		starter.record()
		_ = model(inputs)
		ender.record()
		torch.cuda.synchronize()
		elapsed.append(starter.elapsed_time(ender))

	return float(sum(elapsed) / len(elapsed))


def profile_model_cost(
	model: torch.nn.Module,
	inputs: Dict[str, torch.Tensor],
	warmup: int = 20,
	iters: int = 100,
	trainable_only_params: bool = False,
) -> Dict[str, Any]:
	was_training = model.training
	model.eval()

	if torch.cuda.is_available():
		device = torch.device('cuda')
	else:
		device = next(model.parameters()).device

	model = model.to(device)
	dev_inputs = _to_device(inputs, device)

	param_count = count_parameters(model, trainable_only=trainable_only_params)
	macs, flops, flops_g = compute_macs_and_flops(model, dev_inputs)

	latency_ms = None
	if device.type == 'cuda':
		latency_ms = measure_gpu_latency_ms(model, dev_inputs, warmup=warmup, iters=iters)

	if was_training:
		model.train()

	return {
		'params': int(param_count),
		'params_m': float(param_count / 1e6),
		'macs': float(macs),
		'macs_g': float(macs / 1e9),
		'flops': float(flops),
		'flops_g': float(flops_g),
		'latency_ms': None if latency_ms is None else float(latency_ms),
	}


def profile_ae_and_stereo_cost(
	ae_model: torch.nn.Module,
	stereo_model: torch.nn.Module,
	input_shape: Tuple[int, int, int, int] = (1, 3, 544, 960),
	warmup: int = 20,
	iters: int = 100,
) -> Dict[str, Dict[str, Any]]:
	b, c, h, w = input_shape
	device = 'cuda' if torch.cuda.is_available() else 'cpu'
	inputs = build_stereo_inputs(b, c, h, w, device=device)

	ae_stats = profile_model_cost(ae_model, inputs, warmup=warmup, iters=iters)
	stereo_stats = profile_model_cost(stereo_model, inputs, warmup=warmup, iters=iters)

	return {
		'ae_model': ae_stats,
		'stereo_model': stereo_stats,
	}


def _print_stats(title: str, stats: Dict[str, Any]) -> None:
	print(f'[{title}]')
	print(f"  params: {stats['params']} ({stats['params_m']:.3f} M)")
	print(f"  MACs:   {stats['macs_g']:.3f} G")
	print(f"  FLOPs:  {stats['flops_g']:.3f} G")
	if stats['latency_ms'] is None:
		print('  latency: N/A (CUDA unavailable)')
	else:
		print(f"  latency: {stats['latency_ms']:.3f} ms")


def parse_args():
	parser = argparse.ArgumentParser(description='Compute model params/FLOPs/GPU latency')
	parser.add_argument('--batch', type=int, default=1)
	parser.add_argument('--channels', type=int, default=3)
	parser.add_argument('--height', type=int, default=544)
	parser.add_argument('--width', type=int, default=960)
	parser.add_argument('--warmup', type=int, default=20)
	parser.add_argument('--iters', type=int, default=100)
	parser.add_argument('--json', action='store_true', help='Print output as JSON')
	return parser.parse_args()


if __name__ == '__main__':
	args = parse_args()
	print('comp_util.py provides callable APIs:')
	print('  - profile_model_cost(model, inputs, warmup, iters)')
	print('  - profile_ae_and_stereo_cost(ae_model, stereo_model, input_shape, warmup, iters)')
	print('Import this module and pass your instantiated models.')

	# Keep a tiny runnable demo without project-specific model construction.
	class _DemoStereo(torch.nn.Module):
		def __init__(self):
			super().__init__()
			self.conv = torch.nn.Conv2d(3, 8, 3, 1, 1)
			self.head = torch.nn.Conv2d(8, 1, 3, 1, 1)

		def forward(self, inputs):
			left = torch.relu(self.conv(inputs['left']))
			right = torch.relu(self.conv(inputs['right']))
			disp = self.head(0.5 * (left + right))
			return {'disp_pred': disp}

	demo = _DemoStereo()
	demo_inputs = build_stereo_inputs(
		batch_size=args.batch,
		channels=args.channels,
		height=args.height,
		width=args.width,
		device='cuda' if torch.cuda.is_available() else 'cpu',
	)
	demo_stats = profile_model_cost(demo, demo_inputs, warmup=args.warmup, iters=args.iters)

	if args.json:
		print(json.dumps({'demo_model': demo_stats}, indent=2))
	else:
		_print_stats('demo_model', demo_stats)
