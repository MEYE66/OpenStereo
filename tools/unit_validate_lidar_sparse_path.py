import argparse
from copy import deepcopy
import os
import sys

import torch
from easydict import EasyDict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from cfgs.data_basic import DATA_PATH_DICT
from stereo.datasets import build_dataloader
from stereo.modeling.models.gwcnet.gwcnet_lidar_sparse import LidarGwcNet
from stereo.utils import common_utils


def _load_cfg(cfg_file):
    cfg = EasyDict(common_utils.config_loader(cfg_file))
    for data_info in cfg.DATA_CONFIG.DATA_INFOS:
        dataset_name = data_info.DATASET
        if dataset_name == "KittiDataset":
            dataset_name = "KittiDataset15" if "kitti15" in data_info.DATA_SPLIT.EVALUATING else "KittiDataset12"
        data_info.DATA_PATH = DATA_PATH_DICT[dataset_name]
    return cfg


def _move_to_device(batch, device):
    moved = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            moved[k] = v.to(device)
        else:
            moved[k] = v
    return moved


def main():
    parser = argparse.ArgumentParser(description="Validate dataloader + get_loss path for lidar sparse supervision")
    parser.add_argument("--cfg_file", type=str, default="cfgs/gwcnet/gwcnet_lidar_sparse.yaml")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    cfg = _load_cfg(args.cfg_file)

    dataset, loader, _ = build_dataloader(
        data_cfg=cfg.DATA_CONFIG,
        batch_size=args.batch_size,
        is_dist=False,
        workers=args.workers,
        pin_memory=False,
        mode="training",
    )
    print(f"[OK] dataset length: {len(dataset)}")

    batch = next(iter(loader))
    print(f"[OK] batch keys: {list(batch.keys())}")

    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"[INFO] using device: {device}")

    model = LidarGwcNet(cfg.MODEL).to(device)
    model.train()

    model_input = _move_to_device(deepcopy(batch), device)
    model_preds = model(model_input)
    loss, loss_info = model.get_loss(model_preds, model_input)

    print(f"[OK] left shape: {tuple(model_input['left'].shape)}")
    print(f"[OK] right shape: {tuple(model_input['right'].shape)}")
    print(f"[OK] disp shape: {tuple(model_input['disp'].shape)}")
    print(f"[OK] valid shape: {tuple(model_input['valid'].shape)}")
    print(f"[OK] disp_pred stages: {len(model_preds['disp_preds'])}")
    print(f"[OK] loss: {float(loss.item()):.6f}")
    print(f"[OK] loss_info: {loss_info}")
    print("[PASS] dataloader + forward + get_loss path is valid.")


if __name__ == "__main__":
    main()
