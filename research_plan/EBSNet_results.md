# EBSNetMEFNet Carla Exposure Agent Results

Date: 2026-05-22

## Generated Dataset

- Source: `/home/lgz/dataset/ADEC/carla_600x800/val`
- Output: `/home/lgz/dataset/ADEC/carla_600x800/ae_methods/exposure_agent`
- Manifest: `/home/lgz/dataset/ADEC/carla_600x800/ae_methods/exposure_agent/val.txt`
- Frames: 2780
- Policy checkpoint: `EBSNetMEFNet/checkpoints/day/policy.pth`
- Action mapping: `EBSNetMEFNet/checkpoints/day/action_to_exp_gain.json`

## RAFTStereo Full Evaluation

Command:

```bash
env MPLCONFIGDIR=/tmp/matplotlib /home/lgz/miniconda3/envs/openstereo/bin/python tools/eval.py \
  --cfg_file cfgs/raftstereo/raftstereo_carla_600X800_rgb.yaml \
  --eval_data_cfg_file cfgs/carla_eval_exposure_agent.yaml \
  --pretrained_model output/CarlaStereoDataset/RAFTStereo/raftstereo_carla_600X800_rgb/default/ckpt/checkpoint_epoch_5.pth \
  --workers 0 \
  --save_root_dir output/exposure_agent_eval
```

Log: `output/exposure_agent_eval/CarlaStereoDataset/RAFTStereo/eval/eval_20260522-160942.log`

Metrics:

| metric | value |
| --- | ---: |
| d1_all | 69.9382 |
| epe | 28.8363 |
| thres_1 | 84.6762 |
| thres_2 | 76.8870 |
| thres_3 | 70.1438 |

## Smoke Checks

- EBSNet `DecisionNet` instantiated and loaded `day/policy.pth`.
- EBSNet `HDRNet` instantiated and loaded `day/fusion.pth`.
- Single-frame Carla policy input produced histogram shape `(1, 96, 4, 4)`.
- RAFTStereo smoke eval on 4 frames completed with all five metrics.
