# CUDA_VISIBLE_DEVICES=5 python tools/eval.py --cfg_file cfgs/gwcnet/gwcnet_carla_base_width_2.yaml --eval_data_cfg_file cfgs/carla_eval.yaml --pretrained_model ./output/CarlaStereoDataset/GwcNet/gwcnet_carla_base_width_2/default/ckpt/checkpoint_epoch_19.pth
# CUDA_VISIBLE_DEVICES=1 python tools/eval.py --cfg_file cfgs/gwcnet/gwcnet_carla_base_width_2.yaml --eval_data_cfg_file cfgs/carla_eval.yaml --pretrained_model ./output/CarlaStereoDataset/GwcNet/gwcnet_carla_ldr-ncc/default/ckpt/checkpoint_epoch_19.pth
# CUDA_VISIBLE_DEVICES=7 python tools/infer.py --cfg_file cfgs/gwcnet/gwcnet_carla_base_width_2.yaml --eval_data_cfg_file cfgs/carla_eval.yaml --pretrained_model output/CarlaStereoDataset/GwcNet/gwcnet_carla_ldr-ncc/default/ckpt/checkpoint_epoch_19.pth --save_root_dir ./vis_out/carla_stereo/semantic
# CUDA_VISIBLE_DEVICES=7 python tools/infer.py --cfg_file cfgs/gwcnet/gwcnet_carla_finetune+nae.yaml --eval_data_cfg_file cfgs/carla_eval.yaml --pretrained_model ./output/CarlaStereoDataset/NeuralAEGwcNet/gwcnet_carla_finetune+nae/default/ckpt/checkpoint_epoch_4.pth --save_root_dir ./vis_out/carla_stereo/contrast/
# CUDA_VISIBLE_DEVICES=1 python tools/infer.py --cfg_file cfgs/gwcnet/gwcnet_carla_base_width_2.yaml --eval_data_cfg_file cfgs/carla_eval.yaml --pretrained_model ./output/CarlaStereoDataset/GwcNet/gwcnet_carla_base_width_2/default/ckpt/checkpoint_epoch_19.pth --save_root_dir ./vis_out/carla_width/semantic/
# CUDA_VISIBLE_DEVICES=1 python tools/eval.py --cfg_file cfgs/gwcnet/gwcnet_carla_base_width_2.yaml --eval_data_cfg_file cfgs/carla_eval.yaml --pretrained_model ./output/CarlaStereoDataset/GwcNet/gwcnet_carla_ldr-ncc/default/ckpt/checkpoint_epoch_19.pth


python preprocessing/multi_process_ae_carla.py \
  --source-root /home/lgz/dataset/ADEC/carla_1280x384 \
  --output-root /home/lgz/dataset/ADEC/carla_1280x384/ae_methods \
  --splits val \
  --controller pid \
  --optimizer nelder_mead \
  --metric gradient \
  --n-workers 32
  
  


python preprocessing/multi_process_ae_carla.py \
  --source-root /home/lgz/dataset/ADEC/carla_1280x384 \
  --output-root /home/lgz/dataset/ADEC/carla_1280x384/ae_methods \
  --splits val \
  --controller pid \
  --optimizer nelder_mead \
  --metric mixed \
  --n-workers 32


python preprocessing/multi_process_ae_carla.py \
  --source-root /home/lgz/dataset/ADEC/carla_1280x384 \
  --output-root /home/lgz/dataset/ADEC/carla_1280x384/ae_methods \
  --splits val \
  --controller pid \
  --optimizer nelder_mead \
  --metric semantic \
  --n-workers 32
  
  
  

python preprocessing/multi_process_ae_carla.py \
  --source-root /home/lgz/dataset/ADEC/carla_1280x384 \
  --output-root /home/lgz/dataset/ADEC/carla_1280x384/ae_methods \
  --splits val \
  --controller pid \
  --optimizer pid \
  --metric entropy \
  --n-workers 32


CUDA_VISIBLE_DEVICES=7 python tools/eval.py \
--cfg_file cfgs/raftstereo/raftstereo_carla_600x800_rgb.yaml \
--eval_data_cfg_file cfgs/carla_eval.yaml \
--pretrained_model ./output/CarlaStereoDataset/RAFTStereo/raftstereo_carla_600x800_rgb/default/ckpt/checkpoint_epoch_19.pth
