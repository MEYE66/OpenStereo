# EBSNetMEFNet 在 CarlaStereoDataset 上的执行计划

## 任务目标

目标是让 Codex + GPT-5.5 在当前 OpenStereo 工作区内完成以下链路：

1. 修复 EBSNetMEFNet 在当前 Python 3 / PyTorch 环境下的兼容性问题。
2. 加载 EBSNetMEFNet 权重，对 /home/lgz/dataset/ADEC/carla_600x800/val 中的每一帧 HDR 双目图像生成曝光参数。
3. 使用 OpenStereo 中的 ae_util 渲染链路，把曝光后的左右图写入 /home/lgz/dataset/ADEC/carla_600x800/ae_methods/exposure_agent。
4. 复用 OpenStereo 现有 RAFTStereo 和 metric_per_image 评测链路，输出 d1_all、epe、thres_1、thres_2、thres_3。

## 已确认事实

1. EBSNetMEFNet 权重已经存在：
   - /home/lgz/workspace/OpenStereo/EBSNetMEFNet/checkpoints/day/policy.pth
   - /home/lgz/workspace/OpenStereo/EBSNetMEFNet/checkpoints/day/fusion.pth
   - /home/lgz/workspace/OpenStereo/EBSNetMEFNet/checkpoints/night/policy.pth
   - /home/lgz/workspace/OpenStereo/EBSNetMEFNet/checkpoints/night/fusion.pth
2. EBSNetMEFNet 的 action 空间由 A.txt 定义，不直接输出 exp_time 和 analog_gain。
3. 当前任务已经明确采用 action -> exp/gain 映射表，而不是只保留 action 或只保留选帧索引。
4. disparity 评测链路已经明确采用 RAFTStereo。
5. 现有 OpenStereo 已经有 ae_methods 数据布局先例，例如 pid、gradient、mixed，对应 val.txt 格式已经可直接复用。

## 总体实现路线

整体应按下面四段执行，而不是把所有改动混在一起：

1. 先修 EBSNetMEFNet 的运行时兼容性，只做到模型和权重可以在当前环境中成功加载。
2. 再新增 Carla 专用离线推理脚本，把 Carla HDR 数据适配成 EBSNet policy 需要的 preview 和 histogram 输入，并产出 action。
3. 再把 action 通过显式映射表转成 exp/gain，用 ae_util.py 渲染左右图，生成完整的 exposure_agent 数据目录和 val.txt。
4. 最后复用 OpenStereo 现有工具链跑 RAFTStereo 评测，通过 metric_per_image.py 统计最终指标。

## 第一阶段：修复 EBSNetMEFNet 兼容性

### 目标

让下面文件能在当前环境中被正常 import、实例化和 load_state_dict：

- /home/lgz/workspace/OpenStereo/EBSNetMEFNet/test.py
- /home/lgz/workspace/OpenStereo/EBSNetMEFNet/models/decision.py
- /home/lgz/workspace/OpenStereo/EBSNetMEFNet/models/hdrnet.py
- /home/lgz/workspace/OpenStereo/EBSNetMEFNet/loss/SSIM.py

### 必改项

1. 把 Python 2 的 print 语法改成 Python 3。
2. 把 dict.has_key 改成 key in dict。
3. 把 torch.utils.model_zoo.load_url 改成 torch.hub.load_state_dict_from_url，或者在当前结构下直接规避预训练下载路径。
4. 去掉不必要的 Variable 用法。
5. 修复 torch.meshgrid 的现代接口，补上 indexing 参数。
6. 修复 SSIM 中 window.cuda(img.get_device()) 这一类旧式设备写法，改成基于 tensor.device 的写法。
7. 处理 test.py 里硬编码的 .cuda()，至少让脚本支持统一 device 控制，避免在离线批处理里出现设备不一致。

### 本阶段完成标准

必须满足以下 smoke check：

1. DecisionNet 可以实例化。
2. HDRNet 可以实例化。
3. day 的 policy.pth 可以成功加载。
4. day 的 fusion.pth 可以成功加载。
5. 不再因为 Python 2 语法或旧 API 直接崩溃。

## 第二阶段：新增 Carla 专用曝光推理入口

### 建议新增文件

建议新增：

- /home/lgz/workspace/OpenStereo/EBSNetMEFNet/run_carla_exposure_agent.py

不要强行复用原始 test.py 作为主入口。原始 test.py 假设的数据布局是：

1. 每个样本目录里有 normal 预览图。
2. 每个样本目录里有 step_xxx 的曝光序列图像。
3. 每个样本目录里有 gt.png。

而当前 Carla 数据布局是：

- /home/lgz/dataset/ADEC/carla_600x800/val/Experiment*/hdr_left/*.hdr
- /home/lgz/dataset/ADEC/carla_600x800/val/Experiment*/hdr_right/*.hdr
- /home/lgz/dataset/ADEC/carla_600x800/val/Experiment*/ground_truth_disparity_left/*.npy

所以应新建 Carla 适配脚本，不建议把 test.py 改成既兼容旧数据又兼容 Carla 数据的巨型脚本。

### 该脚本的职责

1. 遍历 val 目录下所有 Experiment。
2. 对每个 frame id 读取左 HDR、右 HDR 和左视差真值路径。
3. 从左 HDR 生成 preview 图像。
4. 复用 EBSNet 的 histogram 构造方式，生成 policy 输入。
5. 用 policy.pth 推理得到 action。
6. 根据 action -> exp/gain 映射得到曝光参数。
7. 调用 ae_util 渲染左右图。
8. 写出 png、disparity 副本、exposure_params.txt、val.txt。

## 第三阶段：构造 EBSNet policy 输入

### 不能直接复用的部分

原始 test.py 读取的是 normal 图像和 step 曝光序列。当前 Carla 数据没有逐帧 normal 图像，也没有 step 序列图像。

### 应如何适配

1. 用每一帧左 HDR 图生成一个 preview。
2. preview resize 到 224x224。
3. 按原始 test.py 的逻辑构造三层 histogram：
   - 1x1 全局 histogram
   - 2x2 子块 histogram
   - 4x4 子块 histogram
4. histogram 的 bin 数保持 32，不要改。

### preview 生成建议

优先从以下文件复用或对齐逻辑：

- /home/lgz/workspace/OpenStereo/stereo/datasets/carla_stereo_dataset.py

建议保留两个 preview_mode：

1. gtm
2. minmax

默认使用 gtm，因为 HDR 直接 minmax 之后送入 DecisionNet，分布可能与原始 EBSNet 训练时差异过大。

### 明确不采用的输入

不要把 Experiment 目录下的 thumbnail_image.png 直接当作每一帧 policy 输入。它是 experiment 级别文件，不是 frame 级别信号。

## 第四阶段：建立 action -> exp/gain 映射

### 关键结论

EBSNet 的输出是 action，而 action 的语义来自 A.txt 中的曝光组合索引，不是物理曝光时间和增益。因此必须单独引入显式映射层。

### 建议实现方式

新增两个配置文件：

- /home/lgz/workspace/OpenStereo/EBSNetMEFNet/checkpoints/day/action_to_exp_gain.json
- /home/lgz/workspace/OpenStereo/EBSNetMEFNet/checkpoints/night/action_to_exp_gain.json

建议结构采用 action id 为键，例如：

{
  "0": {"exp_time": 12.0, "analog_gain": 8.0},
  "1": {"exp_time": 13.0, "analog_gain": 8.0}
}

也可以把 A.txt 中的组合写入注释字段，但运行时以 action id 为主键更直接。

### 不要做的事

1. 不要在代码里硬写启发式推导。
2. 不要把 action 索引直接等同于曝光值。
3. 不要把这部分逻辑埋进模型 forward 内部。

### 当前推荐范围

可以先对齐现有 ae_methods 输出分布，令 exp_time 和 analog_gain 大致落在：

- exp_time: 5 到 20
- analog_gain: 1 到 20

但具体映射关系应保留为可替换配置。

## 第五阶段：用 ae_util 渲染曝光后的左右图

### 目标后端

使用：

- /home/lgz/workspace/OpenStereo/stereo/modeling/models/aegwcnet/ae_util.py

### 推荐做法

不要在业务脚本里散落 tensor 变换和设备逻辑。应先在 ae_util 邻近位置，或在新脚本内部封装一层离线渲染辅助函数，统一完成：

1. HWC numpy HDR -> NCHW torch tensor。
2. radiance_scale。
3. exp_time 和 analog_gain 广播到 batch 维。
4. 调用 ImageFormationModel。
5. 输出转回 HWC numpy。
6. 保存为 png。

### 需要重点检查的问题

1. ae_util.py 中 gaussian_var 和 poisson_scale 目前不是 Parameter 或 buffer，如果模型被移动到 GPU，必须确保这些张量也在同一 device。
2. 输入 HDR 是否先做 radiance_scale，要统一策略，不要左边脚本一套、右边数据集一套。
3. nbits、seed、归一化范围需要固定下来，避免每次跑出不同分布。

### 可接受的 fallback

如果 ae_util 的 torch 版离线批处理不稳定，可以保留一个对拍 fallback，使用：

- /home/lgz/workspace/OpenStereo/preprocessing/utils.py

但首选实现仍然应以 ae_util.py 为准，因为任务要求明确引用该文件的渲染链路。

## 第六阶段：输出 exposure_agent 数据集

### 输出根目录

- /home/lgz/dataset/ADEC/carla_600x800/ae_methods/exposure_agent

### 每个 Experiment 下必须包含

1. hdr_left/*.png
2. hdr_right/*.png
3. ground_truth_disparity_left/*.npy
4. exposure_params.txt

### exposure_params.txt 格式

对齐现有 pid 和 gradient 方法，首行建议：

# frame_id exp gain

后续每行：

frame_id exp gain

例如：

0 13.500000 10.000000

### 根目录 val.txt 格式

必须与现有 ae_methods 目录一致。每行三列，相对 /home/lgz/dataset/ADEC 的路径：

left_png_path right_png_path disparity_npy_path

例如：

carla_600x800/ae_methods/exposure_agent/Experiment1/hdr_left/0.png carla_600x800/ae_methods/exposure_agent/Experiment1/hdr_right/0.png carla_600x800/ae_methods/exposure_agent/Experiment1/ground_truth_disparity_left/disparity_map_0.npy

### 参考但不要盲拷的文件

- /home/lgz/workspace/OpenStereo/optimizer/multi_process_ae.py
- /home/lgz/workspace/OpenStereo/path_generate.py

重点复用它们的输出目录和 manifest 约定，不是原样复制其优化器逻辑。

## 第七阶段：接入 RAFTStereo 做 disparity 评测

### 现成可复用文件

- /home/lgz/workspace/OpenStereo/tools/eval.py
- /home/lgz/workspace/OpenStereo/stereo/evaluation/metric_per_image.py
- /home/lgz/workspace/OpenStereo/stereo/modeling/carla_trainer.py
- /home/lgz/workspace/OpenStereo/cfgs/raftstereo/raftstereo_carla_600X800_rgb.yaml
- /home/lgz/workspace/OpenStereo/output/CarlaStereoDataset/RAFTStereo/raftstereo_carla_600X800_rgb/default/ckpt/checkpoint_epoch_5.pth

### 建议新增评测配置

新增：

- /home/lgz/workspace/OpenStereo/cfgs/carla_eval_exposure_agent.yaml

做法：从 /home/lgz/workspace/OpenStereo/cfgs/carla_eval.yaml 复制一份，只改 EVALUATING 为：

/home/lgz/dataset/ADEC/carla_600x800/ae_methods/exposure_agent/val.txt

### 关键注意点

1. 不要对已经渲染好的 png 再打开 ENABLE_HDR 或 ENABLE_RGB 的二次 tone-mapping 路径。
2. 评测阶段只需要让 CarlaStereoDataset 把 png 作为普通图像加载即可。
3. 最终指标必须来自 metric_per_image.py，而不是另写一套重复统计脚本。

## 推荐执行顺序

严格按下面顺序推进，不要跳步：

1. 修 EBSNetMEFNet 兼容性。
2. 做 checkpoint 加载 smoke test。
3. 新增 Carla 离线推理脚本框架。
4. 打通单帧 preview 和 histogram。
5. 接入 policy，得到单帧 action。
6. 接入 action -> exp/gain 映射。
7. 接入 ae_util，完成单帧左右图渲染。
8. 写单个 Experiment 的 exposure_params.txt 和 val.txt。
9. 扩展到全量 val。
10. 新增 exposure_agent 的 eval 配置。
11. 跑 RAFTStereo smoke eval。
12. 跑全量评测并保存 d1_all、epe、thres_1、thres_2、thres_3。

## 最小验证清单

### 验证 1：模型加载

至少验证：

1. DecisionNet 实例化成功。
2. HDRNet 实例化成功。
3. day/policy.pth 成功加载。
4. day/fusion.pth 成功加载。

### 验证 2：单帧推理

只处理 /home/lgz/dataset/ADEC/carla_600x800/val/Experiment1 的一帧，确认：

1. preview 图生成成功。
2. histogram 张量形状正确。
3. policy 输出 action 成功。
4. action 可以映射到 exp/gain。

### 验证 3：单帧渲染

确认以下文件成功写出：

1. 左图 png。
2. 右图 png。
3. disparity 副本。
4. exposure_params.txt。

### 验证 4：单 Experiment manifest

确认 exposure_params.txt 和 val.txt 的格式与以下现有样例一致：

- /home/lgz/dataset/ADEC/carla_600x800/ae_methods/pid/Experiment1/exposure_params.txt
- /home/lgz/dataset/ADEC/carla_600x800/ae_methods/pid/val.txt

### 验证 5：RAFTStereo smoke eval

用缩小版 split 或只保留少量样本的 val.txt 跑一次 tools/eval.py，确认日志能输出：

1. d1_all
2. epe
3. thres_1
4. thres_2
5. thres_3

## 非目标

以下内容不在本次执行范围内：

1. 重新训练 EBSNet 或 MEFNet。
2. 复现论文中的 PSNR 和 SSIM 主结果。
3. 从论文反推隐式的物理相机曝光模型。
4. 在首版中做自动昼夜分类并自动切换 day/night checkpoint。

## 关键风险与处理建议

### 风险 1：EBSNet 输入分布和 Carla preview 分布不一致

处理建议：

1. 先提供 gtm 和 minmax 两种 preview_mode。
2. 默认用 gtm。
3. 日志中保留 action 分布统计，方便后续看是否塌缩到少数动作。

### 风险 2：action -> exp/gain 映射缺少先验

处理建议：

1. 映射表文件必须独立保存。
2. 允许快速替换多个版本。
3. 首版不要把映射规则硬编码到 Python 逻辑里。

### 风险 3：ae_util 离线渲染的设备和数值范围不稳定

处理建议：

1. 固定 nbits、seed 和 radiance_scale 策略。
2. 为离线渲染单独写辅助函数，不要散落在主流程中。
3. 必要时用 preprocessing/utils.py 做结果对拍。

### 风险 4：评测阶段又对 png 做了不必要的 HDR 处理

处理建议：

1. 新增独立 eval 配置。
2. 检查 ENABLE_HDR、ENABLE_RGB、MINMAX_NORM 等字段，避免重复处理。

## 推荐交付物

执行完成后，至少应交付以下内容：

1. 修复后的 EBSNetMEFNet 兼容性代码。
2. Carla 专用推理脚本。
3. day/night 的 action_to_exp_gain.json。
4. exposure_agent 输出目录。
5. exposure_agent/val.txt。
6. carla_eval_exposure_agent.yaml。
7. 一份最终评测结果汇总，包含 d1_all、epe、thres_1、thres_2、thres_3。

## 建议首先落地的三个最小任务

如果要让 Codex + GPT-5.5 先快速推进，优先做这三个最小任务：

1. 修完 EBSNetMEFNet 的兼容性并验证权重加载。
2. 写 run_carla_exposure_agent.py 的单帧版本，完成 action 和 exp/gain 输出。
3. 接入 ae_util 完成单帧渲染，并在 exposure_agent/Experiment1 下写出一组样例文件。

等这三步通过后，再扩展到全量生成和 RAFTStereo 评测。