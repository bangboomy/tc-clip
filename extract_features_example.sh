#!/bin/bash

# ViFi-CLIP特征提取示例脚本
# 使用方法: bash extract_features_example.sh

export CUDA_VISIBLE_DEVICES=0
export GPUS_PER_NODE=1

# 设置变量
CHECKPOINT_PATH="workspace/expr/fully_supervised_ntu60/vifi_clip_ntu60/fully_supervised_ntu60_vifi_clip_ntu60_vifi_clip/best.pth"  # 替换为你的模型检查点路径
OUTPUT_DIR="./workspace/vifi_features"               # 输出目录
DATASET="ntu60"                                 # 数据集名称 (k400, hmdb51, ucf101, ssv2等)
CONFIG_NAME="fully_supervised"                 # 配置名称 (fully_supervised, zero_shot, few_shot等)
trainer="vifi_clip"

# 数据参数
data=fully_supervised_ntu60
# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 运行特征提取
# 使用Hydra覆盖默认配置参数，确保使用ViFi-CLIP模型
torchrun --nproc_per_node=${GPUS_PER_NODE} extract_visual_features.py --config-name=${CONFIG_NAME}   resume=${CHECKPOINT_PATH}   output=${OUTPUT_DIR}   data=${data} num_workers=8 trainer=${trainer}

echo "ViFi-CLIP特征提取完成! 结果保存在 ${OUTPUT_DIR}"
