#!/bin/bash

# TC-CLIP特征提取示例脚本
# 使用方法: bash extract_features_example.sh

# 设置变量
CHECKPOINT_PATH="path/to/your/checkpoint.pth"  # 替换为你的模型检查点路径
OUTPUT_DIR="./extracted_features"               # 输出目录
DATASET="k400"                                 # 数据集名称 (k400, hmdb51, ucf101, ssv2等)
CONFIG_NAME="fully_supervised"                 # 配置名称 (fully_supervised, zero_shot, few_shot等)

# 创建输出目录
mkdir -p ${OUTPUT_DIR}

# 运行特征提取
# 使用Hydra覆盖默认配置参数
python extract_visual_features.py   --config-name=${CONFIG_NAME}   checkpoint_path=${CHECKPOINT_PATH}   output_dir=${OUTPUT_DIR}   data.val.dataset_name=${DATASET}   data.val.root=path/to/your/dataset   data.val.ann_file=path/to/your/annotations   data.val.label_file=path/to/your/labels   distributed=false   num_frames=16   input_size=224   test_batch_size=8   num_workers=4

echo "特征提取完成! 结果保存在 ${OUTPUT_DIR}"
