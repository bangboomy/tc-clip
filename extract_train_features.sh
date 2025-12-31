#!/bin/bash
# Script to extract features from training data using TC-CLIP

# Set parameters
CHECKPOINT="workspace/expr/fully_supervised_ntu60/tc_clip_ntu60/fully_supervised_ntu60_tc_clip_ntu60_tc_clip/best.pth"
OUTPUT_DIR="./workspace/extracted_train_features"
DATA_CONFIG="fully_supervised_ntu60"
NUM_FRAMES=16
INPUT_SIZE=224
TEST_BATCH_SIZE=8
NUM_WORKERS=8

# Run feature extraction
python extract_visual_features.py   resume=$CHECKPOINT   output=$OUTPUT_DIR   data=$DATA_CONFIG   num_frames=$NUM_FRAMES   input_size=$INPUT_SIZE   test_batch_size=$TEST_BATCH_SIZE   num_workers=$NUM_WORKERS   extract_train=true
