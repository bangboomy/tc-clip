# TC-CLIP eval example with 4 V100 gpus
export CUDA_VISIBLE_DEVICES=0
export GPUS_PER_NODE=1

protocol=zero_shot
dataset_name=k400
data=${protocol}_${dataset_name}
resume=workspace/expr/few_shot/tc_clip_reproduce/few_shot_ntu60_4shot_tc_clip_reproduce_tc_clip/best.pth
trainer=tc_clip

torchrun --nproc_per_node=${GPUS_PER_NODE} main.py \
data=${data} \
eval=test \
output=workspace/results/${data}/${data}_${trainer} \
resume=${resume} \
trainer=${trainer}
