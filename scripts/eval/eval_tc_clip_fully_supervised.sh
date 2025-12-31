# TC-CLIP eval example with 4 V100 gpus
export CUDA_VISIBLE_DEVICES=0
export GPUS_PER_NODE=1

protocol=fully_supervised
dataset_name=ntu60
data=${protocol}_${dataset_name}
resume=workspace/expr/fully_supervised_ntu60/vifi_clip_ntu60/fully_supervised_ntu60_vifi_clip_ntu60_vifi_clip/best.pth
trainer=vifi_clip

torchrun --nproc_per_node=${GPUS_PER_NODE} main.py -cn ${protocol} \
data=${data} \
eval=test \
output=workspace/results/${data}/${data}_${trainer} \
resume=${resume} \
trainer=${trainer}