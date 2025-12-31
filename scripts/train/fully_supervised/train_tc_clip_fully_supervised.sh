# fully-supervised k400 training with 4 V100 gpus
export CUDA_VISIBLE_DEVICES=0
export GPUS_PER_NODE=1

protocol=fully_supervised
dataset_name=ntu60
data=${protocol}_${dataset_name}

expr_name=vifi_clip_ntu60
trainer=vifi_clip
use_wandb=false

torchrun --nproc_per_node=${GPUS_PER_NODE} main.py -cn ${protocol} \
data=${data} \
output=workspace/expr/${data}/${expr_name}/${data}_${expr_name}_${trainer} \
trainer=${trainer} \
use_wandb=${use_wandb}