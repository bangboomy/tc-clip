# few-shot training (using k400 pretrained model) with 4 V100 gpus
export CUDA_VISIBLE_DEVICES=0
export GPUS_PER_NODE=1
export WANDB_API_KEY="8f9eca074355f2fa349335e6e83b70ddea4ac1d9"

protocol=few_shot
dataset_name=ntu60 # choose one of {hmdb51, ucf101, ssv2}
data=${protocol}_${dataset_name}

expr_name=tc_clip_reproduce
trainer=tc_clip
use_wandb=false

resume=zero_shot_k400_llm_tc_clip.pth

# k-shot training
for shot in 4
do
  torchrun --nproc_per_node=${GPUS_PER_NODE} main.py -cn ${protocol} \
  data=${data} \
  shot=${shot} \
  output=workspace/expr/${protocol}/${expr_name}/${protocol}_${dataset_name}_${shot}shot_${expr_name}_${trainer} \
  trainer=${trainer} \
  use_wandb=${use_wandb} \
  resume=${resume}
done