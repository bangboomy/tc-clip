# TC-CLIP eval example with 4 V100 gpus
export CUDA_VISIBLE_DEVICES=0,1,2,3
export GPUS_PER_NODE=4
export WANDB_API_KEY="8f9eca074355f2fa349335e6e83b70ddea4ac1d9"


protocol=few_shot
dataset_name=hmdb51 # choose one of {hmdb51, ucf101, ssv2}
data=${protocol}_${dataset_name}
shot=2  # choose one of {2, 4, 8, 16}
resume=/PATH/TO/TRAINED/MODELS/${protocol}_${dataset_name}_${shot}shot_tc_clip.pth
trainer=tc_clip

torchrun --nproc_per_node=${GPUS_PER_NODE} main.py -cn ${protocol} \
data=${data} \
shot=${shot}
eval=val \
output=workspace/results/${data}/${data}_${trainer} \
trainer=${trainer} \
resume=${resume} \
use_andb=false