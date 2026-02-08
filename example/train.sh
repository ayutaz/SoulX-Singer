#!/bin/bash

script_dir=$(dirname "$(realpath "$0")")
root_dir=$(dirname "$script_dir")

cd $root_dir || exit
export PYTHONPATH=$root_dir:$PYTHONPATH

data_dir=data/dataset
config=soulxsinger/config/soulxsinger.yaml
resume_from=pretrained_models/SoulX-Singer/model.pt
save_dir=checkpoints/finetune
phoneset_path=soulxsinger/utils/phoneme/phone_set.json

accelerate launch -m cli.train \
    --data_dir $data_dir \
    --config $config \
    --resume_from $resume_from \
    --save_dir $save_dir \
    --phoneset_path $phoneset_path \
    --max_steps 10000 \
    --no_wandb
