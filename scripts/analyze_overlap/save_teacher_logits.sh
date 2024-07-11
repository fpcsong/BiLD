#! /bin/bash

NUM_GPUS=8

MODEL_NAME=${1:-'qwen'}
TEACHER_MODEL=${2:-'<PATH_TO_YOUR_TEACHER_MODEL_LAST_CHECKPOINT>'}
TEACHER_SAVE_DIR=${3:-'./results/teacher_logits/'}

EVALDATA=./benchmarking/en_datasets

# save teacher data
mkdir -p $TEACHER_SAVE_DIR

deepspeed --num_gpus $NUM_GPUS ./acc_overlap_eval.py \
    --save_teacher_data \
    --save_logits_len 1024 \
    --teacher_save_path $TEACHER_SAVE_DIR \
    -v_data $EVALDATA \
    --val_files_pattern '/*/eval/input_*.jsonl' \
    --teacher_model_path $TEACHER_MODEL \
    --model_name $MODEL_NAME \
    -max_len 1024 \
    --template_name none \
    --deepspeed true 
