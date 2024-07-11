#! /bin/bash

set -x

export CXX=g++

pwd

LOSS_LIST=(vanilla_kl top_kl rkl dkd nkd normkd bild)

NUM_GPUS=4

LOSS=$1 # Choose from the LOSS_LIST above
BILD_K=${2:-8}
TOPKL_K=${3:-1024}

TEMPERATURE=3.0
BSZ=64
LR=2e-5
LAGLR=2e-5
ALPHA=1e-4

BENCHMARK_PATH=./benchmarking/datasets

TRAINDATA=$BENCHMARK_PATH
EVALDATA=$BENCHMARK_PATH

# Qwen
BASEMODEL=<YOUR_QWEN_0.5B_MODEL_PATH>
TEACHER=./results/teacher/qwen_4b/checkpoint-<YOUR_LAST_CHECKPOINT_STEPS>
STUDENT=$BASEMODEL

OUTDIR=./results/distillation/qwen_4b_to_0.5b/$LOSS

mkdir -p $OUTDIR
rm -rf $OUTDIR/*

deepspeed --num_gpus $NUM_GPUS train.py \
    -it \
    -t_data $TRAINDATA \
    -v_data $EVALDATA \
    --train_files_pattern  '/*/train/input_*.jsonl' \
    --val_files_pattern '/*/eval/input_*.jsonl' \
    --model_name qwen \
    --model_path $BASEMODEL \
    --teacher_model_path $TEACHER \
    --student_model_path $STUDENT \
    --bf16 \
    -output \
    -output_dir $OUTDIR \
    -max_len 1024 \
    -m_bsz 2 -e_bsz 2 \
    --warmup_steps 128 \
    --loss_type $LOSS \
    --temperature $TEMPERATURE \
    --lr_scheduler cosine \
    --eval_steps 204800 \
    --epochs 8 \
    --template_name none \
    --gradient_checkpointing \
    -distil \
    -lr $LR \
    -lag_lr $LAGLR \
    -bsz $BSZ \
    --alpha $ALPHA \
    --deepspeed true \
    --bild_topk $BILD_K \
    --top_kl_k $TOPKL_K \
    2>&1 | tee $OUTDIR/distil.log

for BASEMODEL in $(ls -d $OUTDIR\/checkpoint-* | awk -F'-' '{ print $NF " " $0 }' | sort -n -k1,1 | cut -d' ' -f2-)
do
    if [ -f $BASEMODEL/config.json ]; then    
        python benchmark.py \
        --test \
        -v_data $EVALDATA \
        --train_files_pattern  '/*/train/input_*.jsonl' \
        --val_files_pattern '/*/eval/input_*.jsonl' \
        --model_path $BASEMODEL \
        --model_name qwen \
        --gen_config default \
        --bf16 \
        -output_dir $OUTDIR \
        -m_bsz 2 \
        -e_bsz 2 \
        -max_len 1024 \
        --template_name none \
        -output \
        2>&1 | tee $OUTDIR/eval.log
    fi
done
