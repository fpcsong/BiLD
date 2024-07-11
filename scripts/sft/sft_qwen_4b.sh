#! /bin/bash

set -x

export CXX=g++

pwd
df -h
git show -s

NUM_GPUS=4

BASEMODEL=$1 # <YOUR_QWEN_4B_PATH>

BENCHMARK_PATH=./benchmarking/datasets

TRAINDATA=$BENCHMARK_PATH
EVALDATA=$BENCHMARK_PATH

OUTDIR=./results/teacher/qwen_4b

mkdir -p $OUTDIR
rm -rf $OUTDIR/*

deepspeed --num_gpus $NUM_GPUS train.py -it \
    -t_data $TRAINDATA \
    -v_data $EVALDATA \
    --train_files_pattern  '/*/train/input_*.jsonl' \
    --val_files_pattern '/*/eval/input_*.jsonl' \
    --model_path $BASEMODEL \
    --model_name qwen \
    --gradient_checkpointing \
    --bf16 \
    -output_dir $OUTDIR \
    -m_bsz 2 \
    -e_bsz 2 \
    --warmup_steps 64 \
    -max_len 1024 \
    --epochs 3 \
    --template_name none \
    -lr 1e-5 \
    -bsz 64 \
    -output \
    --deepspeed true \
    2>&1 | tee $OUTDIR/train.log

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
