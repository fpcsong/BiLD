#! /bin/bash

# only eval a certain model

MODEL_NAME=$1
LOSS=$2

BENCHMARK_PATH=./benchmarking/datasets

MODEL_DIR=$3

TRAINDATA=$BENCHMARK_PATH
EVALDATA=$BENCHMARK_PATH

for BASEMODEL in $(ls -d $MODEL_DIR\/checkpoint-* | awk -F'-' '{ print $NF " " $0 }' | sort -n -k1,1 | cut -d' ' -f2-)
do
    if [ -f $BASEMODEL/config.json ]; then    
        python benchmark.py \
        --test \
        -v_data $EVALDATA \
        --train_files_pattern  '/*/train/input_*.jsonl' \
        --val_files_pattern '/*/eval/input_*.jsonl' \
        --model_path $BASEMODEL \
        --model_name $MODEL_NAME \
        --gen_config default \
        --bf16 \
        -output_dir $MODEL_DIR \
        -m_bsz 2 \
        -e_bsz 2 \
        -max_len 1024 \
        --template_name none \
        -output \
        2>&1 | tee $MODEL_DIR/eval.log
    fi
done
