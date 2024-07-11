#! /bin/bash

NUM_GPUS=1

MODEL_NAME=${1:-'qwen'}
OVERLAP_TOPK=${2:-32}
STUDENT_MODEL=${3:-'./results/distillation/qwen_4b_to_0.5b/bild'}
TEACHER_SAVE_DIR=${4:-'./results/teacher_logits'}

OUTDIR=$STUDENT_MODEL
STUDENT_MODEL_PATH=$OUTDIR/checkpoint-<STUDENT_LAST_CHECKPOINT>


# save teacher data
rm -rf $OUTDIR/logits_eval.log
rm -rf $OUTDIR/logits_eval.txt

deepspeed --num_gpus $NUM_GPUS ./acc_overlap_eval.py \
    --teacher_save_path $TEACHER_SAVE_DIR \
    --overlap_topk $OVERLAP_TOPK \
    --train_files_pattern  '/*/train/input_*.jsonl' \
    --val_files_pattern '/*/eval/input_*.jsonl' \
    --student_model_path $STUDENT_MODEL_PATH \
    --model_name $MODEL_NAME \
    --gen_config default \
    --bf16 \
    -m_bsz 1 \
    -e_bsz 1 \
    -max_len 1024 \
    --template_name none \
    --deepspeed true \
    -output \
    -output_dir $OUTDIR \
    2>&1 | tee $OUTDIR/logits_eval.log
