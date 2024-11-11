#!/usr/bin/env bash

set -a

PYTHONIOENCODING=utf-8

# Mandarin & English bilingual
ARGS+=" --symbol-set english_mandarin_basic"

# Initialize weights with a pre-trained English model
#bash scripts/download_models.sh fastpitch
ARGS+=" --init-from-checkpoint /root/autodl-tmp/exps/output_aishell3_bigv/FastPitch_checkpoint_80.pt"
#ARGS+=" --sampling_rate=24000"
#ARGS+=" --hop_length=300"
#ARGS+=" --win_length=1200"
#ARGS+=" --filter_length=2048"
ARGS+=" --pitch-mean 240.74597808628425"
ARGS+=" --pitch-std 106.86045804979788"
ARGS+=" --load-mel-from-disk"


AMP=false  # FP32 training for better stability

: ${DATASET_PATH:=data/GTR_125}
: ${TRAIN_FILELIST:=filelists/gtr125_audio_pitch_text_train2.txt}
: ${VAL_FILELIST:=filelists/gtr125_audio_pitch_text_val.txt}
: ${OUTPUT_DIR:=/root//autodl-tmp/exps//output_gtr125_newmelftbigv}
: ${NSPEAKERS:=200}
: ${LEARNING_RATE:=0.01}

NUM_GPUS=1 GRAD_ACCUMULATION=2 BATCH_SIZE=16 bash scripts/train.sh $ARGS "$@"
