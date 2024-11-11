#!/usr/bin/env bash

set -a

#bash scripts/download_models.sh waveglow

PYTHONIOENCODING=utf-8

: ${BATCH_SIZE:=1}
: ${FILELIST:="filelists/gen20.tsv"}
: ${FASTPITCH:="./output_gtr125_newmelft/FastPitch_checkpoint_100.pt"}
: ${OUTPUT_DIR:="output_gtr125_newmelft//audiotest_aishell_100_421"}
: ${DATAPATH:="data/GTR_125"}
: ${SPEAKER:=199}
: ${G:=4}
: ${T:=2}
: ${R:=1}

# Disable HiFi-GAN and enable WaveGlow
: ${DENOISING:=0.0}
: ${HIFIGAN_CONFIG:="pretrained_models/hifigan/LibriTTS/config.json"}
HIFIGAN="pretrained_models/hifigan/LibriTTS/g_00935000"
#HIFIGAN="./pretrained_models/hifigan/hifigan_gen_checkpoint_10000_ft.pt"
WAVEGLOW=""
#WAVEGLOW="pretrained_models/waveglow/nvidia_waveglow256pyt_fp16.pt"

bash scripts/inference_example.sh "$@"
