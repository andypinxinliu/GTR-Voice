#!/usr/bin/env bash

set -a

#bash scripts/download_models.sh waveglow

PYTHONIOENCODING=utf-8

: ${BATCH_SIZE:=1}
: ${FILELIST:="filelists/gen20.tsv"}
: ${FASTPITCH:="/root/autodl-tmp/exps/output_gtr125_newmelftbigv/FastPitch_checkpoint_3000.pt"}
: ${OUTPUT_DIR:="/root/autodl-tmp/exps/output_gtr125_newmelftbigv//output_3000_443"}
: ${DATAPATH:="data/GTR_125"}
: ${SPEAKER:=199}
: ${G:=4}
: ${T:=4}
: ${R:=3}

# Disable HiFi-GAN and enable WaveGlow
#: ${DENOISING:=0.0}
#: ${HIFIGAN_CONFIG:="pretrained_models/hifigan/LibriTTS/config.json"}
HIFIGAN=""
#HIFIGAN="./pretrained_models/hifigan/hifigan_gen_checkpoint_10000_ft.pt"
WAVEGLOW=""
#WAVEGLOW="pretrained_models/waveglow/nvidia_waveglow256pyt_fp16.pt"

bash scripts/inference_example_bigv.sh "$@"

rm ${OUTPUT_DIR}/*json*

python BigVGAN/inference_e2e.py \
--checkpoint_file /root//autodl-tmp/bigv_24/g_05000000 \
--input_mels_dir ${OUTPUT_DIR} \
--output_dir ${OUTPUT_DIR}

