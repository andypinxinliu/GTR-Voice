#!/usr/bin/env bash

set -a

#bash scripts/download_models.sh waveglow

PYTHONIOENCODING=utf-8

# for loop with the subfolder name
for gtr in ~/autodl-tmp/datasets/gtr_dataset_sliced_FREEZE/gtr_125/*; do
  # get second last of the string
  G=${gtr: -3:-2}
  T=${gtr: -2:-1}
  R=${gtr: -1}

  : ${BATCH_SIZE:=1}
  : ${FILELIST:="filelists/gen20.tsv"}
  : ${FASTPITCH:="/root/autodl-tmp/exps/output_gtr125_newmelftbigv/FastPitch_checkpoint_3000.pt"}
  OUTPUT_DIR="output_gtr125_final/syn125_20/${G}${T}${R}"
#  echo $OUTPUT_DIR
  : ${DATAPATH:="data/GTR_125"}
  : ${SPEAKER:=199}
#  : ${G:=$G}
#  : ${T:=$T}
#  : ${R:=$R}

  HIFIGAN=""
  WAVEGLOW=""
  bash scripts/inference_example_bigv.sh "$@"

  rm ${OUTPUT_DIR}/*json*

  python BigVGAN/inference_e2e.py \
  --checkpoint_file /root//autodl-tmp/bigv_24/g_05000000 \
  --input_mels_dir ${OUTPUT_DIR} \
  --output_dir ${OUTPUT_DIR}


done
exit

