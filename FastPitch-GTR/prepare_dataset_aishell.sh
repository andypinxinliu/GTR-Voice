set -e

#URL="https://catalog.ngc.nvidia.com/orgs/nvidia/resources/sf_bilingual_speech_zh_en"
#
#if [[ $1 == "" ]]; then
#    echo -e "\n**************************************************************************************"
#    echo -e "\nThe dataset needs to be downloaded manually from NGC by a signed in user:"
#    echo -e "\n\t$URL\n"
#    echo -e "Save as files.zip and run the script:"
#    echo -e "\n\tbash $0 path/to/files.zip\n"
#    echo -e "**************************************************************************************\n"
#    exit 0
#fi
#
#mkdir -p data

#echo "Extracting the data..."
## The dataset downloaded from NGC might be double-zipped as:
##     SF_bilingual -> SF_bilingual.zip -> files.zip
#if [ $(basename $1) == "files.zip" ]; then
#    unzip $1 -d data/
#    unzip data/SF_bilingual.zip -d data/
#elif [ $(basename $1) == "SF_bilingual.zip" ]; then
#    unzip $1 -d data/
#else
#    echo "Unknown input file. Supply either files.zip or SF_bilingual.zip as the first argument:"
#    echo "\t$0 [files.zip|SF_bilingual.zip]"
#    exit 1
#fi
#echo "Extracting the data... OK"

# Make filelists
#echo "Generating filelists..."
#python scripts/mandarin_chinese/split_sf.py data/SF_bilingual/text_SF.txt filelists/
#echo "Generating filelists... OK"

# Extract pitch (optionally extract mels)
set -e

export PYTHONIOENCODING=utf-8

: ${ARGS="--extract-mels"}

echo "Extracting pitch..."
python prepare_dataset.py \
    --wav-text-filelists filelists/aishell3_audio_text.txt \
    --n-workers 16 \
    --batch-size 1 \
    --dataset-path data/AISHELL3 \
    --extract-pitch \
    --f0-method pyin \
    --symbol_set english_mandarin_basic \
    --sampling-rate 24000 \
    --hop-length 256 \
    --win-length 1024 \
    --filter-length 1024 \
    --n-speakers 200 \
    $ARGS

echo "Extracting pitch... OK"
echo "data prepared successfully."
