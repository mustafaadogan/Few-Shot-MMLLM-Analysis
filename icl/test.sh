#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate icl

export MODEL=openflamingo

export IMAGE_DIR=/media/Mustafa/MyBook/In-Context-Learning/dataset/imgs/all/
export EXP_MODE=RANDOM
export DEVICE=cuda
export SCORING_TYPE=perplexity
export PROMPT_TYPE=GPT4VisionwoTiePp
export SEED=43
export FOLDER_NAME=43

#########################################

export TASK_NAME=relations

export EXP_COUNT=0
export OUTPUT_FILE=/media/Mustafa/MyBook/In-Context-Learning/results/main4/${EXP_COUNT}_shot/${TASK_NAME}_${MODEL}.json
export ANNO_FILE=/media/Mustafa/MyBook/In-Context-Learning/dataset/annotations/$TASK_NAME.json

python test.py --model $MODEL --annotation_file $ANNO_FILE --support_example_count $EXP_COUNT --image_dir $IMAGE_DIR --output_file $OUTPUT_FILE --sup_exp_mode $EXP_MODE --device $DEVICE --scoring_type $SCORING_TYPE --prompt_type $PROMPT_TYPE --seed $SEED