#!/bin/bash

source ~/anaconda3/etc/profile.d/conda.sh
conda activate openflamingo

export TASK_NAME=relations
export MODEL=idefics

export OUTPUT_FILE=/media/mustafa/MyBook/In-Context-Learning/results/$TASK_NAME.json
export ANNO_FILE=/media/mustafa/MyBook/In-Context-Learning/dataset/annotations/$TASK_NAME.json

export IMAGE_DIR=/media/mustafa/MyBook/In-Context-Learning/dataset/imgs/all/
export EXP_COUNT=2
export EXP_MODE=RANDOM
export DEVICE=cpu
export SCORING_TYPE=generated_text

python test.py --model $MODEL --annotation_file $ANNO_FILE --support_example_count $EXP_COUNT --image_dir $IMAGE_DIR --output_file $OUTPUT_FILE --sup_exp_mode $EXP_MODE --device $DEVICE --scoring_type $SCORING_TYPE

conda deactivate
conda activate vl-bench
python ./bin/eval.py /media/mustafa/MyBook/In-Context-Learning/results/${TASK_NAME}_${MODEL}.json --mode generated_text
