#!/bin/bash

export REPO_DIR=/kuacc/users/hpc-mdogan/hpc_run/In-Context-Learning/

export IMAGE_DIR=${REPO_DIR}/dataset/imgs/all/

export MODEL=clip
export MODEL_NAME=CLIP
export HF_PATH=openai/clip-vit-base-patch32

export DEVICE=cuda
export SEED=42
export FOLDER_NAME=similarities

echo "Settings:"
echo MODEL=$MODEL
echo MODEL_NAME=$MODEL_NAME
echo HF_PATH=$HF_PATH
echo IMAGE_DIR=$IMAGE_DIR
echo DEVICE=$DEVICE
echo SEED=$SEED
echo FOLDER_NAME=$FOLDER_NAME

tasks=(
    "existence"   
    "plurals"
    "counting-hard"
    "counting-small-quant"
    "counting-adversarial"
    "relations"
    "action-replacement"
    "actant-swap"
    "coreference-standard"
    "coreference-hard"
    "foil-it"
)

for task in "${tasks[@]}"; do
    
      export TASK_NAME=$task
      export OUTPUT_FILE=${REPO_DIR}/example_info/${FOLDER_NAME}/${MODEL}/${TASK_NAME}.json
      export ANNO_FILE=${REPO_DIR}/dataset/annotations/$TASK_NAME.json

      echo TASK_NAME=$TASK_NAME
      echo OUTPUT_FILE=$OUTPUT_FILE
      echo ANNO_FILE=$ANNO_FILE

      python select_examples.py --model $MODEL --annotation_file $ANNO_FILE --image_dir $IMAGE_DIR --output_file $OUTPUT_FILE --device $DEVICE --seed $SEED --hf_path $HF_PATH   
done