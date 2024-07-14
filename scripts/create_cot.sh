#!/bin/bash

export REPO_DIR=/kuacc/users/hpc-mdogan/hpc_run/In-Context-Learning/

export MODEL=llava
export MODEL_NAME=LLaVA_NeXT
export HF_PATH=llava-hf/llava-v1.6-34b-hf

#export MODEL=llava_llama3
#export MODEL_NAME=LLaVA_LLaMA3
#export HF_PATH=xtuner/llava-llama-3-8b-v1_1-transformers

#export MODEL=internLMXComposer2
#export MODEL_NAME=InterLMXComposer2
#export HF_PATH=internlm/internlm-xcomposer2-vl-7b-4bit

export IMAGE_DIR=${REPO_DIR}/dataset/imgs/all/
export DEVICE=cuda
export SEED=42
export FOLDER_NAME=CoT
export PROMPT_TYPE=Generate_CoT

echo "Settings:"
echo MODEL=$MODEL
echo MODEL_NAME=$MODEL_NAME
echo HF_PATH=$HF_PATH
echo IMAGE_DIR=$IMAGE_DIR
echo DEVICE=$DEVICE
echo SEED=$SEED
echo FOLDER_NAME=$FOLDER_NAME
echo PROMPT_TYPE=$PROMPT_TYPE

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
      export OUTPUT_FILE=${REPO_DIR}/example_info/${FOLDER_NAME}/${MODEL_NAME}/${TASK_NAME}.json
      export ANNO_FILE=${REPO_DIR}/dataset/annotations/$TASK_NAME.json

      echo TASK_NAME=$TASK_NAME
      echo OUTPUT_FILE=$OUTPUT_FILE
      echo ANNO_FILE=$ANNO_FILE

      python create_cot.py --model $MODEL --annotation_file $ANNO_FILE --image_dir $IMAGE_DIR --output_file $OUTPUT_FILE --device $DEVICE --prompt_type $PROMPT_TYPE --seed $SEED --hf_path $HF_PATH
done