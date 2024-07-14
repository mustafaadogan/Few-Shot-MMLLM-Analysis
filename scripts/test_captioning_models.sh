#!/bin/bash

source activate common

export REPO_DIR=/kuacc/users/hpc-mdogan/hpc_run/In-Context-Learning

export MODEL=internLMXComposer2
export HF_PATH=internlm/internlm-xcomposer2-vl-7b-4bit
export FOLDER_NAME=InternLM-XComposer2
export SECOND_FOLDER_NAME=InternLM-XComposer2-VL

#export MODEL=internVLChat
#export HF_PATH=OpenGVLab/InternVL-Chat-V1-5-Int8
#export FOLDER_NAME=InternVLChat
#export SECOND_FOLDER_NAME=InternVLChat_1_5

#export MODEL=paligemma
#export HF_PATH=google/paligemma-3b-mix-224
#export FOLDER_NAME=PaliGemma
#export SECOND_FOLDER_NAME=PaliGemma_3B

#export MODEL=llava
#export HF_PATH=llava-hf/llava-v1.6-34b-hf
#export FOLDER_NAME=LLaVA
#export SECOND_FOLDER_NAME=LLaVA_NeXT_34B


export IMAGE_DIR=${REPO_DIR}/dataset/imgs/all/
export EXP_MODE=RANDOM
export EXP_NAME=exp1
export DEVICE=cuda
export SCORING_TYPE=generated_text
export PROMPT_TYPE=ITM
export SEED=42
export ZERO_COT_ACTIVE=is_zero_cot_active
export FEW_COT_ACTIVE=no-is_few_cot_active


echo "Settings:"
echo MODEL=$MODEL
echo HF_PATH=$HF_PATH

echo IMAGE_DIR=$IMAGE_DIR
echo EXP_MODE=$EXP_MODE
echo EXP_NAME=$EXP_NAME
echo DEVICE=$DEVICE
echo SCORING_TYPE=$SCORING_TYPE
echo PROMPT_TYPE=$PROMPT_TYPE
echo SEED=$SEED
echo FOLDER_NAME=$FOLDER_NAME
echo SECOND_FOLDER_NAME=$SECOND_FOLDER_NAME
echo ZERO_COT_ACTIVE=$ZERO_COT_ACTIVE


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
      export OUTPUT_FILE=${REPO_DIR}/additional_results/${EXP_NAME}/${FOLDER_NAME}/${SECOND_FOLDER_NAME}/${TASK_NAME}_${MODEL}.json
      export ANNO_FILE=${REPO_DIR}/dataset/annotations/$TASK_NAME.json

      echo TASK_NAME=$TASK_NAME
      echo EXP_COUNT=$EXP_COUNT
      echo OUTPUT_FILE=$OUTPUT_FILE
      echo ANNO_FILE=$ANNO_FILE

      python test.py --model $MODEL --annotation_file $ANNO_FILE --image_dir $IMAGE_DIR --output_file $OUTPUT_FILE --sup_exp_mode $EXP_MODE --device $DEVICE --scoring_type $SCORING_TYPE --prompt_type $PROMPT_TYPE --seed $SEED --hf_path $HF_PATH --$ZERO_COT_ACTIVE --$FEW_COT_ACTIVE
done