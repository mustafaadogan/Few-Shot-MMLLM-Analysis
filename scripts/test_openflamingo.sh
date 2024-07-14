#!/bin/bash

source activate openflamingo

export REPO_DIR=/kuacc/users/hpc-mdogan/hpc_run/In-Context-Learning

export MODEL=openflamingo
export FOLDER_NAME=OpenFamingo

#3B
export LANG_ENCODER_PATH=anas-awadalla/mpt-1b-redpajama-200b
export TOKENIZER_PATH=anas-awadalla/mpt-1b-redpajama-200b
export CROSS_ATTN_LAYER_NUMBER=1
export HF_PATH=openflamingo/OpenFlamingo-3B-vitl-mpt1b
export THIRD_FOLDER_NAME=OpenFlamingo_3B

#3B-Instruct
#export LANG_ENCODER_PATH=anas-awadalla/mpt-1b-redpajama-200b-dolly
#export TOKENIZER_PATH=anas-awadalla/mpt-1b-redpajama-200b-dolly
#export CROSS_ATTN_LAYER_NUMBER=1
#export HF_PATH=openflamingo/OpenFlamingo-3B-vitl-mpt1b-langinstruct
#export THIRD_FOLDER_NAME=OpenFlamingo_3B_Instruct

#4B
#export LANG_ENCODER_PATH=togethercomputer/RedPajama-INCITE-Base-3B-v1
#export TOKENIZER_PATH=togethercomputer/RedPajama-INCITE-Base-3B-v1
#export CROSS_ATTN_LAYER_NUMBER=2
#export HF_PATH=openflamingo/OpenFlamingo-4B-vitl-rpj3b
#export THIRD_FOLDER_NAME=OpenFlamingo_4B

#4B-Instruct
#export LANG_ENCODER_PATH=togethercomputer/RedPajama-INCITE-Instruct-3B-v1
#export TOKENIZER_PATH=togethercomputer/RedPajama-INCITE-Instruct-3B-v1
#export CROSS_ATTN_LAYER_NUMBER=2
#export HF_PATH=openflamingo/OpenFlamingo-4B-vitl-rpj3b-langinstruct
#export THIRD_FOLDER_NAME=OpenFlamingo_4B_Instruct

#9B
#export LANG_ENCODER_PATH=anas-awadalla/mpt-7b
#export TOKENIZER_PATH=anas-awadalla/mpt-7b
#export CROSS_ATTN_LAYER_NUMBER=4
#export HF_PATH=openflamingo/OpenFlamingo-9B-vitl-mpt7b
#export THIRD_FOLDER_NAME=OpenFlamingo_9B


export IMAGE_DIR=${REPO_DIR}/dataset/imgs/all/
export EXP_MODE=RANDOM
export EXP_NAME=exp1
export DEVICE=cuda
export SCORING_TYPE=generated_text
export PROMPT_TYPE=ITM
export SEED=42
export ZERO_COT_ACTIVE=is_zero_cot_active
export FEW_COT_ACTIVE=no-is_few_cot_active
export SIMILARITY_MODEL=ClipModel
export COT_MODEL=llava
export TOP_K=100
export SECOND_FOLDER_NAME=TOP_K_${TOP_K}
export SC_EXP_CNT=1


echo "Settings:"
echo MODEL=$MODEL
echo HF_PATH=$HF_PATH
echo LANG_ENCODER_PATH=$LANG_ENCODER_PATH
echo TOKENIZER_PATH=$TOKENIZER_PATH
echo CROSS_ATTN_LAYER_NUMBER=$CROSS_ATTN_LAYER_NUMBER

echo IMAGE_DIR=$IMAGE_DIR
echo EXP_MODE=$EXP_MODE
echo EXP_NAME=$EXP_NAME
echo DEVICE=$DEVICE
echo SCORING_TYPE=$SCORING_TYPE
echo PROMPT_TYPE=$PROMPT_TYPE
echo SEED=$SEED
echo ZERO_COT_ACTIVE=$ZERO_COT_ACTIVE
echo FEW_COT_ACTIVE=$FEW_COT_ACTIVE
echo TOP_K=$TOP_K
echo FOLDER_NAME=$FOLDER_NAME
echo SECOND_FOLDER_NAME=$SECOND_FOLDER_NAME
echo THIRD_FOLDER_NAME=$THIRD_FOLDER_NAME
echo SIMILARITY_MODEL=$SIMILARITY_MODEL
echo COT_MODEL=$COT_MODEL
echo SC_EXP_CNT=$SC_EXP_CNT

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

shot_counts=( 0 )

for task in "${tasks[@]}"; do
    for exp_count in "${shot_counts[@]}"; do
        export TASK_NAME=$task
        export EXP_COUNT=$exp_count
        export TOP_N=$exp_count
        export OUTPUT_FILE=${REPO_DIR}/additional_results/${EXP_NAME}/${FOLDER_NAME}/${SECOND_FOLDER_NAME}/${THIRD_FOLDER_NAME}/${EXP_COUNT}_shot/${TASK_NAME}_${MODEL}.json
        export ANNO_FILE=${REPO_DIR}/dataset/annotations/$TASK_NAME.json
        export SIMILARITY_ANNO_FILE=${REPO_DIR}/example_info/similarities/${SIMILARITY_MODEL}/$TASK_NAME.json
        export COT_ANNO_FILE=${REPO_DIR}/example_info/CoT/${COT_MODEL}/$TASK_NAME.json
        
        echo TASK_NAME=$TASK_NAME
        echo EXP_COUNT=$EXP_COUNT
        echo TOP_N=$TOP_N
        echo OUTPUT_FILE=$OUTPUT_FILE
        echo ANNO_FILE=$ANNO_FILE
        echo SIMILARITY_ANNO_FILE=$SIMILARITY_ANNO_FILE
        echo COT_ANNO_FILE=$COT_ANNO_FILE

        python test.py --model $MODEL --annotation_file $ANNO_FILE --support_example_count $EXP_COUNT --image_dir $IMAGE_DIR --output_file $OUTPUT_FILE --sup_exp_mode $EXP_MODE --device $DEVICE --scoring_type $SCORING_TYPE --prompt_type $PROMPT_TYPE --seed $SEED --lang_encoder_path $LANG_ENCODER_PATH --tokenizer_path $TOKENIZER_PATH --cross_attn_every_n_layers $CROSS_ATTN_LAYER_NUMBER --hf_path $HF_PATH --similarity_data_path $SIMILARITY_ANNO_FILE --top_k $TOP_K --top_n $TOP_N --$ZERO_COT_ACTIVE --$FEW_COT_ACTIVE --sc_exp_cnt $SC_EXP_CNT --cot_desc_data_path $COT_ANNO_FILE
    done
done