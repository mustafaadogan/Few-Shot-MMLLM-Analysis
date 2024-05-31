#!/bin/bash

#SBATCH --job-name=test-9-llava
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:2
#SBATCH --constraint=nvidia_a40
#SBATCH --mem=100G
#SBATCH --time=7-0
#SBATCH --output=./logs/exp9/test-llava-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dogankaas@gmail.com

echo "Activating anaconda/2022.10..."
module load anaconda/2022.10

echo "Activating python/3.10.6..."
module load python/3.10.6

echo "Activating cuda/11.8.0..."
module load cuda/11.8.0

echo "Activating glibc/2.27..."
module load glibc/2.27


# Set stack size to unlimited nvidia_a40 gpu:tesla_k80:8 gpu:tesla_v100:8 gpu:nvidia_a40:8
# Not Transformers 4.32.1 kullaniliyor
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a

export REPO_DIR=/kuacc/users/hpc-mdogan/hpc_run/In-Context-Learning
source activate icl2

export MODEL=llava
export HF_PATH=llava-hf/llava-v1.6-34b-hf
export SECOND_FOLDER_NAME=LLaVA_NeXT_34B

export IMAGE_DIR=${REPO_DIR}/dataset/imgs/all/
export EXP_MODE=RANDOM
export EXP_NAME=exp9
export DEVICE=cuda
export SCORING_TYPE=generated_text
export PROMPT_TYPE=ITM
export SEED=42
export SIMILARITY_MODEL=ClipModel
export COT_MODEL=llava
export TOP_K=20
export FOLDER_NAME=LLaVA
export COT_ACTIVE=no-is_cot_active
export SC_EXP_CNT=1


echo "Settings:"
echo $MODEL
echo $HF_PATH

echo IMAGE_DIR=$IMAGE_DIR
echo EXP_MODE=$EXP_MODE
echo EXP_NAME=$EXP_NAME
echo DEVICE=$DEVICE
echo SCORING_TYPE=$SCORING_TYPE
echo PROMPT_TYPE=$PROMPT_TYPE
echo SEED=$SEED
echo SIMILARITY_MODEL=$SIMILARITY_MODEL
echo COT_MODEL=$COT_MODEL
echo TOP_K=$TOP_K
echo FOLDER_NAME=$FOLDER_NAME
echo SECOND_FOLDER_NAME=$SECOND_FOLDER_NAME
echo THIRD_FOLDER_NAME=$THIRD_FOLDER_NAME
echo COT_ACTIVE=$COT_ACTIVE
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
        export OUTPUT_FILE=${REPO_DIR}/results/${EXP_NAME}/${FOLDER_NAME}/${SECOND_FOLDER_NAME}/${THIRD_FOLDER_NAME}/${EXP_COUNT}_shot/${TASK_NAME}_${MODEL}.json
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

        python test.py --model $MODEL --annotation_file $ANNO_FILE --support_example_count $EXP_COUNT --image_dir $IMAGE_DIR --output_file $OUTPUT_FILE --sup_exp_mode $EXP_MODE --device $DEVICE --scoring_type $SCORING_TYPE --prompt_type $PROMPT_TYPE --seed $SEED --hf_path $HF_PATH --similarity_data_path $SIMILARITY_ANNO_FILE --top_k $TOP_K --top_n $TOP_N --$COT_ACTIVE --sc_exp_cnt $SC_EXP_CNT --cot_desc_data_path $COT_ANNO_FILE
    done
done
