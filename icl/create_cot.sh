#!/bin/bash

#SBATCH --job-name=create-cot
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:nvidia_a40:8
#SBATCH --mem=100G
#SBATCH --time=1-0
#SBATCH --output=create-cot-%j.out
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


# Set stack size to unlimited nvidia_a40 gpu:tesla_k80:8 gpu:tesla_v100:8 gpu:nvidia_a40:8 gpu:tesla_t4:8
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a

source activate icl2

export MODEL=llava
export HF_PATH=llava-hf/llava-v1.6-34b-hf

#export MODEL=LLaVA_1_6_Mistral_7B_HF
#export HF_PATH=llava-hf/llava-v1.6-mistral-7b-hf 

#export MODEL=llava_llama3
#export HF_PATH=xtuner/llava-llama-3-8b-v1_1-transformers

#export MODEL=internLMXComposer2
#export HF_PATH=internlm/internlm-xcomposer2-vl-7b-4bit

export IMAGE_DIR=/kuacc/users/hpc-mdogan/hpc_run/In-Context-Learning/dataset/imgs/all/
export EXP_MODE=RANDOM
export DEVICE=cuda
export SCORING_TYPE=generated_text
export SEED=42
export FOLDER_NAME=CoT
export SIMILARITY_MODEL=ClipModel
export TOP_K=20
export TOP_N=0
export PROMPT_TYPE=ITM
export EXP_COUNT=0

echo "Settings:"
echo MODEL=$MODEL
echo HF_PATH=$HF_PATH
echo IMAGE_DIR=$IMAGE_DIR
echo EXP_MODE=$EXP_MODE
echo DEVICE=$DEVICE
echo SCORING_TYPE=$SCORING_TYPE
echo SEED=$SEED
echo FOLDER_NAME=$FOLDER_NAME
echo SIMILARITY_MODEL=$SIMILARITY_MODEL
echo TOP_K=$TOP_K
echo TOP_N=$TOP_N
echo PROMPT_TYPE=$PROMPT_TYPE
echo EXP_COUNT=$EXP_COUNT

tasks=(
    "foil-it"
)

#"existence"   
#"plurals"
#"counting-hard"
#"counting-small-quant"
#"counting-adversarial"
#"relations"
#"action-replacement"
#"actant-swap"
#"coreference-standard"
#"coreference-hard"
#"foil-it"


for task in "${tasks[@]}"; do
    
      export TASK_NAME=$task
      export OUTPUT_FILE=/kuacc/users/hpc-mdogan/hpc_run/In-Context-Learning/example_info/${FOLDER_NAME}/${MODEL}/${TASK_NAME}.json
      export ANNO_FILE=/kuacc/users/hpc-mdogan/hpc_run/In-Context-Learning/dataset/annotations/$TASK_NAME.json
      export SIMILARITY_ANNO_FILE=/kuacc/users/hpc-mdogan/hpc_run/In-Context-Learning/example_info/similarities/${SIMILARITY_MODEL}/$TASK_NAME.json

      echo TASK_NAME=$TASK_NAME
      echo OUTPUT_FILE=$OUTPUT_FILE
      echo ANNO_FILE=$ANNO_FILE
      echo SIMILARITY_ANNO_FILE=$SIMILARITY_ANNO_FILE

      python create_cot.py --model $MODEL --annotation_file $ANNO_FILE --support_example_count $EXP_COUNT --image_dir $IMAGE_DIR --output_file $OUTPUT_FILE --sup_exp_mode $EXP_MODE --device $DEVICE --scoring_type $SCORING_TYPE --prompt_type $PROMPT_TYPE --seed $SEED --hf_path $HF_PATH --similarity_data_path $SIMILARITY_ANNO_FILE --top_k $TOP_K --top_n $TOP_N
    
done
