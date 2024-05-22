#!/bin/bash

#SBATCH --job-name=create-similarities
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=ai
#SBATCH --qos=ai
#SBATCH --account=ai
#SBATCH --gres=gpu:nvidia_a40:8
#SBATCH --mem=100G
#SBATCH --time=1-0
#SBATCH --output=test-create-similarities-%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=dogankaas@gmail.com

echo "Activating anaconda/2022.10..."
module load anaconda/2022.10

echo "Activating python/3.9.5..."
module load python/3.9.5

echo "Activating cuda/11.8.0..."
module load cuda/11.8.0

echo "Activating glibc/2.27..."
module load glibc/2.27


# Set stack size to unlimited nvidia_a40 gpu:tesla_k80:8 gpu:tesla_v100:8 gpu:nvidia_a40:8
echo "Setting stack size to unlimited..."
ulimit -s unlimited
ulimit -l unlimited
ulimit -a

source activate icl

export MODEL=ClipModel
export HF_PATH=openai/clip-vit-base-patch32
export IMAGE_DIR=/kuacc/users/hpc-mdogan/hpc_run/In-Context-Learning/dataset/imgs/all/
export EXP_MODE=RANDOM
export DEVICE=cuda
export SEED=42
export FOLDER_NAME=similarities

echo "Settings:"
echo MODEL=$MODEL
echo HF_PATH=$HF_PATH
echo IMAGE_DIR=$IMAGE_DIR
echo EXP_MODE=$EXP_MODE
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
      export OUTPUT_FILE=/kuacc/users/hpc-mdogan/hpc_run/In-Context-Learning/example_info/${FOLDER_NAME}/${MODEL}/${TASK_NAME}.json
      export ANNO_FILE=/kuacc/users/hpc-mdogan/hpc_run/In-Context-Learning/dataset/annotations/$TASK_NAME.json

      echo TASK_NAME=$TASK_NAME
      echo OUTPUT_FILE=$OUTPUT_FILE
      echo ANNO_FILE=$ANNO_FILE

      python select_examples.py --model $MODEL --annotation_file $ANNO_FILE --image_dir $IMAGE_DIR --output_file $OUTPUT_FILE --device $DEVICE --seed $SEED --hf_path $HF_PATH
    
done
