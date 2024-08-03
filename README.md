# In-Context-Learning
## Project Overview
This project aims to assess the capabilities of Multimodal Large Language Models (MLLMs) using few-shot learning techniques. Specifically, we evaluate models pretrained on captioning and interleaved image and text datasets. Our goal is to understand how In-Context Learning (ICL) and Chain-of-Thought (CoT) prompting impact the performance of these models in tasks that require complex reasoning and contextual understanding.

## News and Updates
- `2024.07.17` 🎉🎉🎉 Our study is now available on [arXiv](https://arxiv.org/pdf/2407.12498).

## Environment Installation
To set up the environment for testing the models, follow these steps:

### 1. Clone the repository:
```
git clone https://github.com/mustafaadogan/In-Context-Learning.git
cd In-Context-Learning
```

### 2. Create and activate the Conda environment
- For OpenFlamingo variants:
  ```
  conda env create -f envs/openflamingo.yml
  conda activate openflamingo
  ```
- For the rest of the models:
  ```
  conda env create -f envs/common.yml
  conda activate common
  ```
## Similarity Generation
The [similarity_generation](similarity_generation) folder contains the model implementation to create image-text similarity data. We utilize these scores in selecting more similar demonstration examples in the few-shot settings. You can generate similarity scores with the following:
```
bash scripts/create_similarity.sh
```
You can add new model implementations for similarity generation under the [similarity_generation](similarity_generation) folder.

## Chain of Thought Reasoning Generation
The [cot_generation](cot_generation) folder contains the model implementation to create CoT reasoning descriptions. These descriptions help models perform reasoning tasks more effectively. You can generate descriptions with the following:
```
bash scripts/create_cot.sh
```
You can add new model implementations for description generation under the [cot_generation](cot_generation) folder.

## Testing Models
We provide three bash scripts to test the models, each tailored to different pretraining datasets:

### 1. Models trained on Captioning Datasets
Test the zero-shot performance of models trained on captioning datasets.
```
bash scripts/test_captioning_models.sh
```

### 2.  Models trained on Interleaved Image-Text Datasets
Test models trained on interleaved image and text datasets. This script handles more parameters compared to captioning models.
```
bash scripts/test_interleaved_models.sh
```

### 3.  OpenFlamingo Variants
Test OpenFlamingo variants.
```
bash scripts/test_openflamingo.sh
```

## Parameters for Testing
The [test.py](test.py) script is called by the above bash scripts to test the respective models. Below are the parameters used in test.py:

```
usage: test.py [-h] [--model list(model_registry.models.keys())] [--annotation_file ANNOTATION_FILE]
            [--similarity_data_path SIMILARITY_DATA_PATH]
            [--support_example_count SUPPORT_EXAMPLE_COUNT] [--top_k TOP_K] [--top_n TOP_N]
            [--image_dir IMAGE_DIR] [--output_file OUTPUT_FILE]
            [--sup_exp_mode {CLASS,RANDOM,SIMILAR}] [--device {cpu,cuda}]
            [--scoring_type {generated_text,perplexity}] [--prompt_type PROMPT_TYPE] [--seed SEED]
            [--lang_encoder_path LANG_ENCODER_PATH] [--tokenizer_path TOKENIZER_PATH]
            [--cross_attn_every_n_layers CROSS_ATTN_EVERY_N_LAYERS] [--hf_path HF_PATH]
            [--is_zero_cot_active | --no-is_zero_cot_active]
            [--is_few_cot_active | --no-is_few_cot_active]
            [--cot_desc_data_path COT_DESC_DATA_PATH] [--sc_exp_cnt SC_EXP_CNT]

Test script for various models

options:
  -h, --help            show this help message and exit
  --model list(model_registry.models.keys())
                        Model name registered in the model_registry dictionary
  --annotation_file ANNOTATION_FILE
                        Path to the JSON annotation file
  --similarity_data_path SIMILARITY_DATA_PATH
                        Path to the similarity JSON annotation file
  --support_example_count SUPPORT_EXAMPLE_COUNT
                        Support example count
  --top_k TOP_K         Top k visiaul similar examples
  --top_n TOP_N         Top n textual similar examples
  --image_dir IMAGE_DIR
                        Path to the source image directory
  --output_file OUTPUT_FILE
                        Path to the output JSON file
  --sup_exp_mode {CLASS,RANDOM,SIMILAR}
                        Support example mode
  --device {cpu,cuda}   Device type
  --scoring_type {generated_text,perplexity}
                        Scoring type to be used to calculate results
  --prompt_type PROMPT_TYPE
                        Prompt type to be used to calculate results
  --seed SEED           Random seed
  --lang_encoder_path LANG_ENCODER_PATH
                        lang_encoder_path
  --tokenizer_path TOKENIZER_PATH
                        tokenizer_path
  --cross_attn_every_n_layers CROSS_ATTN_EVERY_N_LAYERS
                        cross_attn_every_n_layers
  --hf_path HF_PATH     hf_path
  --is_zero_cot_active, --no-is_zero_cot_active
                        is Zero-Shot Chain of Thought active
  --is_few_cot_active, --no-is_few_cot_active
                        is Few-Shot Chain of Thought active
  --cot_desc_data_path COT_DESC_DATA_PATH
                        Path to JSON file containing CoT description of image-text pairs.
  --sc_exp_cnt SC_EXP_CNT
                        Self-Consistency experiment count
```
## Adding New Models
To add new models for similarity generation, CoT generation, or testing, place your implementations in the respective folders ([similarity_generation](similarity_generation), [cot_generation](cot_generation), [test](test)). Ensure that your codes are compatible with the existing framework and follow the provided examples for integration.

## Contributing
We welcome contributions to enhance the functionality and scope of this project. Please fork the repository and create a pull request with detailed information about the changes you propose.

## Contact
For questions or support, please contact [dogan_mustafa\@hacettepe.edu.tr](mailto:dogan_mustafa@hacettepe.edu.tr).
