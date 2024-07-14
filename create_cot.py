import argparse
from utils.dataset import Dataset_v1
from utils.eval import process_scores
from utils.util import set_seed
from cot_generation.models import model_registry


def main():
    parser = argparse.ArgumentParser(description="Test script for various models")
    parser.add_argument("--model", choices=list(model_registry.models.keys()), help="Model to test")
    parser.add_argument("--annotation_file", help="Path to the JSON annotation file")
    parser.add_argument("--image_dir", help="Path to the source image directory")
    parser.add_argument("--output_file", help="Path to the output JSON file")
    parser.add_argument("--device", choices = ["cpu", "cuda"], help="Device type", default="cuda")
    parser.add_argument("--prompt_type", help="Prompt type to be used to calculate results", default="ITM")
    parser.add_argument("--seed", type=int, help="Random seed", default=42)
    parser.add_argument("--hf_path", type=str, help="hf_path")

    args = parser.parse_args()
    set_seed(args.seed)
    data = Dataset_v1(
        json_path=args.annotation_file, 
        img_dir=args.image_dir, 
        prompt_type=args.prompt_type
    )

    if args.model in model_registry.models:
        model_load_func, model_create_desc_func, model_write_res_func = model_registry.models[args.model]
        model_load_func(args)
        model_create_desc_func(data)
        model_write_res_func()
        scores = process_scores(args.output_file)
        print(scores)
    else:
        print(f"Model '{args.model}' is not registered.")

if __name__ == "__main__":
    main()