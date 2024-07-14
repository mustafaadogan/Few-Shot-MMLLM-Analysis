import argparse
from utils.dataset import Dataset_v1
from utils.util import set_seed
from similarity_generation.models import model_registry

def main():
    parser = argparse.ArgumentParser(description="Test script for various models")
    parser.add_argument("--model", choices=list(model_registry.models.keys()), help="Model to be used in example selection")
    parser.add_argument("--annotation_file", help="Path to the JSON annotation file")
    parser.add_argument("--hf_path", help="Path to the HuggingFace")
    parser.add_argument("--image_dir", help="Path to the source image directory")
    parser.add_argument("--output_file", help="Path to the output JSON file")
    parser.add_argument("--device", choices = ["cpu", "cuda"], help="Device type", default="cuda")
    parser.add_argument("--seed", type=int, help="Random seed", default=50)

    args = parser.parse_args()
    set_seed(args.seed)
    data = Dataset_v1(
        json_path=args.annotation_file, 
        img_dir=args.image_dir
    )

    if args.model in model_registry.models:
        model_load_func, model_calculate_similarities_func, model_write_res_func = model_registry.models[args.model]
        model_load_func(args)
        model_calculate_similarities_func(data)
        model_write_res_func(args.output_file)

    else:
        print(f"Model '{args.model}' is not registered.")

if __name__ == "__main__":
    main()