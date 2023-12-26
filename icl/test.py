import argparse
from models.models import model_registry
from utils.dataset import Dataset_v1

def main():
    parser = argparse.ArgumentParser(description="Test script for various models")
    parser.add_argument("--model", choices=list(model_registry.models.keys()), help="Model to test")
    parser.add_argument("--annotation_file", help="Path to the JSON annotation file")
    parser.add_argument("--support_example_count", type=int, help="Support example count", default=2)
    parser.add_argument("--image_dir", help="Path to the source image directory")
    parser.add_argument("--output_file", help="Path to the output JSON file")
    parser.add_argument("--sup_exp_mode", choices = ["CLASS", "RANDOM"], help="Support example mode", default="CLASS")

    args = parser.parse_args()
    data = Dataset_v1(args.annotation_file, args.support_example_count, args.image_dir, args.sup_exp_mode)

    if args.model in model_registry.models:
        model_load_func, model_test_func, model_write_res_func = model_registry.models[args.model]
        model_load_func()
        model_test_func(data)
        model_write_res_func(args.output_file)
    else:
        print(f"Model '{args.model}' is not registered.")

if __name__ == "__main__":
    main()
