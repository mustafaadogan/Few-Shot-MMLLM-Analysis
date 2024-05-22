import argparse
from utils.dataset import Dataset_v1
from utils.eval import process_scores
from utils.util import set_seed
from cot_generation.models import model_registry


def main():
    parser = argparse.ArgumentParser(description="Test script for various models")
    parser.add_argument("--model", choices=list(model_registry.models.keys()), help="Model to test")
    parser.add_argument("--annotation_file", help="Path to the JSON annotation file")
    parser.add_argument("--similarity_data_path", help="Path to the similarity JSON annotation file")
    parser.add_argument("--support_example_count", type=int, help="Support example count", default=0)
    parser.add_argument("--top_k", type=int, help="Top k visiaul similar examples", default=20)
    parser.add_argument("--top_n", type=int, help="Top n textual similar examples", default=0)
    parser.add_argument("--image_dir", help="Path to the source image directory")
    parser.add_argument("--output_file", help="Path to the output JSON file")
    parser.add_argument("--sup_exp_mode", choices = ["CLASS", "RANDOM", "SIMILAR"], help="Support example mode", default="CLASS")
    parser.add_argument("--device", choices = ["cpu", "cuda"], help="Device type", default="cuda")
    parser.add_argument("--scoring_type", choices = ["generated_text", "perplexity"], help="Scoring type to be used to calculate results", default="generated_text")
    parser.add_argument("--prompt_type", help="Prompt type to be used to calculate results", default="ITM")
    parser.add_argument("--seed", type=int, help="Random seed", default=42)
    parser.add_argument("--hf_path", type=str, help="hf_path")

    args = parser.parse_args()
    set_seed(args.seed)
    data = Dataset_v1(
        json_path=args.annotation_file, 
        num_support=args.support_example_count, 
        similarity_data_path=args.similarity_data_path,
        top_k=args.top_k,
        top_n=args.top_n,
        img_dir=args.image_dir, 
        mode=args.sup_exp_mode, 
        prompt_type=args.prompt_type
    )

    if args.model in model_registry.models:
        model_load_func, model_test_func, model_write_res_func = model_registry.models[args.model]
        model_load_func(args)
        model_test_func(data)
        model_write_res_func(args.output_file)
        scores = process_scores(args.output_file, args.scoring_type)
        print(scores)
    else:
        print(f"Model '{args.model}' is not registered.")

if __name__ == "__main__":
    main()