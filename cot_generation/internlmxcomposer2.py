from tqdm import tqdm
from utils.util import write_results, check_answer
from .models import model_registry
import torch

class InternLMXComposer2:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.results= {}
        self.device = None
        self.generation_cfg = {}
        self.output_file = None
        
    def load_model(self, args) -> None:
        """
        Load InternLMXComposer2 model.

        Parameters:
        - args: The args to load the model.

        Returns:
        None
        """
        
        import auto_gptq
        from auto_gptq.modeling import BaseGPTQForCausalLM
        from transformers import AutoTokenizer

        auto_gptq.modeling._base.SUPPORTED_MODELS = ["internlm"]
        torch.set_grad_enabled(False)

        class InternLMXComposer2QForCausalLM(BaseGPTQForCausalLM):
            layers_block_name = "model.layers"
            outside_layer_modules = [
                'vit', 'vision_proj', 'model.tok_embeddings', 'model.norm', 'output', 
            ]
            inside_layer_modules = [
                ["attention.wqkv.linear"],
                ["attention.wo.linear"],
                ["feed_forward.w1.linear", "feed_forward.w3.linear"],
                ["feed_forward.w2.linear"],
            ]

        print('Loading InternLMXComposer2!!!')
        self.device = args.device
        self.output_file = args.output_file
        
        self.model = InternLMXComposer2QForCausalLM.from_quantized(args.hf_path, trust_remote_code=True, device="cuda:0").eval()
        self.tokenizer = AutoTokenizer.from_pretrained(args.hf_path, trust_remote_code=True)
        
        for n, p in self.model.named_parameters():
            p.requires_grad = False
            
        self.generation_cfg = {
            #'num_beams': 3,
            #'max_new_tokens': 512,
            #"temperature": 0.2,
            #"top_p": 0.7,
            #'top_k': 0,
            #'length_penalty': -2.0,
            #'num_return_sequences': 1,
            'do_sample': False,
            #'early_stopping': False,
            #'use_cache': True
        }
        print('InternLMXComposer2 loaded!!!')

    def calculate_generated_text(self, prompt, vision_x):
        """
        Calculate generated text given a prompt and vision data.

        Parameters:
        - prompt (str): The input prompt.
        - vision_x (torch.Tensor): Tensor containing vision data.

        Returns:
        Tuple[str, str]: Tuple containing the result and generated text.
        """
        
        if self.model is None or self.tokenizer is None:
            raise AttributeError('Model or tokenizer is not initialized. Call load_model first!')
    
        with torch.cuda.amp.autocast():
            salt_answer, _ = self.model.chat(self.tokenizer, query=prompt, image=vision_x, history=[], do_sample=self.generation_cfg["do_sample"])

        prediction = salt_answer.split("Answer: ")[-1]

        return salt_answer.strip(), prediction.strip()

    def create_descriptions(self, data):
        """
        Create CoT descriptions using the InternLMXComposer2 model on the given data.

        Parameters:
        - data (List[Dict]): List of input data dictionaries.

        Returns:
        None
        """
        
        for item in tqdm(data):

            caption_prompt = f"<ImageHere>{item['query_prompt']} Sentence: {item['query_raw_texts'][0]}"
            foil_prompt = f"<ImageHere>{item['query_prompt']} Sentence: {item['query_raw_texts'][1]}"

            item_result = {}
  
            salt_caption_answer, caption_prediction = self.calculate_generated_text(caption_prompt, item['query_image_path'])
            salt_foil_answer, foil_prediction = self.calculate_generated_text(foil_prompt, item['query_image_path'])
            
            caption_score = check_answer(caption_prediction, 0) 
            foil_score = check_answer(foil_prediction, 1)

            if caption_score == foil_score and caption_score == [1, 0]:
                score = [1, 0]
            else:
                score = [0, 1]

            item_result = {
                'scores': score,
                'caption_score': caption_score,
                'foil_score': foil_score,
                'salt_caption_answer': salt_caption_answer,
                'salt_foil_answer': salt_foil_answer
            }

            item_id = item['item_id']
            self.results[item_id] = item_result

    def prepare_results(self):
        """
        Prepare and write the results to a JSON file.

        Parameters:
        - None

        Returns:
        None
        """
        write_results(self.output_file, self.results)
        
internLMXComposer2_instance = InternLMXComposer2()
model_registry.register_model("internLMXComposer2", (internLMXComposer2_instance.load_model, internLMXComposer2_instance.create_descriptions, internLMXComposer2_instance.prepare_results))