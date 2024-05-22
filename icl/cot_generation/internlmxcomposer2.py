from tqdm import tqdm
from utils.util import write_results, check_answer
from .models import model_registry
import torch, auto_gptq
from auto_gptq.modeling import BaseGPTQForCausalLM
from transformers import AutoModel, AutoTokenizer

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

class InternLMXComposer2:
    def __init__(self):
        self.model = None
        self.image_processor = None
        self.tokenizer = None
        self.results= {}
        self.device = None
        

    def load_model(self, args) -> None:
        """
        Load LLaVA model.

        Parameters:
        - args: The args to load the model.

        Returns:
        None
        """
        
        print('Loading InternLMXComposer2!!!')
        self.device = args.device
        
        self.model = InternLMXComposer2QForCausalLM.from_quantized(args.hf_path, trust_remote_code=True, device="cuda:0").eval()
        self.tokenizer = AutoTokenizer.from_pretrained(args.hf_path, trust_remote_code=True)
        
        #self.model.to(self.device)
        
        #self.model.eval()
        
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
        self.scoring_type = args.scoring_type
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
        
        if self.model is None:
            raise AttributeError('Model is not initialized. Call load_model first!')

        
        
        with torch.cuda.amp.autocast():
            salt_answer, _ = self.model.chat(self.tokenizer, query=prompt, image=vision_x, history=[], do_sample=self.generation_cfg["do_sample"])


        prediction = salt_answer.split("Final Answer")[-1]

        return salt_answer.strip(), prediction.strip()

    def test(self, data):
        """
        Test the model on the given data using the specified scoring type.

        Parameters:
        - data (List[Dict]): List of input data dictionaries.

        Returns:
        None
        """
        
        for item in tqdm(data):

            caption_prompt = f"<ImageHere>Analyze the image step by step using available information and determine if the sentence accurately describes it. Your final answer will be Final Answer: Yes/No. Sentence: {item['query_raw_texts'][0]}"
            foil_prompt = f"<ImageHere>Analyze the image step by step using available information and determine if the sentence accurately describes it. Your final answer will be Final Answer: Yes/No. Sentence: {item['query_raw_texts'][1]}"

            item_result = {}

            if self.scoring_type == 'generated_text':
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
            else:
                raise NotImplementedError(f'{self.scoring_type} not implemented yet!')

            item_id = item['item_id']
            self.results[item_id] = item_result

    def prepare_results(self, file_name):
        """
        Prepare and write the results to a JSON file.

        Parameters:
        - file_name (str): Name of the output file.

        Returns:
        None
        """
        write_results(file_name, self.results)
        
internLMXComposer2_instance = InternLMXComposer2()
model_registry.register_model("internLMXComposer2", (internLMXComposer2_instance.load_model, internLMXComposer2_instance.test, internLMXComposer2_instance.prepare_results))
