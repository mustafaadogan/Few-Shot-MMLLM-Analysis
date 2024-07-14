from tqdm import tqdm
from utils.util import write_results, check_answer
from .models import model_registry

import torch

class PaliGemma:
    def __init__(self):
        self.model = None
        self.processor = None
        self.results = {}
        self.device = None
        self.sc_exp_cnt = 1
        self.generation_cfg = {}
        self.output_file = None
        self.scoring_type = None        

    def load_model(self, args) -> None:
        
        """
        Load method model.

        Parameters:
        - args: The args to load the model.

        Returns:
        None
        """
        
        from transformers import AutoProcessor, PaliGemmaForConditionalGeneration, BitsAndBytesConfig

        print('Loading PaliGemma!!!')
        self.device = args.device
        self.output_file = args.output_file
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            args.hf_path,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config
        )
        
        self.model.eval()

        self.processor = AutoProcessor.from_pretrained(args.hf_path)
        
        self.generation_cfg = {
            #'num_beams': 3,
            #'max_new_tokens': 20,
            #"temperature": 0.2,
            #"top_p": 0.7,
            #'top_k': 0,
            #'length_penalty': -2.0,
            #'num_return_sequences': 1,
            'do_sample': False,
            #'early_stopping': False,
            #'use_cache': True
        }

        if args.is_zero_cot_active:
          self.generation_cfg['max_new_tokens'] = 512
        else:
          self.generation_cfg['max_new_tokens'] = 20
          
        self.scoring_type = args.scoring_type
        print('PaliGemma loaded!!!')

    def calculate_generated_text(self, prompt, vision_x):
        """
        Calculate generated text given a prompt and vision data.

        Parameters:
        - prompt (str): The input prompt.
        - vision_x (torch.Tensor): Tensor containing vision data.

        Returns:
        decoded (str): String containing the generated text.
        """
        
        if self.model is None or self.processor is None:
            raise AttributeError('Model or processor is not initialized. Call load_model first!')

        inputs = self.processor(text=prompt, images=vision_x, return_tensors="pt").to(self.device)
        
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = self.model.generate(**inputs, max_new_tokens=self.generation_cfg["max_new_tokens"], do_sample=self.generation_cfg["do_sample"])
            generation = generation[0][input_len:]
            decoded = self.processor.decode(generation, skip_special_tokens=True)

        return decoded

    def test(self, data):
        """
        Test the PaliGemma model on the given data.

        Parameters:
        - data (List[Dict]): List of input data dictionaries.

        Returns:
        None
        """
        
        for item in tqdm(data):

            item_result = {}

            query_caption, query_foil, is_caption_query = item['query_raw_texts'][0], item['query_raw_texts'][1], item['query_raw_texts'][2]
            prompt = ""

            if is_caption_query:
                prompt += f"{item['query_prompt']} {query_caption} Answer:"
            else:
                prompt += f"{item['query_prompt']} {query_foil} Answer:"

            if self.scoring_type == 'generated_text':
                    answer = self.calculate_generated_text(prompt, item['query_image'])
                    
                    score = check_answer(answer, is_caption_query) 
                    item_result = {
                        'scores': score,
                        'caption_order': is_caption_query,
                        'generated_text': answer 
                    }                    
            else:
                raise NotImplementedError(f'{self.scoring_type} not implemented yet!')


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
        
paligemma_instance = PaliGemma()
model_registry.register_model("paligemma", (paligemma_instance.load_model, paligemma_instance.test, paligemma_instance.prepare_results))