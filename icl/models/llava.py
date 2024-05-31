from tqdm import tqdm
from utils.util import write_results, check_answer
from .models import model_registry

import torch

class LLaVA:
    def __init__(self):
        self.model = None
        self.image_processor = None
        self.tokenizer = None
        self.results= {}
        self.device = None
        

    def load_model(self, args) -> None:
        from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, BitsAndBytesConfig
        """
        Load LLaVA model.

        Parameters:
        - args: The args to load the model.

        Returns:
        None
        """
        
        print('Loading LLaVA!!!')
        self.device = args.device
        self.output_file = args.output_file
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            args.hf_path,
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True,
            quantization_config=quantization_config,
            #device_map="auto"
            #load_in_4bit=True
        )
        
        self.processor = LlavaNextProcessor.from_pretrained(args.hf_path)
        
        #self.model.to(self.device)
        
        self.model.eval()
        
        self.generation_cfg = {
            'num_beams': 3,
            'max_new_tokens': 20,
            "temperature": 0.2,
            "top_p": 0.7,
            #'top_k': 0,
            #'length_penalty': -2.0,
            #'num_return_sequences': 1,
            'do_sample': True,
            #'early_stopping': False,
            'use_cache': True
        }
        self.scoring_type = args.scoring_type
        print('LLaVA loaded!!!')

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

        inputs = self.processor(prompt, vision_x, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generate_ids = self.model.generate(
                **inputs, 
                num_beams=1, 
                max_new_tokens=self.generation_cfg['max_new_tokens'], 
                temperature=self.generation_cfg['temperature'], 
                top_p=self.generation_cfg['top_p'], 
                do_sample=self.generation_cfg['do_sample'], 
                use_cache=self.generation_cfg['use_cache']
            ) 

        raw_answer = self.processor.decode(generate_ids[0], skip_special_tokens=True)

        salt_result_index = raw_answer.find("assistant")
        salt_answer = raw_answer[salt_result_index:]

        return raw_answer, salt_answer.strip()

    def test(self, data):
        """
        Test the model on the given data using the specified scoring type.

        Parameters:
        - data (List[Dict]): List of input data dictionaries.

        Returns:
        None
        """
        
        for item in tqdm(data):

            item_result = {}

            query_caption, query_foil, is_caption_query = item['query_raw_texts'][0], item['query_raw_texts'][1], item['query_raw_texts'][2]
            prompt = f"<|im_start|>system\n{item['prompt']}<|im_end|>"

            if is_caption_query:
                prompt += f"<|im_start|>user\n<image>\n{item['prompt']} {query_caption} Answer:<|im_end|><|im_start|>assistant\n"
            else:
                prompt += f"<|im_start|>user\n<image>\n{item['prompt']} {query_foil} Answer:<|im_end|><|im_start|>assistant\n"

            if self.scoring_type == 'generated_text':
                    raw_answer, salt_answer = self.calculate_generated_text(prompt, item['query_image'])
                    
                    score = check_answer(salt_answer, is_caption_query) 
                    item_result = {'scores': score,
                                   'caption_order': is_caption_query,
                                   'generated_text': raw_answer,
                                   'salt_answer': salt_answer}
                                
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
        
llava_instance = LLaVA()
model_registry.register_model("llava", (llava_instance.load_model, llava_instance.test, llava_instance.prepare_results))