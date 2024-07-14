from tqdm import tqdm
from utils.util import write_results, check_answer
from .models import model_registry

import torch

class LLaVA_Llama3:
    def __init__(self):
        self.model = None
        self.image_processor = None
        self.tokenizer = None
        self.results= {}
        self.device = None
        self.generation_cfg = {}
        self.output_file = None
        

    def load_model(self, args) -> None:
        """
        Load LLaVA model.

        Parameters:
        - args: The args to load the model.

        Returns:
        None
        """
        
        from transformers import AutoProcessor, LlavaForConditionalGeneration

        print('Loading LLaVA_Llama3!!!')

        self.device = args.device
        
        self.model = LlavaForConditionalGeneration.from_pretrained(
            args.hf_path, 
            torch_dtype=torch.float16, 
            low_cpu_mem_usage=True, 
        ).to(self.device)
        
        self.processor = AutoProcessor.from_pretrained(args.hf_path)
        
        self.model.eval()
        
        self.generation_cfg = {
            'num_beams': 3,
            'max_new_tokens': 512,
            "temperature": 0.2,
            "top_p": 0.7,
            #'top_k': 0,
            #'length_penalty': -2.0,
            #'num_return_sequences': 1,
            'do_sample': False,
            #'early_stopping': False,
            'use_cache': True
        }

        print('LLaVA_Llama3 loaded!!!')

    def calculate_generated_text(self, prompt, vision_x):
        """
        Calculate generated text given a prompt and vision data.

        Parameters:
        - prompt (str): The input prompt.
        - vision_x (torch.Tensor): Tensor containing vision data.

        Returns:
        Tuple[str, str]: Tuple containing the result and generated text.
        """
        
        if self.model is None or self.processor is None:
            raise AttributeError('Model or processor is not initialized. Call load_model first!')

        inputs = self.processor(prompt, vision_x, return_tensors='pt').to(self.device, torch.float16)

        with torch.no_grad():
            generate_ids = self.model.generate(
                **inputs, 
                #num_beams=1, 
                max_new_tokens=self.generation_cfg['max_new_tokens'], 
                #temperature=self.generation_cfg['temperature'], 
                #top_p=self.generation_cfg['top_p'], 
                do_sample=self.generation_cfg['do_sample'], 
                use_cache=self.generation_cfg['use_cache']
            ) 

        raw_answer = self.processor.decode(generate_ids[0][2:], skip_special_tokens=True)

        salt_result_index = raw_answer.find("assistant")
        salt_answer = raw_answer[salt_result_index:]

        prediction = salt_answer.split("Final Answer")[-1]

        return raw_answer, salt_answer.strip(), prediction.strip()

    def test(self, data):
        """
        Test the model on the given data using the specified scoring type.

        Parameters:
        - data (List[Dict]): List of input data dictionaries.

        Returns:
        None
        """
        
        for item in tqdm(data):

            caption_prompt = f"<|start_header_id|>user<|end_header_id|>\n\n<image>\n{item['query_prompt']} Sentence: {item['query_raw_texts'][0]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
          
            foil_prompt = f"<|start_header_id|>user<|end_header_id|>\n\n<image>\n{item['query_prompt']} Sentence: {item['query_raw_texts'][1]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

            item_result = {}

            raw_caption_answer, salt_caption_answer, caption_prediction = self.calculate_generated_text(caption_prompt, item['query_image'])
            raw_foil_answer, salt_foil_answer, foil_prediction = self.calculate_generated_text(foil_prompt, item['query_image'])
            
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
                'salt_foil_answer': salt_foil_answer,
                'raw_caption_answer': raw_caption_answer,
                'raw_foil_answer': raw_foil_answer
            }

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

llava_llama3_instance = LLaVA_Llama3()
model_registry.register_model("llava_llama3", (llava_llama3_instance.load_model, llava_llama3_instance.test, llava_llama3_instance.prepare_results))