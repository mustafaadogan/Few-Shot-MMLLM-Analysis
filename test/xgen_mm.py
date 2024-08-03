from .models import model_registry
from tqdm import tqdm
from utils.util import write_results, check_answer, get_random_index_list

import torch 
                     
class XGenMM:
    def __init__(self):
        self.model = None
        self.image_processor = None
        self.tokenizer = None
        self.results = {}
        self.device = None
        self.sc_exp_cnt = 1
        self.generation_cfg = {}
        self.output_file = None
        self.scoring_type = None        
   
    def load_model(self, args) -> None:
        """
        Load the XGenMM model and processor.

        Parameters:
        - args: The args to load the model.

        Returns:
        None
        """
        
        from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor
        
        print('Loading XGenMM!!!')
        self.device = args.device
        self.scoring_type = args.scoring_type
        self.output_file = args.output_file
        self.sc_exp_cnt = args.sc_exp_cnt
        
        self.model = AutoModelForVision2Seq.from_pretrained(args.hf_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(args.hf_path, trust_remote_code=True, use_fast=False, legacy=False)
        self.image_processor = AutoImageProcessor.from_pretrained(args.hf_path, trust_remote_code=True)
        self.tokenizer = self.model.update_special_tokens(self.tokenizer)
        self.tokenizer.padding_side = "left"
           
        num_tokens_per_vis = self.model.vlm.num_tokens_per_vis
        self.placeholder_image_tokens = "<image placeholder>" * (num_tokens_per_vis - 1)
            
            
        self.model.to(self.device)

        self.model.eval()
            
        self.generation_cfg = {
            'num_beams': 1,
            'top_p': None,
            'do_sample': False,
            'length_penalty': 1.0,
            'repetition_penalty': 2.0
        }

        if args.is_zero_cot_active or args.is_few_cot_active:
          self.generation_cfg['max_new_tokens'] = 512
        else:
          self.generation_cfg['max_new_tokens'] = 5

        print('XGenMM loaded!!!')

    def calculate_generated_text(self, prompt, vision_x):
        """
        Calculate generated text given a prompt and vision data.

        Parameters:
        - prompt (str): The input prompt.
        - vision_x (torch.Tensor): Tensor containing vision data.

        Returns:
        Tuple[str, str]: Tuple containing the raw and salt answer text.
        """
        
        if self.model is None or self.tokenizer is None or self.image_processor is None:
            raise AttributeError('Model or tokenizer or image processor is not initialized. Call load_model first!')

   
        inputs = self.image_processor(vision_x, return_tensors="pt")
        language_inputs = self.tokenizer([prompt], return_tensors="pt")
        inputs.update(language_inputs)
        
        inputs = {name: tensor.to(self.device) for name, tensor in inputs.items()}
        
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            generated_text = self.model.generate(
                **inputs,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=self.generation_cfg['do_sample'],
                max_new_tokens=self.generation_cfg['max_new_tokens'],
                top_p=self.generation_cfg['top_p'],
                num_beams=self.generation_cfg['num_beams'],
                length_penalty=self.generation_cfg['length_penalty'],
                repetition_penalty=self.generation_cfg['repetition_penalty'],
            ) 
                                            
        prediction = self.tokenizer.decode(generated_text[0], skip_special_tokens=True)     

        return prediction

    def test(self, data):
        """
        Test the model on the given data using the specified scoring type.

        Parameters:
        - data (List[Dict]): List of input data dictionaries.

        Returns:
        None
        """
        
        for item in tqdm(data):
            assert len(item['support_classes_image_list']) == len(item['support_classes_raw_texts']), "Image-Caption count mismatch!"
            sc_results = {}
            sc_results['raw_results'] = []
            sc_results['score_list'] = []
            initial_index_list = list(range(len(item['support_classes_image_list'])))
            index_list = get_random_index_list(initial_index_list, self.sc_exp_cnt)

            for indexes in index_list:
                support_class_image_list = [item['support_classes_image_list'][i] for i in indexes]
                support_class_text_list = [item['support_classes_raw_texts'][i] for i in indexes]
                cot_info_list = [item['cot_info'][i] for i in indexes]
                
                vision_x = support_class_image_list
                vision_x += [item['query_image']]
                
                prompt = ''
                
                for raw_texts, cot_info in zip(support_class_text_list, cot_info_list):
                    support_caption, support_foil = raw_texts[0], raw_texts[1]
                    cot_caption, cot_foil, is_caption_example = cot_info[0], cot_info[1], cot_info[2]

                    if is_caption_example:
                        prompt += f"<image>{self.placeholder_image_tokens} {item['support_prompt']} {support_caption} {cot_caption} <|endofchunk|>"
                    else:
                        prompt += f"<image>{self.placeholder_image_tokens} {item['support_prompt']} {support_foil} {cot_foil} <|endofchunk|>"

                query_caption, query_foil, is_caption_query = item['query_raw_texts'][0], item['query_raw_texts'][1], item['query_raw_texts'][2]

                if is_caption_query:
                    prompt += f"<image>{self.placeholder_image_tokens} {item['query_prompt']} {query_caption} Answer:"
                else:
                    prompt += f"<image>{self.placeholder_image_tokens} {item['query_prompt']} {query_foil} Answer:"

                item_result = {}
                score = [0, 1]

                if self.scoring_type == 'generated_text':
                    
                    answer = self.calculate_generated_text(prompt, vision_x)
                    
                    score = check_answer(answer, is_caption_query) 

                    item_result = {
                        'scores': score,
                        'caption_order': is_caption_query,
                        'generated_text': answer,
                        #'prompt': prompt
                    }
                                
                else:
                    raise NotImplementedError(f'{self.scoring_type} not implemented yet!')
     
                sc_results['raw_results'].append(item_result)
                sc_results['score_list'].append(score)

                if sc_results['score_list'].count([1, 0]) >= (self.sc_exp_cnt/2) or sc_results['score_list'].count([0, 1]) >= (self.sc_exp_cnt/2):
                    break

            if sc_results['score_list'].count([1, 0]) >= (self.sc_exp_cnt/2):
                final_result = [1, 0] 
            else:
                final_result = [0, 1]
                
            sc_results['scores'] = final_result
            item_id = item['item_id']
            self.results[item_id] = sc_results
    
    def prepare_results(self):
        """
        Prepare and write the results to a JSON file.

        Parameters:
        - None

        Returns:
        None
        """
        write_results(self.output_file, self.results)

xgen_mm_instance = XGenMM()
model_registry.register_model("xgen_mm", (xgen_mm_instance.load_model, xgen_mm_instance.test, xgen_mm_instance.prepare_results))