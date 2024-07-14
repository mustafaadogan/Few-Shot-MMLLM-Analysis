from .models import model_registry
from tqdm import tqdm
from utils.util import write_results, check_answer, get_random_index_list

import torch


class QwenVLChat:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.results = {}
        self.device = None
        self.sc_exp_cnt = 1
        self.generation_cfg = {}
        self.output_file = None
        self.scoring_type = None   
        

    def load_model(self, args) -> None:
        """
        Load the QwenVLChat model and processor.

        Parameters:
        - args: The args to load the model.

        Returns:
        None
        """

        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        print('Loading QwenVLChat!!!')
        self.device = args.device
        self.scoring_type = args.scoring_type
        self.output_file = args.output_file
        self.sc_exp_cnt = args.sc_exp_cnt
        self.model = AutoModelForCausalLM.from_pretrained(
            args.hf_path,
            device_map=self.device,
            trust_remote_code=True
        ).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            args.hf_path,
            trust_remote_code=True
        )

        self.tokenizer.padding_side = 'left'
        self.tokenizer.pad_token_id = self.tokenizer.eod_id

        self.generation_cfg = {
            'num_beams': 3,
            'length_penalty': -2.0,
            'do_sample': False,
            'num_beams': 1,
            'min_new_tokens': 1,
            'length_penalty': 1,
            'num_return_sequences': 1,
            'output_hidden_states': True,
            'use_cache': True,
            'pad_token_id': self.tokenizer.eod_id,
            'eos_token_id': self.tokenizer.eod_id,
        }

        if args.is_zero_cot_active or args.is_few_cot_active:
          self.generation_cfg['max_new_tokens'] = 512
        else:
          self.generation_cfg['max_new_tokens'] = 5

        print('QwenVLChat loaded!!!')

    def calculate_generated_text(self, prompt):
        """
        Calculate generated text given a prompt which contains image paths.

        Parameters:
        - prompt (str): The input prompt.

        Returns:
        answer (str): String containing the salt answer text.
        """
        
        if self.model is None:
            raise AttributeError('Model is not initialized. Call load_model first!')

        lang_x = self.tokenizer(prompt, return_tensors='pt', padding='longest')

        with torch.no_grad():

            generated_text = self.model.generate(
                input_ids=lang_x.input_ids.to(self.device),
                attention_mask=lang_x.attention_mask.to(self.device),
                **self.generation_cfg,
            )

        answer = self.tokenizer.decode(generated_text[0][lang_x.input_ids.size(1):],skip_special_tokens=True).strip()

        return answer


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
                support_classes_image_path_list = [item['support_classes_image_path_list'][i] for i in indexes]
                support_class_text_list = [item['support_classes_raw_texts'][i] for i in indexes]
                cot_info_list = [item['cot_info'][i] for i in indexes]
                

                prompt = ''
                for raw_texts, cot_info, img_path in zip(support_class_text_list, cot_info_list, support_classes_image_path_list):
                    support_caption, support_foil = raw_texts[0], raw_texts[1]
                    cot_caption, cot_foil, is_caption_example = cot_info[0], cot_info[1], cot_info[2]

                    if is_caption_example:
                        prompt += f"<img>{img_path}</img> {item['support_prompt']} {support_caption} {cot_caption}"
                    else:
                        prompt += f"<img>{img_path}</img> {item['support_prompt']} {support_foil} {cot_foil}"


                query_caption, query_foil, is_caption_query = item['query_raw_texts'][0], item['query_raw_texts'][1], item['query_raw_texts'][2]

                if is_caption_query:
                    prompt += f"<img>{item['query_image_path']}</img> {item['query_prompt']} {query_caption} Answer:"
                else:
                    prompt += f"<img>{item['query_image_path']}</img> {item['query_prompt']} {query_foil} Answer:"

                item_result = {}
                score = [0, 1]

                if self.scoring_type == 'generated_text':
                    answer = self.calculate_generated_text(prompt)
                    
                    score = check_answer(answer, is_caption_query) 

                    item_result = {
                        'scores': score,
                        'caption_order': is_caption_query,
                        'generated_text': answer,
                        'prompt': prompt
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

qwenVLChat_instance = QwenVLChat()
model_registry.register_model("qwenVLChat", (qwenVLChat_instance.load_model, qwenVLChat_instance.test, qwenVLChat_instance.prepare_results))