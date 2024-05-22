from .models import model_registry
from tqdm import tqdm
from utils.util import write_results, check_answer, get_random_index_list

import torch
from transformers import IdeficsForVisionText2Text, AutoProcessor

def prepare_prompt(user_prompt, support_data, query_data):
    """
    Prepare prompts using user prompt, support example  data, and query example data.

    Parameters:
    - user_prompt (str): User prompt for the assistant.
    - support_data (list): List of dictionaries containing support data.
    - query_data (dict): Dictionary containing query data.

    Returns:
    tuple: Tuple containing prompt, and query caption order.
    """
    
    prompt = []
    for item in support_data:
        img, caption, foil, cot_caption, cot_foil, is_caption_example = item['image'], item['caption'], item['foil'], item['cot_caption'], item['cot_foil'], item['is_caption_example']
        
        if not is_caption_example:
            caption, foil = foil, caption

        prompt += [
            "User:",
            img,
            f"{user_prompt} {caption} \nAssistant: {cot_caption if is_caption_example else cot_foil}\n"
        ]

    query_img, query_caption_text, query_foil_text = query_data['image'], query_data['caption'], query_data['foil']
    
    if not query_data['is_caption_query']:
        query_caption_text, query_foil_text = query_foil_text, query_caption_text

    prompt += [
        "User:",
        query_img,
        f"{user_prompt} {query_caption_text} \nAssistant:" 
    ]

    return prompt, query_data['is_caption_query']

def prepare_answer(raw_generated_text, query_prompt):
    """
    Prepare answer using raw generated text and query prompt. 

    Parameters:
    - raw_generated_text (str): Generated text by Idefics.
    - query_prompt (str): Query to find the salt answer.

    Returns:
    salt_result (str): Actual answer of Idefics.
    """

    salt_result_index = raw_generated_text.find(query_prompt) + len(query_prompt)
    salt_result = raw_generated_text[salt_result_index:]

    return salt_result


class Idefics:
    def __init__(self):
        self.model = None
        self.processor = None
        self.results = {}

    def load_model(self, args):
        """
        Load the IDEFICS model and processor.

        Parameters:
        - args: The args to load the model.

        Returns:
        None
        """
        print('Loading Idefics!!!')
        checkpoint_path = args.hf_path
        self.device = args.device
        self.scoring_type = args.scoring_type
        self.model = IdeficsForVisionText2Text.from_pretrained(checkpoint_path, torch_dtype=torch.float16).to(args.device)
        self.processor = AutoProcessor.from_pretrained(checkpoint_path)
        self.model.eval() 
        self.output_file = args.output_file
        self.sc_exp_cnt = args.sc_exp_cnt 

        bad_words_ids = self.processor.tokenizer(["<image>", "<fake_token_around_image>"], add_special_tokens=False).input_ids

        self.generation_cfg = {
            'bad_words_ids': bad_words_ids,
            'do_sample': False,
            'num_beams': 3
        } 

        self.is_cot_active = args.is_cot_active

        if self.is_cot_active:
          self.generation_cfg['max_new_tokens'] = 512
        else:
          self.generation_cfg['max_new_tokens'] = 5

        print('Idefics loaded!!!')

    def generate_text(self, prompt):
        """
        Generate text based on the given prompts, support data, and query data.

        Parameters:
        - prompt (str): Prompt for the assistant.

        Returns:
        raw_generated_text (str): Generated text data.
        """
        
        inputs = self.processor(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,  
                bad_words_ids=self.generation_cfg['bad_words_ids'], 
                do_sample=self.generation_cfg['do_sample'],
                num_beams=self.generation_cfg['num_beams'],
                max_new_tokens=self.generation_cfg['max_new_tokens'],
            )
            
        raw_generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        return raw_generated_text

    def test(self, data):
        """
        Test the IDEFICS model on the given data.

        Parameters:
        - data (list): List of data items.

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

                support_data = [
                    {
                        'image': img,
                        'caption': caption,
                        'foil': foil,
                        'cot_caption': cot_caption,
                        'cot_foil': cot_foil,
                        'is_caption_example': is_caption_example,
                    } for img, (caption, foil), (cot_caption, cot_foil, is_caption_example) in zip(
                        support_class_image_list,
                        support_class_text_list,
                        cot_info_list
                    )
                ]
               
                query_data = {
                    'image': item['query_image'],
                    'caption': item['query_raw_texts'][0],
                    'foil': item['query_raw_texts'][1],
                    'is_caption_query': item['query_raw_texts'][2]
                }
                

                model_prompt, query_caption_order = prepare_prompt(item['prompt'], support_data, query_data)

                item_result = {}
                score = [0, 1]

                if self.scoring_type == "generated_text":
                    model_generated_text = self.generate_text(model_prompt)
                    model_salt_generated_text = prepare_answer(model_generated_text, model_prompt[-1])


                    score = check_answer(model_salt_generated_text, query_caption_order) 

                    
                    item_result = {
                        'scores': score,
                        'query_caption_order': query_caption_order,
                        'prompt': str(model_prompt),
                        'raw_result': str(model_generated_text),
                        'salt_result': str(model_salt_generated_text)
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


idefics_instance = Idefics()
model_registry.register_model("idefics", (idefics_instance.load_model, idefics_instance.test, idefics_instance.prepare_results))
