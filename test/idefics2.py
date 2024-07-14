from .models import model_registry
from tqdm import tqdm
from utils.util import write_results, check_answer, get_random_index_list

import torch

def prepare_prompt(user_prompt, support_data, query_data, query_prompt):
    """
    Prepare prompts using user prompt, support example  data, and query example data.

    Parameters:
    - user_prompt (str): User prompt for the assistant.
    - support_data (list): List of dictionaries containing support data.
    - query_data (dict): Dictionary containing query data.

    Returns:
    tuple: Tuple containing prompt, and ground truth value.
    """
    
    prompt = []
    for item in support_data:
        caption, foil, cot_caption, cot_foil, is_caption_example = item['caption'], item['foil'], item['cot_caption'], item['cot_foil'], item['is_caption_example']
        
        if not is_caption_example:
            caption, foil = foil, caption

        user_message = {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"{user_prompt} {caption}"},
            ]
        }
        
        assistant_message = {
            "role": "assistant",
            "content": [
                {"type": "text", "text": f"{cot_caption if is_caption_example else cot_foil}"},
            ]
        }
        
        prompt.append(user_message)
        prompt.append(assistant_message)
        

    query_caption_text, query_foil_text = query_data['caption'], query_data['foil']
    
    if not query_data['is_caption_query']:
        query_caption_text, query_foil_text = query_foil_text, query_caption_text

    query_message = {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": f"{query_prompt} {query_caption_text}"},
        ]
    }
    
    prompt.append(query_message)

    return prompt, query_data['is_caption_query']

def prepare_answer(raw_generated_text):
    """
    Prepare answer using raw generated text and query prompt. 

    Parameters:
    - raw_generated_text (str): Generated text by Idefics2.
    - query_prompt (str): Query to find the salt answer.

    Returns:
    salt_result (str): Actual answer of Idefics2.
    """

    salt_result = raw_generated_text.strip().split("Assistant")[-1].strip()

    return salt_result


class Idefics2:
    def __init__(self):
        self.model = None
        self.processor = None
        self.results = {}
        self.device = None
        self.sc_exp_cnt = 1
        self.generation_cfg = {}
        self.output_file = None
        self.scoring_type = None

    def load_model(self, args):
        """
        Load the IDEFICS2 model and processor.

        Parameters:
        - args: The args to load the model.

        Returns:
        None
        """
        
        from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
        
        print('Loading Idefics2!!!')
        checkpoint_path = args.hf_path
        self.device = args.device
        self.scoring_type = args.scoring_type
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.model = AutoModelForVision2Seq.from_pretrained(
            checkpoint_path,  
            torch_dtype=torch.float16,
            quantization_config=quantization_config,
            _attn_implementation="flash_attention_2"  
        )
        
        self.processor = AutoProcessor.from_pretrained(
            checkpoint_path,
            do_image_splitting=False,
            size= {"longest_edge": 448, "shortest_edge": 378}
        ) 
        
        self.model.eval() 
        self.output_file = args.output_file
        self.sc_exp_cnt = args.sc_exp_cnt 

        self.generation_cfg = {
            #'bad_words_ids': bad_words_ids,
            #'do_sample': False,
            #'num_beams': 3
        } 

        if args.is_zero_cot_active or args.is_few_cot_active:
          self.generation_cfg['max_new_tokens'] = 512
        else:
          self.generation_cfg['max_new_tokens'] = 5

        print('Idefics2 loaded!!!')

    def calculate_generated_text(self, prompt, vision_x):
        """
        Calculate generated text given a prompt and vision data.

        Parameters:
        - prompt (str): The input prompt.
        - vision_x (torch.Tensor): Tensor containing vision data.

        Returns:
        Tuple[str, str]: Tuple containing the raw and salt answer text.
        """
        
        if self.model is None or self.processor is None:
            raise AttributeError('Model or processor is not initialized. Call load_model first!')
                  
        prompt = self.processor.apply_chat_template(prompt, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=vision_x, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.generation_cfg['max_new_tokens'])
        
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        
        
        return generated_texts

    def test(self, data):
        """
        Test the IDEFICS2 model on the given data.

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

                support_data = [
                    {
                        'caption': caption,
                        'foil': foil,
                        'cot_caption': cot_caption,
                        'cot_foil': cot_foil,
                        'is_caption_example': is_caption_example,
                    } for (caption, foil), (cot_caption, cot_foil, is_caption_example) in zip(
                        support_class_text_list,
                        cot_info_list
                    )
                ]
               
                query_data = {
                    'caption': item['query_raw_texts'][0],
                    'foil': item['query_raw_texts'][1],
                    'is_caption_query': item['query_raw_texts'][2]
                }
                
                model_prompt, query_caption_order = prepare_prompt(item['support_prompt'], support_data, query_data, item['query_prompt'])
                img_list = support_class_image_list + [item['query_image']]

                item_result = {}
                score = [0, 1]

                if self.scoring_type == "generated_text":
                    model_generated_text = self.calculate_generated_text(model_prompt, img_list)
                    model_salt_generated_text = prepare_answer(model_generated_text)#, model_prompt[-1]["content"][1]["text"])


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


idefics2_instance = Idefics2()
model_registry.register_model("idefics2", (idefics2_instance.load_model, idefics2_instance.test, idefics2_instance.prepare_results))