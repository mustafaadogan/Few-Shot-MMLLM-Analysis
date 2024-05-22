from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
from .models import model_registry
from tqdm import tqdm
from utils.util import write_results, check_answer, get_random_index_list

import torch


class OpenFlamingo:
    def __init__(self):
        self.model = None
        self.image_processor = None
        self.tokenizer = None
        self.results= {}
        self.device = None
        self.sc_exp_cnt = 1
        

    def load_model(self, args) -> None:
        """
        Load the OpenFlamingo model and processor.

        Parameters:
        - args: The args to load the model.

        Returns:
        None
        """
        
        print('Loading OpenFlamingo!!!')
        self.device = args.device
        self.scoring_type = args.scoring_type
        self.output_file = args.output_file
        self.sc_exp_cnt = args.sc_exp_cnt
        self.model, self.image_processor, self.tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path=args.lang_encoder_path, 
            tokenizer_path=args.tokenizer_path,
            cross_attn_every_n_layers=args.cross_attn_every_n_layers#2
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model.to(self.device)

        self.tokenizer.padding_side = "left"
        checkpoint_path = hf_hub_download(args.hf_path, "checkpoint.pt")
        self.model.load_state_dict(torch.load(checkpoint_path), strict=False)
        self.model.eval()
        
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": ["<|endofchunk|>", "<image>"]}
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        
        self.generation_cfg = {
            'num_beams': 3,
            'length_penalty': -2.0,
        }

        self.is_cot_active = args.is_cot_active
        
        if self.is_cot_active:
          self.generation_cfg['max_new_tokens'] = 512
        else:
          self.generation_cfg['max_new_tokens'] = 5

        print('OpenFlamingo loaded!!!')

    def calculate_generated_text(self, prompt, vision_x):
        """
        Calculate generated text given a prompt and vision data.

        Parameters:
        - prompt (str): The input prompt.
        - vision_x (torch.Tensor): Tensor containing vision data.

        Returns:
        Tuple[str, str]: Tuple containing the raw and salt answer text.
        """
        
        if self.model is None:
            raise AttributeError('Model is not initialized. Call load_model first!')

        lang_x = self.tokenizer(
            [prompt],
            return_tensors="pt",
            padding='longest',
            truncation=True,
            max_length=2000,
        )

        with torch.no_grad():
            generated_text = self.model.generate(
                vision_x=vision_x.to(self.device),
                lang_x=lang_x["input_ids"].to(self.device),
                attention_mask=lang_x["attention_mask"].to(self.device),
                pad_token_id=self.tokenizer.pad_token_id,
                **self.generation_cfg,
            )


        raw_answer = self.tokenizer.decode(generated_text[0], skip_special_tokens=False)
        salt_answer  = raw_answer[len(prompt):].split("<|endofchunk|>")[0]

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
                vision_x = [self.image_processor(x).unsqueeze(0) for x in support_class_image_list]
                vision_x += [self.image_processor(item['query_image']).unsqueeze(0)]
                vision_x = torch.cat(vision_x, dim=0).unsqueeze(1).unsqueeze(0)

                prompt = ''
                for raw_texts, cot_info in zip(support_class_text_list, cot_info_list):
                    support_caption, support_foil = raw_texts[0], raw_texts[1]
                    cot_caption, cot_foil, is_caption_example = cot_info[0], cot_info[1], cot_info[2]

                    if is_caption_example:
                        prompt += f"<image>{item['prompt']} {support_caption} {cot_caption}<|endofchunk|>"
                    else:
                        prompt += f"<image>{item['prompt']} {support_foil} {cot_foil}<|endofchunk|>"


                query_caption, query_foil, is_caption_query = item['query_raw_texts'][0], item['query_raw_texts'][1], item['query_raw_texts'][2]

                if is_caption_query:
                    prompt += f"<image>{item['prompt']} {query_caption} Answer:"
                else:
                    prompt += f"<image>{item['prompt']} {query_foil} Answer:"

                item_result = {}
                score = [0, 1]

                if self.scoring_type == 'generated_text':
                    raw_answer, salt_answer = self.calculate_generated_text(prompt, vision_x)
                    
                    score = check_answer(salt_answer, is_caption_query) 

                    item_result = {'scores': score,
                                'caption_order': is_caption_query,
                                'generated_text': raw_answer,
                                'salt_answer': salt_answer}
                                
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

openflamingo_instance = OpenFlamingo()
model_registry.register_model("openflamingo", (openflamingo_instance.load_model, openflamingo_instance.test, openflamingo_instance.prepare_results))
