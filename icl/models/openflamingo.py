from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
from .models import model_registry
from tqdm import tqdm
from ..utils.util import write_results
from ..utils.util import get_random_number
from ..utils.util import check_answer
import torch.nn.functional as F

import torch
import torch.nn as nn


class OpenFlamingo:
    def __init__(self):
        self.model = None
        self.image_processor = None
        self.tokenizer = None
        self.results = {}
        self.device = None
        

    def load_model(self, device) -> None:
        """
        Load OpenFlamingo model.

        Parameters:
        - device (Any): The device on which to load the model.

        Returns:
        None
        """
        
        print('Loading OpenFlamingo!!!')
        self.device = device
        self.model, self.image_processor, self.tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="anas-awadalla/mpt-7b",
            tokenizer_path="anas-awadalla/mpt-7b",
            cross_attn_every_n_layers=4 
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model.to(self.device)
        self.crit = nn.CrossEntropyLoss(
            reduction='none',
            ignore_index=self.tokenizer.pad_token_id,
        )
        self.tokenizer.padding_side = "left"
        checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-9B-vitl-mpt7b", "checkpoint.pt")
        self.model.load_state_dict(torch.load(checkpoint_path), strict=False)
        self.model.eval()
        print('OpenFlamingo loaded!!!')

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

        lang_x = self.tokenizer(
            [prompt],
            return_tensors="pt",
            padding='longest',
            truncation=True,
            max_length=2000,
            add_special_tokens=False,
        )

        with torch.no_grad():
            generated_text = self.model.generate(
                vision_x=vision_x.to(self.device),
                lang_x=lang_x["input_ids"].to(self.device),
                attention_mask=lang_x["attention_mask"].to(self.device),
                max_new_tokens=30,
                num_beams=1,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )

        text = self.tokenizer.decode(generated_text[0], skip_special_tokens=True)

        return text

    def calculate_perplexity(self, prompt, vision_x):
        """
        Calculate the perplexity score given a prompt and vision data.

        Parameters:
        - prompt (str): The input prompt.
        - vision_x (torch.Tensor): Tensor containing vision data.

        Returns:
        float: Model score.
        """
        
        if self.model is None:
            raise AttributeError('Model is not initialized. Call load_model first!')

        lang_x = self.tokenizer(
            [prompt],
            return_tensors="pt",
        )

        with torch.no_grad():
            model_output = self.model(
                vision_x=vision_x.to(self.device),
                lang_x=lang_x["input_ids"].to(self.device),
                attention_mask=lang_x["attention_mask"].to(self.device)
            )

        logits = model_output.logits[0].to(self.device)
        true_labels = lang_x["input_ids"].view(-1).to(self.device)  # Flatten the true labels
        
        # Calculate cross-entropy loss
        loss = F.cross_entropy(logits, true_labels)
        
        # Calculate perplexity
        perplexity = torch.exp(loss)

        return float(perplexity)

    def test(self, data, scoring_type):
        """
        Test the model on the given data using the specified scoring type.

        Parameters:
        - data (List[Dict]): List of input data dictionaries.
        - scoring_type (str): Type of scoring to perform.

        Returns:
        None
        """
        
        for item in tqdm(data):
            vision_x = [self.image_processor(x).unsqueeze(0) for x in item['support_classes_image_list']]
            vision_x += [self.image_processor(item['query_image']).unsqueeze(0)]
            vision_x = torch.cat(vision_x, dim=0).unsqueeze(1).unsqueeze(0)

            prompt = ''
            for raw_texts in item['support_classes_raw_texts']:
                support_caption, support_foil = raw_texts[0], raw_texts[1]
                support_example_caption_order = get_random_number(0, 1)

                if support_example_caption_order == 1:
                    prompt += f'<image>{item["prompt"]} caption 0: {support_foil} caption 1: {support_caption} '
                else:
                    prompt += f'<image>{item["prompt"]} caption 0: {support_caption} caption 1: {support_foil} '

                if scoring_type == 'generated_text':
                    prompt += f'Final Answer: caption {support_example_caption_order}<|endofchunk|>'
                elif scoring_type == 'perplexity':
                    prompt += f'Correct caption: {support_caption}<|endofchunk|>'
                else:
                    raise NotImplementedError(f'{scoring_type} not implemented yet!')

            query_caption, query_foil = item['query_raw_texts'][0], item['query_raw_texts'][1]
            query_caption_order = get_random_number(0, 1)

            if query_caption_order == 1:
                prompt += f'<image>{item["prompt"]}caption 0: {query_foil} caption 1: {query_caption} '
            else:
                prompt += f'<image>{item["prompt"]}caption 0: {query_caption} caption 1: {query_foil} '

            

            item_result = {}

            if scoring_type == 'generated_text':
                prompt += 'Final Answer: '
                generated_text = self.calculate_generated_text(prompt, vision_x)
                
                score = check_answer(generated_text, query_caption_order) 

                item_result = {'scores': score,
                               'caption_order': query_caption_order,
                               'generated_text': generated_text}

            elif scoring_type == 'perplexity':
                caption_prompt = prompt + f'Correct caption: {query_caption} '
                foil_prompt = prompt + f'Correct caption: {query_foil} '

                caption_score = self.calculate_perplexity(caption_prompt, vision_x)
                foil_score = self.calculate_perplexity(foil_prompt, vision_x)

                item_result = {'scores': [caption_score, foil_score]}

            else:
                raise NotImplementedError(f'{scoring_type} not implemented yet!')

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

openflamingo_instance = OpenFlamingo()
model_registry.register_model("openflamingo", (openflamingo_instance.load_model, openflamingo_instance.test, openflamingo_instance.prepare_results))
