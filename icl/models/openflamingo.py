import torch
from open_flamingo import create_model_and_transforms
from huggingface_hub import hf_hub_download
from .models import model_registry
from tqdm import tqdm
from utils.util import write_results
import torch.nn as nn

class OpenFlamingo:
    def __init__(self):
        self.model = None
        self.image_processor = None
        self.tokenizer = None
        self.results = {}

    def load_model(self):
        self.model, self.image_processor, self.tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
            tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
            cross_attn_every_n_layers=1
        )
        self.crit = nn.CrossEntropyLoss(
                        reduction='none',
                        ignore_index=self.tokenizer.pad_token_id,
                    )
        self.tokenizer.padding_side = "left" # For generation padding tokens should be on the left
        checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
        self.model.load_state_dict(torch.load(checkpoint_path), strict=False)
        self.model.eval()
    
    def calculate_score(self, prompt, vision_x):
        if self.model is None:
            raise AttributeError('Model is not initialized. Call load_model first!')
        
        lang_x = self.tokenizer(
            [prompt],
            return_tensors="pt",
        )

        with torch.no_grad():
            # Pass the processed images and text tokens through the model
            model_output = self.model(vision_x=vision_x,
                                lang_x=lang_x["input_ids"],
                                attention_mask=lang_x["attention_mask"]
                            )
            
            logits = model_output.logits
            labels = lang_x['input_ids']
            lengths = lang_x['attention_mask'].sum(dim=-1)
            score = self.crit(logits.reshape(-1, logits.shape[-1]), labels.view(-1).to(logits.device))
            score = score.reshape_as(labels)
            score = score.sum(dim=1) / lengths.to(score.device)
            score = score.mean(dim=0).exp().tolist()
        
        return score
    
    def test(self, data):

        for item in tqdm(data):
    
            """
                Step 1: Preprocessing images
                Details: For OpenFlamingo, we expect the image to be a torch tensor of shape 
                batch_size x num_media x num_frames x channels x height x width. 
                In this case batch_size = 1, num_media = 3, num_frames = 1,
                channels = 3, height = 224, width = 224.
            """
            vision_x = []
            vision_x += [self.image_processor(x).unsqueeze(0) for x in item['support_classes_image_list']]
            vision_x += [self.image_processor(x).unsqueeze(0) for x in item['support_classes_foil_image_list']]
            vision_x += [self.image_processor(item['query_image']).unsqueeze(0)]
            vision_x = torch.cat(vision_x, dim=0)
            vision_x = vision_x.unsqueeze(1).unsqueeze(0)

            """
                Step 2: Preprocessing text
                Details: In the text we expect an <image> special token to indicate where an image is.
                We also expect an <|endofchunk|> special token to indicate the end of the text 
                portion associated with an image.
            """    
            prompt = ''

            for support_caption_text in item['support_classes_raw_texts']:
                prompt += f'<image>{support_caption_text[0]}<|endofchunk|>'

            for support_foil_text in item['support_classes_foil_raw_texts']:
                prompt += f'<image>{support_foil_text[1]}<|endofchunk|>'

            query_caption_text = item['query_raw_texts'][0]
            query_foil_text = item['query_raw_texts'][1]

            caption_prompt = prompt + f'<image>{query_caption_text}<|endofchunk|>'
            foil_prompt = prompt + f'<image>{query_foil_text}<|endofchunk|>'

            """
                Step 3: Calculate perplexity
            """

            caption_score = self.calculate_score(caption_prompt, vision_x)
            foil_score = self.calculate_score(foil_prompt, vision_x)
            item_id = item['item_id']
            self.results[item_id] = {'scores': [caption_score, foil_score]}
    
    def prepare_results(self, task_name):
        write_results(task_name[:-5] + "_openflamingo.json", self.results)

openflamingo_instance = OpenFlamingo()
model_registry.register_model("openflamingo", (openflamingo_instance.load_model, openflamingo_instance.test, openflamingo_instance.prepare_results))
