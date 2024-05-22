import torch
from transformers import AutoProcessor, CLIPModel, CLIPTokenizer, CLIPTextModel
import torch.nn as nn
from tqdm import tqdm
from utils.util import write_results
from .models import model_registry

class ClipModel:
    def __init__(self):
        self.device = None
        self.processor = None
        self.model = None
        self.tokenizer = None
        self.text_encoder = None
        self.cos = nn.CosineSimilarity(dim=0)
        self.results = {}

    def load_model(self, args):
        self.device = args.device
        self.processor = AutoProcessor.from_pretrained(args.hf_path)
        self.model = CLIPModel.from_pretrained(args.hf_path).to(self.device)
        self.tokenizer = CLIPTokenizer.from_pretrained(args.hf_path)
        self.text_encoder = CLIPTextModel.from_pretrained(args.hf_path).to(self.device)

    def process_image_features(self, image):
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors='pt').to(self.device)
            image_features = self.model.get_image_features(**inputs)
        return image_features

    def process_text_features(self, text):
        with torch.no_grad():
            inputs = self.tokenizer(
                    text, 
                    padding="max_length", 
                    return_tensors="pt",
                    ).to(self.device)
            text_features = torch.flatten(self.text_encoder(inputs.input_ids.to(self.device))['last_hidden_state'],1,-1)
        return text_features

    def compute_similarity(self, features1, features2):
        similarity = self.cos(features1[0], features2[0]).item()
        similarity = (similarity + 1) / 2
        return similarity

    def calculate_similarities(self, dataset):
        total_similarities = {}
        
        for target_item in tqdm(dataset):
            target_item_id = target_item['item_id']
            image_similarities = {}
            text_similarities = {}
            for example_item in dataset:
                example_id = example_item['item_id']

                if target_item_id == example_id:
                    continue

                target_image_feature = self.process_image_features(target_item['query_image'])
                example_image_feature = self.process_image_features(example_item['query_image'])
                image_similarity = self.compute_similarity(target_image_feature, example_image_feature)

                target_text_feature = self.process_text_features(target_item['query_raw_texts'][0])
                example_text_feature = self.process_text_features(example_item['query_raw_texts'][0])
                text_similarity = self.compute_similarity(target_text_feature, example_text_feature)

                

                image_similarities[example_id] = image_similarity
                text_similarities[example_id] = text_similarity

            total_similarities[target_item_id] = {
                                                    'similarities': {
                                                        'image': image_similarities, 
                                                        'text': text_similarities
                                                    }
                                                }

        self.results = total_similarities
      
    def prepare_results(self, file_name):
        """
        Prepare and write the results to a JSON file.

        Parameters:
        - file_name (str): Name of the output file.

        Returns:
        None
        """
        write_results(file_name, self.results)

clip_model_instance = ClipModel()
model_registry.register_model("ClipModel", (clip_model_instance.load_model, clip_model_instance.calculate_similarities, clip_model_instance.prepare_results))