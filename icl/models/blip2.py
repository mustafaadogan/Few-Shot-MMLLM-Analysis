from tqdm import tqdm
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

from .models import model_registry
from ..utils.util import write_results
from ..utils.util import get_random_number
from ..utils.util import check_answer
import torch.nn.functional as F


class Blip:
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        self.results = {}

    def load_model(self, device) -> None:
        """
        Load BLIP2 model.

        Parameters:
        - device (Any): The device on which to load the model.

        Returns:
        None
        """
        print("Loading BLIP2!!!")
        self.device = device
        checkpoint_path = "Salesforce/blip2-opt-2.7b"
        self.processor = AutoProcessor.from_pretrained(checkpoint_path)
        self.model = Blip2ForConditionalGeneration.from_pretrained(checkpoint_path, torch_dtype=torch.float16)
        self.model.to(device)
        self.model.eval()
        print("BLIP2 loaded!!!")

    def test(self, data, scoring_type):
        """
        Test the BLIP2 model on the given data.

        Parameters:
        - data (list): List of data items.
        - scoring_type: Scoring type (not used in the provided code).

        Returns:
        None
        """
        for item in tqdm(data):
            query_image, query_caption, query_foil = item['query_image'], item['query_raw_texts'][0], \
            item['query_raw_texts'][1]
            query_caption_order = get_random_number(0, 1)
            if query_caption_order == 0:
                query_texts = [query_caption, query_foil]
            else:
                query_texts = [query_foil, query_caption]

            prompt = f"Question: {item['prompt']} caption 0: {query_texts[0]} caption 1: {query_texts[1]}. \nAnswer:"
            if scoring_type == "generated_text":
                generated_text = self.generate_text(prompt, query_image)
                score = check_answer(generated_text, query_caption_order)
                item_result = {
                    "score": score,
                    "caption_order": query_caption_order,
                    "generated_text": generated_text,
                    "prompt": prompt
                }
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

    def generate_text(self, prompt, image):
        """
        Generate text based on the given prompts, query data.

        Parameters:
        - prompt (str): The input prompt.
        - image (Image): The input image.

        Returns:
        generated_text (str): Generated text data.
        """
        inputs = self.processor(image, text=prompt, return_tensors="pt").to(self.device, torch.float16)
        generated_ids = self.model.generate(**inputs)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        print(f"generated_text = {generated_text}")
        return generated_text

    def calculate_perplexity(self, prompt):
        """
        Generate text based on the given prompts, support data, and query data.

        Parameters:
        - prompt (str): Prompt for the assistant.

        Returns:
        score (float): Perplexity score.
        """
        # --batched mode
        inputs = self.processor(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            model_output = self.model(**inputs)

        logits = model_output.logits[0].to(self.device)
        true_labels = inputs["input_ids"].to(self.device)  # Flatten the true labels

        # Calculate cross-entropy loss
        loss = F.cross_entropy(logits, true_labels)

        # Calculate perplexity
        perplexity = torch.exp(loss)

        return float(perplexity)


blip2_instance = Blip()
model_registry.register_model("blip2", (blip2_instance.load_model, blip2_instance.test, blip2_instance.prepare_results))
