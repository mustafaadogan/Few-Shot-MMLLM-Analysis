from tqdm import tqdm
from utils.util import write_results, check_answer
from .models import model_registry


import torch
import torchvision.transforms as T

from torchvision.transforms.functional import InterpolationMode

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image, input_size=448, max_num=6):
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

class InterVLChat:
    def __init__(self):
        self.model = None
        self.image_processor = None
        self.tokenizer = None
        self.results= {}
        self.device = None
        

    def load_model(self, args) -> None:
        from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
        #import os
        #os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        """
        Load LLaVA model.

        Parameters:
        - args: The args to load the model.

        Returns:
        None
        """
        
        print('Loading InternVL-Chat!!!')
        self.device = args.device
        self.output_file = args.output_file
        
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
        )
        
        self.model = AutoModel.from_pretrained(
            args.hf_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            #device_map='auto'
        )

        
        self.tokenizer = AutoTokenizer.from_pretrained(args.hf_path, trust_remote_code=True)
        
        #self.model.to(self.device)
        
        self.model.eval()
        
        self.generation_cfg = {
            'num_beams': 1,
            'max_new_tokens': 20,
            #"temperature": 0.2,
            #"top_p": 0.7,
            #'top_k': 0,
            #'length_penalty': -2.0,
            #'num_return_sequences': 1,
            'do_sample': False,
            #'early_stopping': False,
            #'use_cache': True
        }
        self.scoring_type = args.scoring_type
        print('InternVL-Chat loaded!!!')

    def calculate_generated_text(self, prompt, vision_x):
        """
        Calculate generated text given a prompt and vision data.

        Parameters:
        - prompt (str): The input prompt.
        - vision_x (torch.Tensor): Tensor containing vision data.

        Returns:
        Tuple[str, str]: Tuple containing the result and generated text.
        """
        
        response = self.model.chat(self.tokenizer, vision_x, prompt, self.generation_cfg)


        return response

    def test(self, data):
        """
        Test the model on the given data using the specified scoring type.

        Parameters:
        - data (List[Dict]): List of input data dictionaries.

        Returns:
        None
        """
        
        for item in tqdm(data):

            item_result = {}

            query_caption, query_foil, is_caption_query = item['query_raw_texts'][0], item['query_raw_texts'][1], item['query_raw_texts'][2]
            prompt = ""

            if is_caption_query:
                prompt += f"{item['prompt']} {query_caption} Answer:"
            else:
                prompt += f"{item['prompt']} {query_foil} Answer:"

            pixel_values = load_image(item['query_image'], max_num=6).to(torch.bfloat16).to(self.device)
            
            if self.scoring_type == 'generated_text':
                    raw_answer = self.calculate_generated_text(prompt, pixel_values)
                    
                    score = check_answer(raw_answer, is_caption_query) 
                    item_result = {
                        'scores': score,
                        'caption_order': is_caption_query,
                        'generated_text': raw_answer,
                    }
                                
            else:
                raise NotImplementedError(f'{self.scoring_type} not implemented yet!')


            item_id = item['item_id']
            self.results[item_id] = item_result

    def prepare_results(self):
        """
        Prepare and write the results to a JSON file.

        Parameters:
        - None

        Returns:
        None
        """
        write_results(self.output_file, self.results)
        
internVLChat_instance = InterVLChat()
model_registry.register_model("internVLChat", (internVLChat_instance.load_model, internVLChat_instance.test, internVLChat_instance.prepare_results))