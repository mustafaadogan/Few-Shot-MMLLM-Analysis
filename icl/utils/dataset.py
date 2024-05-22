import os.path as osp
import json
from copy import deepcopy
from torch.utils.data import Dataset
from PIL import Image
from .util import process_path, get_random_number, get_random_sample, hash_dict
from .custom_prompt import Prompt


def capitalize_and_add_dot(string):
    # Capitalize the first letter
    capitalized_string = string.capitalize()
    
    # Add a dot at the end if there isn't one already
    if not capitalized_string.endswith('.'):
        capitalized_string += '.'
    
    return capitalized_string.strip()

class BaseDataset(Dataset):
    """
    Only loads the JSON annotations, and creates support set.
    """
    def __init__(self, json_path, num_support, mode, similarity_data_path, top_k, top_n, is_cot_active, cot_desc_data_path):
        """
        Initialize the BaseDataset with the specified JSON file path, number of support examples, and mode.

        Parameters:
        - json_path (str): Path to the JSON file containing annotations.
        - num_support (int): Number of support examples to include in each data item.
        - mode (str): Mode for selecting support examples ('RANDOM', 'CLASS', 'SIMILAR').
        - similarity_data_path (str): Path to JSON file containing similarity scores of image and texts.
        - top_k (int): Number of visual similar examples.
        - top_n (int): Number of textual similar examples.
        - is_cot_active (bool): CoT analysis active or not.
        - cot_desc_data_path (str): Path to JSON file containing CoT description of image-text pairs.
        
        Returns:
        None
        """
        # Initialize BaseDataset
        self.is_cot_active = is_cot_active
        json_path = process_path(json_path)
        assert osp.isfile(json_path), 'Needs a valid annotation file path'
        with open(json_path, 'r') as f:
            self.json_data = json.load(f)
        self.ids = sorted(self.json_data.keys())
        for item_id in self.ids:
            self.json_data[item_id]['item_id'] = item_id

        self.num_support = num_support

        similarity_data_path = process_path(similarity_data_path)  
        assert mode != "SIMILAR" or osp.isfile(similarity_data_path), 'In SIMILAR example selection mode, needs a valid similarity annotation file path'
        if mode == "SIMILAR":
            with open(similarity_data_path, 'r') as f:
                self.similarity_data = json.load(f)

        cot_desc_data_path = process_path(cot_desc_data_path)
        assert is_cot_active == False or osp.isfile(cot_desc_data_path),'In CoT active mode, needs a valid CoT description file path'
        if is_cot_active:
            with open(cot_desc_data_path, 'r') as f:
                self.cot_data = json.load(f)

        self.top_k = top_k
        self.top_n = top_n

        assert mode != "SIMILAR" or top_k >= top_n, 'TOP_K should be equal or greater than TOP_N in SIMILAR example selection mode'

        self.icl_data = self._generate_data(mode)
        print(f'Dataset being processed: {json_path}')
        print(f'Dataset length: {len(self.icl_data)}')

    def __len__(self):
        return len(self.icl_data)

    def _generate_data(self, mode):
        """
        Generate the dataset based on the specified mode.

        Parameters:
        - mode (str): Mode for selecting support examples ('RANDOM' or 'CLASS').

        Returns:
        list: List of data items, each containing support and query examples.
        """
        example_count_per_set = self.num_support

        if mode == 'SIMILAR':
            example_count_per_set = self.top_n

        # Generate dataset based on the specified mode
        data_list = []

        for item_id in self.ids:
            item = self.json_data[item_id]
            
            if item['mturk']['caption'] <= 1:
                continue

            support_classes_examples = self._get_support_examples(item_id, mode)

            if len(support_classes_examples) != example_count_per_set:
                continue
            
            for v in support_classes_examples:
              v.update(
                  {
                      'caption': capitalize_and_add_dot(v['caption']),
                      'foil':    capitalize_and_add_dot(v['foil']) 
                  }
              )
              is_caption_example = get_random_number(0, 1, hash_dict(v)) == 0
              v['is_caption_example'] = is_caption_example

            data_item = {
                'support_classes_examples': support_classes_examples,
                'query': item
            }
            data_list.append(data_item)

        return data_list

    def _get_support_examples(self, item_id, mode):
        """
        Get support examples based on the specified mode.

        Parameters:
        - item_id (str): ID of the query item.
        - mode (str): Mode for selecting support examples ('RANDOM', 'CLASS' or 'SIMILAR').

        Returns:
        list: List of support examples.
        """
        # Get support examples based on the specified mode
        examples = []

        if mode == "SIMILAR":
            sorted_similar_image_list = list(sorted(self.similarity_data[item_id]["similarities"]["image"].keys(), key=lambda x: self.similarity_data[item_id]["similarities"]["image"][x], reverse=True))
            similar_image_list = []
            if self.is_cot_active:
                for k in sorted_similar_image_list:

                    if self.cot_data[k]["scores"] == [1, 0]:
                        similar_image_list.append(k)
            else:
                similar_image_list = sorted_similar_image_list

            similar_text_list = []

            for k in similar_image_list[:self.top_k]:
                similar_text_list.append((k, self.similarity_data[item_id]["similarities"]["text"][k]))

            sorted_similar_text_list = sorted(similar_text_list, key=lambda x: x[1], reverse=True)[:self.top_n]

            for k in sorted_similar_text_list:
                similar_example = self.json_data[k[0]]

                if self.is_cot_active:
                    similar_example["caption_cot_desc"] = self.cot_data[k[0]]["salt_caption_answer"]
                    similar_example["foil_cot_desc"] = self.cot_data[k[0]]["salt_foil_answer"]
                
                examples.append(similar_example)

        else:

            for other_id, other_item in self.json_data.items():
                if self._is_valid_example(other_id, item_id, other_item, mode):
                    example = other_item
                else:
                    continue
                if self.is_cot_active:
                    example["caption_cot_desc"] = self.cot_data[k[0]]["salt_caption_answer"]
                    example["foil_cot_desc"] = self.cot_data[k[0]]["salt_foil_answer"]
                
                examples.append(example)
            
            examples = get_random_sample(examples, min(self.num_support, len(examples)))

        return examples

    def _is_valid_example(self, other_id, item_id, other_item, mode):
        """
        Check if an example is valid based on mode and foil condition.

        Parameters:
        - other_id (str): ID of the other item.
        - item_id (str): ID of the query item.
        - other_item (dict): Annotation data of the other item.
        - mode (str): Mode for selecting support examples ('RANDOM' or 'CLASS').

        Returns:
        bool: True if the example is valid, False otherwise.
        """
        # Check if an example is valid based on mode and foil condition
        return (
            other_id != item_id and
            (self.is_cot_active == False or self.cot_data[item_id]["scores"] == [1, 0]) and
            other_item['mturk']['caption'] > 1 and
            ((mode == 'RANDOM') or
            (mode == 'CLASS' and (
                other_item['classes'] == self.json_data[item_id]['classes'])
            ))        
        )

    def __getitem__(self, index):
        return self.icl_data[index]
    

class Dataset_v1(Dataset):
    """
    Read also the images in addition to the raw JSON data.
    """

    def __init__(
            self,
            json_path,
            num_support=0,
            similarity_data_path=None,
            top_k=20,
            top_n=0,
            img_dir=None,
            mode='RANDOM',
            prompt_type='ITM',
            is_cot_active=False,
            cot_desc_data_path=None,
            tokenizer=None,
            **kwargs,
    ):
        """
        Initialize Dataset_v1 with the specified parameters.

        Parameters:
        - json_path (str): Path to the JSON file containing annotations.
        - num_support (int): Number of support examples to include in each data item.
        - similarity_data_path (str): Path to JSON file containing similarity scores of image and texts.
        - top_k (int): Number of visual similar examples.
        - top_n (int): Number of textual similar examples.
        - img_dir (str): Directory containing images.
        - mode (str): Mode for selecting support examples ('RANDOM' or 'CLASS').
        - prompt_type (str): Type of prompt used.
        - is_cot_active (bool): CoT analysis active or not.
        - cot_desc_data_path (str): Path to JSON file containing CoT description of image-text pairs.
        - tokenizer: Tokenizer for processing text.
        - **kwargs: Additional keyword arguments.

        Returns:
        None
        """
        # Initialize Dataset_v1
        self.icl_data = BaseDataset(json_path=json_path, 
                                    num_support=num_support, 
                                    mode=mode, 
                                    similarity_data_path=similarity_data_path, 
                                    top_k=top_k, 
                                    top_n=top_n,
                                    is_cot_active=is_cot_active,
                                    cot_desc_data_path=cot_desc_data_path)
        
        self.is_cot_active = is_cot_active
        if is_cot_active:
          prompt_type = 'CoT_ITM'
          
        self.prompt = Prompt(prompt_type).prompt
        self.tokenizer = tokenizer
        self.kwargs = kwargs
        self.img_dir = process_path(img_dir) if img_dir is not None else None

    def _read_image(self, item):
        """
        Read an image from the specified item.

        Parameters:
        - item (dict): Item containing information about the image.

        Returns:
        tuple: Tuple containing image and image path.
        """
        image_file = item['image_file']
        image_path = osp.join(self.img_dir, image_file) if self.img_dir else None
        image = Image.open(image_path) if self.img_dir else None
        return image, image_path

    def __len__(self):
        return len(self.icl_data)

    def __getitem__(self, index):
        # Get an item from the dataset
        entry = deepcopy(self.icl_data[index])
        item = self.icl_data[index]['query']

        support_img_list, support_img_path_list, support_raw_text_list = self._process_support_examples(
            entry['support_classes_examples']
        )

        cot_info = []
        for e in entry['support_classes_examples']:
            if self.is_cot_active:
                cot_info.append(("Answer:" + e['caption_cot_desc'], "Answer:" + e['foil_cot_desc'], e['is_caption_example']))
            else:
                cot_info.append(("Answer: Yes", "Answer: No", e['is_caption_example']))

        query_img, query_img_path = self._read_image(item)

        item = {
            'index': index,
            'item_id': item['item_id'],
            'query_image': query_img,
            'query_raw_texts': [capitalize_and_add_dot(item['caption']), capitalize_and_add_dot(item['foil']), get_random_number(0, 1, hash_dict(entry)) == 0],
            'query_image_path': query_img_path,
            'support_classes_image_list': support_img_list,
            'support_classes_raw_texts': support_raw_text_list,
            'support_classes_image_path_list': support_img_path_list,
            'prompt': self.prompt,
            'cot_info': cot_info
        }
        return item

    def _process_support_examples(self, examples):
        """
        Process support examples and return relevant lists.

        Parameters:
        - examples (list): List of support examples.

        Returns:
        tuple: Tuple containing lists of images, image paths, and raw texts.
        """
        img_list, img_path_list, raw_text_list = [], [], []
        for subentry in examples:
            img, img_path = self._read_image(subentry)
            img_list.append(img)
            img_path_list.append(img_path)
            raw_text_list.append([subentry['caption'], subentry['foil']])
        return img_list, img_path_list, raw_text_list