import os.path as osp
import json
from copy import deepcopy
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image
from .util import process_path
from .custom_prompt import Prompt
from .util import get_random_sample


class BaseDataset(Dataset):
    """
    Only loads the JSON annotations.
    """
    def __init__(self, json_path, num_support, mode):
        """
        Initialize the BaseDataset with the specified JSON file path, number of support examples, and mode.

        Parameters:
        - json_path (str): Path to the JSON file containing annotations.
        - num_support (int): Number of support examples to include in each data item.
        - mode (str): Mode for selecting support examples ('RANDOM' or 'CLASS').

        Returns:
        None
        """
        # Initialize BaseDataset
        json_path = process_path(json_path)
        assert osp.isfile(json_path)
        with open(json_path, 'r') as f:
            self.json_data = json.load(f)
        self.ids = sorted(self.json_data.keys())
        for item_id in self.ids:
            self.json_data[item_id]['item_id'] = item_id

        self.num_support = num_support
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
        # Generate dataset based on the specified mode
        data_list = []

        for item_id in self.ids:
            item = self.json_data[item_id]
            support_classes_examples = self._get_support_examples(item_id, mode)
            support_classes_foil_examples = self._get_support_examples(item_id, mode, is_foil=True)

            if len(support_classes_examples) < self.num_support or len(support_classes_foil_examples) < self.num_support:
                continue

            support_classes_examples = get_random_sample(support_classes_examples, self.num_support)
            support_classes_foil_examples = get_random_sample(support_classes_foil_examples, self.num_support)

            data_item = {
                'support_classes_examples': support_classes_examples,
                'support_classes_foil_examples': support_classes_foil_examples,
                'query': item
            }
            data_list.append(data_item)

        return data_list

    def _get_support_examples(self, item_id, mode, is_foil=False):
        """
        Get support examples based on the specified mode.

        Parameters:
        - item_id (str): ID of the query item.
        - mode (str): Mode for selecting support examples ('RANDOM' or 'CLASS').
        - is_foil (bool): Flag indicating whether to get foil examples.

        Returns:
        list: List of support examples.
        """
        # Get support examples based on the specified mode
        return [
            other_item for other_id, other_item in self.json_data.items()
            if self._is_valid_example(other_id, item_id, other_item, mode, is_foil)
        ]

    def _is_valid_example(self, other_id, item_id, other_item, mode, is_foil):
        """
        Check if an example is valid based on mode and foil condition.

        Parameters:
        - other_id (str): ID of the other item.
        - item_id (str): ID of the query item.
        - other_item (dict): Annotation data of the other item.
        - mode (str): Mode for selecting support examples ('RANDOM' or 'CLASS').
        - is_foil (bool): Flag indicating whether to check foil condition.

        Returns:
        bool: True if the example is valid, False otherwise.
        """
        # Check if an example is valid based on mode and foil condition
        return (
            other_id != item_id and
            other_item['mturk']['caption'] > 1 and
            ((mode == 'RANDOM') or
            (mode == 'CLASS' and (
                other_item['classes'] == self.json_data[item_id]['classes_foil'] if is_foil else
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
            num_support,
            img_dir=None,
            mode='CLASS',
            prompt_type='GPT4VisionwoTie',
            tokenizer=None,
            **kwargs,
    ):
        """
        Initialize Dataset_v1 with the specified parameters.

        Parameters:
        - json_path (str): Path to the JSON file containing annotations.
        - num_support (int): Number of support examples to include in each data item.
        - img_dir (str): Directory containing images.
        - mode (str): Mode for selecting support examples ('RANDOM' or 'CLASS').
        - prompt_type (str): Type of prompt used.
        - tokenizer: Tokenizer for processing text.
        - **kwargs: Additional keyword arguments.

        Returns:
        None
        """
        # Initialize Dataset_v1
        self.icl_data = BaseDataset(json_path, num_support, mode)
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
        image_path = osp.join(self.img_dir, image_file)
        image = Image.open(image_path) if self.img_dir else None
        return image, image_path

    def __len__(self):
        return len(self.icl_data)

    def __getitem__(self, index):
        # Get an item from the dataset
        entry = deepcopy(self.icl_data[index])
        item = self.icl_data[index]['query']

        support_img_list, support_img_path_list, support_raw_text_list = self._process_support_examples(
            entry['support_classes_examples'])
        support_foil_img_list, support_foil_img_path_list, support_foil_raw_text_list = self._process_support_examples(
            entry['support_classes_foil_examples'])

        query_img, query_img_path = self._read_image(item)

        item = {
            'index': index,
            'item_id': item['item_id'],
            'query_image': query_img,
            'query_raw_texts': [item['caption'], item['foil']],
            'query_image_path': query_img_path,
            'support_classes_image_list': support_img_list,
            'support_classes_raw_texts': support_raw_text_list,
            'support_classes_image_path_list': support_img_path_list,
            'support_classes_foil_image_list': support_foil_img_list,
            'support_classes_foil_raw_texts': support_foil_raw_text_list,
            'support_classes_foil_image_path_list': support_foil_img_path_list,
            'prompt': self.prompt,
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

