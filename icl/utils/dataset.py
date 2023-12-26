import os
import os.path as osp
import json
import random
from copy import deepcopy

import numpy as np
from torch.utils.data import Dataset
from torchvision.io import read_image
from PIL import Image
from .util import process_path


class BaseDataset(Dataset):
    """
    Only loads the JSON annotations.
    """
    def __init__(self, json_path, num_support, mode):
        self.num_support = num_support
        json_path = process_path(json_path)
        assert osp.isfile(json_path)
        with open(json_path, 'r') as f:
            self.json_data = json.load(f)
        self.ids = list(self.json_data.keys())
        self.ids.sort()
        for item_id in self.ids:
            self.json_data[item_id]['item_id'] = item_id
        
        self.icl_data = self._generate_data(mode)

    def __len__(self):
        return len(self.icl_data)
    
    def _generate_data(self, mode):
        data_list = []
        random.seed(42)
        
        for item_id in self.ids:
            item = self.json_data[item_id]
            support_classes_examples = []
            support_classes_foil_examples = []

            # Extract support classes examples
            for caption_item_id in self.ids:
                if caption_item_id == item_id:
                    continue
                caption_item = self.json_data[caption_item_id]

                if mode == 'RANDOM':
                    support_classes_examples.append(caption_item)
                elif mode == 'CLASS':
                    if caption_item['classes'] == item['classes']:
                        support_classes_examples.append(caption_item)
                else:
                    raise NotImplementedError('This mode is not supported in this context. Use RANDOM or CLASS instead.')
            
            # Check support classes example count
            if len(support_classes_examples) < self.num_support:
                continue
            else:
                support_classes_examples = random.sample(support_classes_examples, self.num_support)
                

            # Extract support classes foil examples
            for foil_item_id in self.ids:
                if foil_item_id == item_id:
                    continue
                foil_item = self.json_data[foil_item_id]

                if mode == 'RANDOM':
                    support_classes_foil_examples.append(foil_item)
                elif mode == 'CLASS':
                    if foil_item['classes'] == item['classes_foil']:
                        support_classes_foil_examples.append(foil_item)
                else:
                    raise NotImplementedError('This mode is not supported in this context. Use RANDOM or CLASS instead.')
                
            
            # Check support classes foil example count
            if len(support_classes_foil_examples) < self.num_support:
                continue
            else:
                support_classes_foil_examples = random.sample(support_classes_foil_examples, self.num_support)
                    
            data_item = {
                'support_classes_examples': support_classes_examples,
                'support_classes_foil_examples': support_classes_foil_examples,
                'query': item
            }
            data_list.append(data_item)

        return data_list

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
        tokenizer=None,
        **kwargs,
    ):
        self.icl_data = BaseDataset(json_path, num_support, mode)
        self.tokenizer = tokenizer
        self.kwargs = kwargs
        
        self.img_dir = None
        if img_dir is not None:
            self.img_dir = process_path(img_dir)
    
    def _read_image(self, item):
        # find the full path
        image_file = item['image_file']
        image_path = None
        image_dir = self.img_dir
        image_path = osp.join(image_dir, image_file)

        image = Image.open(
                image_path,
            )
        
        return image, image_path

    def __len__(self):
        return len(self.icl_data)

    def __getitem__(self, index):
        entry = deepcopy(self.icl_data[index])

        support_img_list = []
        support_img_path_list = []
        support_raw_text_list = []

        for subentry in entry['support_classes_examples']:
            try:
                img, img_path = self._read_image(subentry)
            except RuntimeError:
                img = None
            
            support_img_list.append(img)
            support_img_path_list.append(img_path)
            support_raw_text_list.append([subentry['caption']] + [subentry['foil']])

        support_foil_img_list = []
        support_foil_img_path_list = []
        support_foil_raw_text_list = []

        for subentry in entry['support_classes_foil_examples']:
            try:
                img, img_path = self._read_image(subentry)
            except RuntimeError:
                img = None
            
            support_foil_img_list.append(img)
            support_foil_img_path_list.append(img_path)
            support_foil_raw_text_list.append([subentry['caption']] + [subentry['foil']]) 

        try:
            query_img, query_img_path = self._read_image(entry['query'])
        except RuntimeError:
            query_img = None

        query_raw_texts = [entry['query']['caption']] + [entry['query']['foil']]

        item = {
            'index': index,
            'item_id': self.icl_data[index]['query']['item_id'],
            'query_image': query_img,
            'query_raw_texts': query_raw_texts,
            'query_image_path': query_img_path,
            'support_classes_image_list': support_img_list,
            'support_classes_raw_texts': support_raw_text_list,
            'support_classes_image_path_list': support_img_path_list,
            'support_classes_foil_image_list': support_foil_img_list,
            'support_classes_foil_raw_texts': support_foil_raw_text_list,
            'support_classes_foil_image_path_list': support_foil_img_path_list,
        }
        return item
