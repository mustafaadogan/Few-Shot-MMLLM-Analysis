import os.path as osp
import torch
import json
import random

random.seed(43)

def process_path(path):
    path = osp.expanduser(path)
    path = osp.abspath(path)
    return path

def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return torch.device(device)

def write_results(file_name, results):
    with open(process_path(file_name), 'w') as f:
        json.dump(results, f, indent=4)

def get_random_number(start, stop):
    return random.randint(start, stop)

def get_random_sample(sample_list, count):
    return random.sample(sample_list, count)

def set_seed(seed):
    print(f"Setting seed to {seed}!")
    random.seed(seed)

def check_answer(answer, caption_order):
    caption_list = ["caption 0", "Caption 0", "Final Answer: caption 0", "Final Answer: Caption 0"]
    foil_list = ["caption 1", "Caption 1", "Final Answer: caption 1", "Final Answer: Caption 1"]
    generic_check = ["caption 0/caption 1", "Caption 0/Caption 1", "caption 1/caption 0", "Caption 1/Caption 0"]
    if caption_order == 1:
        caption_list, foil_list = foil_list, caption_list

    caption_check = any(substring in answer for substring in caption_list)
    foil_check = any(substring in answer for substring in foil_list)
    generic_check = all(substring not in answer for substring in generic_check)

    if generic_check:
        if caption_check and not foil_check:
            return [1, 0]
        elif not caption_check and foil_check:
            return [0, 1]
        else:
            return [-1, -1]
    else:
        return [-1, -1]