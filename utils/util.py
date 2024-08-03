import os.path as osp
import os
import torch
import json
import random
from itertools import permutations
import hashlib

SEED_NUMBER = 42
random.seed(SEED_NUMBER)

def get_random_index_list(sample_list, count):
   return get_random_sample(list(permutations(sample_list)), count)

def hash_list(lst):
    # Convert the list to a string
    list_str = str(lst)
    # Hash the string representation of the list
    hashed = hashlib.sha256(list_str.encode()).hexdigest()
    # Convert the hash value to an integer
    hash_int = int(hashed, 16)
    return hash_int

def hash_dict(d):
    # Convert the dictionary to a JSON string
    json_str = json.dumps(d, sort_keys=True)
    # Hash the JSON string
    hashed = hashlib.sha256(json_str.encode()).hexdigest()
    # Convert the hash value to an integer
    hash_int = int(hashed, 16)
    return hash_int

def process_path(path):
    if path == None:
        return ""

    path = osp.expanduser(path)
    path = osp.abspath(path)
    return path

def get_device():
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return torch.device(device)

def write_results(file_name, results):
    file_path = process_path(file_name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4)

def get_random_number(start, stop, seed):
    random.seed(seed)
    return random.randint(start, stop)

def get_random_sample(sample_list, count):
    random.seed(hash_list(sample_list))
    return random.sample(sample_list, count)

def set_seed(seed):
    global SEED_NUMBER
    print(f"Setting seed to {seed}!")
    SEED_NUMBER = seed
    random.seed(SEED_NUMBER)

def check_answer(answer, caption_order):
    yes_terms = ['yes', 'Yes', 'YES', 'true', 'True', 'TRUE']
    no_terms = ['no', 'No', 'NO', 'false', 'False', 'FALSE']

    contains_yes = any(term in answer for term in yes_terms)
    contains_no = any(term in answer for term in no_terms)

    if caption_order:
        if contains_yes and not contains_no:
            return [1, 0]
        else:
            return [0, 1]
    else:
        if contains_no and not contains_yes:
            return [1, 0]
        else:
            return [0, 1]
