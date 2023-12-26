import os.path as osp
import torch
import json


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