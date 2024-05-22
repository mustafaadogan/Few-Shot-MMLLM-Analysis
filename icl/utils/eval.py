import json
import torch
from torchmetrics.functional.classification import multiclass_accuracy, confusion_matrix, auroc
from .util import process_path

def format_score(val):
    return round(100 * val.item() if isinstance(val, torch.Tensor) else 100 * val, 2)

def process_scores(input_file, mode):
    """
    Process scores based on the specified mode.

    Parameters:
    - input_file (str): Path to the input file.
    - mode (str): The processing mode.

    Returns:
    dict: Results based on the processing mode.
    """
    input_file = process_path(input_file)
    with open(input_file, 'r') as f:
        pred = json.load(f)

    num_texts = max(len(item['scores']) for item in pred.values())
    num_examples = len(pred)
    scores = torch.zeros((num_examples, num_texts), dtype=torch.double).fill_(-torch.inf if mode == 'perplexity' else 0)

    for idx, item in enumerate(pred.values()):
        item_scores = torch.tensor(item['scores'])
        item_num_texts = item_scores.numel()

        if mode == 'perplexity':
            item_scores = -item_scores

        scores[idx, :item_num_texts] = item_scores

    results = {}

    labels = torch.zeros(num_examples, dtype=torch.long)
    acc_r = multiclass_accuracy(scores, labels, num_classes=num_texts, average='micro')
    results['acc_r'] = format_score(acc_r)

    caption_probs = scores[:, 0]
    foil_probs = scores[:, 1:].flatten()
    probs = torch.cat([caption_probs, foil_probs])
    labels = torch.zeros(scores.numel(), dtype=torch.long)
    labels[:num_examples] = 1

    mat = confusion_matrix(probs, labels, task='binary')
    TP, FP, TN, FN = mat[1, 1], mat[0, 1], mat[0, 0], mat[1, 0]

    p_c = PPV = format_score(TP / (TP + FP))
    p_f = NPV = format_score(TN / (TN + FN))
    acc = format_score((TP + TN) / (TP + TN + FP + FN))
    auroc_val = format_score(auroc(probs, labels, task='binary'))

    results['p_c'] = p_c
    results['p_f'] = p_f
    results['min(p_c, p_f)'] = min(p_c, p_f)
    results['acc'] = acc
    results['auroc'] = auroc_val

    return results