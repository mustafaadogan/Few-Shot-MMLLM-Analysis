import json
from .util import process_path

def format_score(val):
    """Format the score as a percentage with two decimal places."""
    return round(100 * val, 2)

def process_scores(input_file):
    """
    Process scores from the input file and calculate accuracy.

    Parameters:
    - input_file (str): Path to the input file.

    Returns:
    dict: Results with the calculated accuracy.
    """
    input_file = process_path(input_file)
    with open(input_file, 'r') as f:
        pred = json.load(f)

    num_examples = len(pred)
    positive_count = sum(1 for item in pred.values() if item['scores'][0] == 1)

    accuracy = positive_count / num_examples
    formatted_accuracy = format_score(accuracy)

    return {'accuracy': formatted_accuracy}