import pandas as pd
import torch
import os
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_card",
        default='affahrizain/roberta-base-finetuned-jigsaw-toxic',
        type=str,
        help="the name of pretrained model card"
    )
    parser.add_argument(
        "--data_path",
        default = './data/jigsaw-unintended-bias-in-toxicity-classification/',
        type=str,
        help="the path of data"
    )

    return parser

def test_classifier_on_jigsaw(args):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_card = args.model_card
    DATA_PATH = args.data_path

    # Load data
    data = pd.read_csv(os.path.join(DATA_PATH, 'test_data.csv'))

    # Convert toxicity to binary
    data['toxicity'] = (data['toxicity'] >= 0.5).astype(int)

    # Define tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_card)

    # Tokenize data
    encodings = tokenizer(data['comment_text'].tolist(), truncation=True, padding=True, return_tensors="pt")

    # Load pre-trained model
    model = AutoModelForSequenceClassification.from_pretrained(model_card).to(device)

    # Make sure model is in evaluation mode
    model.eval()

    batch_size = 128

    # Initialize variables
    predictions = []
    num_batches = len(encodings.input_ids) // batch_size + int(len(encodings.input_ids) % batch_size > 0)

    # Process batches
    for i in tqdm(range(num_batches)):
        # Get batch
        start = i * batch_size
        end = start + batch_size
        input_ids = encodings.input_ids[start:end].to(model.device)
        attention_mask = encodings.attention_mask[start:end].to(model.device)

        # Make prediction
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

        # Get predicted class
        batch_predictions = torch.argmax(logits, dim=-1)
        predictions.extend(batch_predictions.cpu().numpy())


    # Convert predictions to tensor
    predictions = torch.tensor(predictions)

    # assert len(predictions) == len(data['toxicity'].tolist())

    # Calculate accuracy
    accuracy = (predictions == torch.Tensor(data['toxicity'].tolist()[:len(predictions)])).float().mean().item()
    print(f'Accuracy: {accuracy}')

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    test_classifier_on_jigsaw(args)