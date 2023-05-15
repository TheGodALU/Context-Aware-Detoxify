import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import argparse
import os


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

    parser.add_argument(
        "--output_dir",
        default = './results/',
        type=str,
        help="the path of log directory"
    )

    parser.add_argument(
        "--save_dir",
        default = './models/',
        type=str,
        help="the path of saved model directory"
    )

    return parser

class ToxicDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: (val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def get_preprocessed_data(args):
    data_path = args.data_path

    # Load data
    data = pd.read_csv(os.path.join(data_path, 'all_data.csv'))

    # Split data
    train_data = data[data['split'] == 'train']

    # Drop rows with NaN or null values in 'comment_text' or 'toxicity'
    train_data = train_data.dropna(subset=['comment_text', 'toxicity'])

    # Ensure 'comment_text' column contains only strings
    train_data['comment_text'] = train_data['comment_text'].astype(str)

    # Convert toxicity to binary
    train_data['toxicity'] = (train_data['toxicity'] >= 0.5).astype(int)

    return train_data

def finetune_classifier_on_jigsaw(train_data, args):

    # Define tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_card)

    # Tokenize data
    train_encodings = tokenizer(train_data['comment_text'].tolist(), truncation=True, padding=True)


    # Create datasets
    train_dataset = ToxicDataset(train_encodings, train_data['toxicity'].tolist())

    # Load pre-trained model
    model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,          
        num_train_epochs=3,              
        per_device_train_batch_size=256,  
        per_device_eval_batch_size=64,   
        warmup_steps=500,                
        weight_decay=0.01,               
        logging_dir='./logs',            
    )

    # Define trainer
    trainer = Trainer(
        model=model,                         
        args=training_args,                  
        train_dataset=train_dataset,         
    )

    # Train model
    trainer.train()

    # save pretrained model
    model.save_pretrained(args.save_dir)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()

    # get preprocessed data
    train_data = get_preprocessed_data(args)

    # finetune classifier on jigsaw
    finetune_classifier_on_jigsaw(train_data, args)