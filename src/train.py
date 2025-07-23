import logging
import sys
import argparse
import subprocess
import os

from transformers import AutoTokenizer,AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
import torch
import numpy as np
import boto3
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import pandas as pd
from datasets import Dataset
import evaluate


def install_dependencies():
    """
    Install required Python dependencies.
    Alternatively, consider using a pre-built container with these dependencies pre-installed.
    """
    try:
        print("Installing required dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "evaluate", "datasets"])
    except subprocess.CalledProcessError as e:
        print(f"Failed to install dependencies: {e}")
        sys.exit(1)

def load_dataset(args):
    "loads dataset from s3 bucket"
    data = pd.read_csv(os.path.join(args.train, "cleaned_train.csv"))
    
    #convert the dataset to Hugging Face Dataset format
    #split data into train and validation
    dataset = Dataset.from_pandas(data)
    dataset = dataset.train_test_split(test_size=0.3)
    return dataset

def tokenization(dataset, args):
    #fetch the tokenizer
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    
    #tokenize dataset
    def tokenize(batch):
        temp = tokenizer(batch['meme_text'], padding=True, truncation=True, max_length=300)
        return temp
    tokenized_dataset = dataset.map(tokenize, batched=True, batch_size=None)
    return tokenized_dataset, tokenizer

def train_model(args):
    #install dependencies
    install_dependencies()

    # Set up logging
    logger = logging.getLogger(__name__)

    logging.basicConfig(
        level=logging.getLevelName("INFO"),
        handlers=[logging.StreamHandler(sys.stdout)],
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Loading the training dataset ........")
    feature_dataset = load_dataset(args)
    logger.info("Completed the loading of the the training dataset ........")
    
    #convert the sentiments to integer labels
    label2id={'0': 0, '1': 1}
    id2label = {0: 'not-toxic', 1: 'toxic'}
    dataset = feature_dataset.map(lambda x: {'label': label2id[str(x['y'])]})

    logger.info("Tokenizing  the training dataset ........")
    tokenized_dataset, tokenizer = tokenization(dataset, args)
    logger.info("Completed the tokenization of  the training dataset ........")

    logger.info("Setting up model training ........")
    #load the model
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2, label2id=label2id, id2label=id2label)

    #training arguments
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        learning_rate=float(args.learning_rate),
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        evaluation_strategy='epoch',
    )
 
    #compute accuracy
    accuracy = evaluate.load("accuracy")
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)
        
    #finetune model
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train'],
        eval_dataset=tokenized_dataset['test'],
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    #Start training
    logger.info("Starting model training ........")
    trainer.train()

    logger.info("Starting model evaluation ........")
    trainer.evaluate()


    logger.info("Saving trained model to s3 bucket ........")
    # Saves the model and tokenizer to s3
    trainer.save_model(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)



if __name__ == "__main__":

    # SageMaker passes hyperparameters  as command-line arguments to the script
    # Parsing them below...
    parser = argparse.ArgumentParser()

    #hyperparameter arguments
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--learning_rate", type=str, default=2e-5)

    # Data, model, and output directories

    parser.add_argument("--train", type=str, default=os.environ["SM_CHANNEL_TRAIN"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"])

    args, _ = parser.parse_known_args()

    train_model(args)