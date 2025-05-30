import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
from tqdm import tqdm
import accelerate
from accelerate import Accelerator
import gc
import psutil
import time

# Configuration
RANDOM_SEED = 42
DATASET_PATH = "reddit_wsb_with_sentiments.csv"
MODEL_NAME = "ProsusAI/finbert"
OUTPUT_DIR = "finbert_wsb_model"
MAX_LENGTH = 256
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 16
LEARNING_RATE = 1e-5
NUM_EPOCHS = 2
VALIDATION_SPLIT = 0.2
WARMUP_STEPS = 500
WEIGHT_DECAY = 0.01

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize accelerator
accelerator = Accelerator()
print(f"Using accelerator: {accelerator.device}")

# Load the processed dataset
print(f"Loading dataset from {DATASET_PATH}...")
df = pd.read_csv(DATASET_PATH)

# Verify the loaded data
print(f"Loaded {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")
print("Sentiment label distribution:")
print(df["sentiment_label_encoded"].value_counts())


# Create a custom dataset class
class WallStreetBetsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # Return as tensors
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


# Function to compute metrics for evaluation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted"
    )
    acc = accuracy_score(labels, preds)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def main():
    print("Starting FinBERT fine-tuning process...")

    # Check transformers version
    import transformers

    print(f"Transformers version: {transformers.__version__}")

    # Load the pre-trained model and tokenizer
    print(f"Loading pre-trained FinBERT model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=3)

    # Explicitly move model to device
    model = model.to(device)
    print(f"Is model on correct device? {next(model.parameters()).device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Create directory for the fine-tuned model
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Extract texts and labels
    texts = df["full_text"].values
    labels = df["sentiment_label_encoded"].values

    # Create and split the dataset
    dataset = WallStreetBetsDataset(texts, labels, tokenizer, MAX_LENGTH)
    train_size = int((1 - VALIDATION_SPLIT) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=EVAL_BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        weight_decay=WEIGHT_DECAY,
        logging_dir=f"{OUTPUT_DIR}/logs",
        logging_steps=50,
        save_total_limit=2,
        learning_rate=LEARNING_RATE,
        gradient_accumulation_steps=4,
        # Enable fp16 if using GPU
        fp16=torch.cuda.is_available(),
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    print("\nStarting training...")
    trainer.train()

    # Evaluate the model
    print("\nEvaluating the model...")
    eval_results = trainer.evaluate()
    print(f"Evaluation results: {eval_results}")

    # Save the fine-tuned model and tokenizer
    print(f"\nSaving the model to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Model training and saving complete!")


# Function to test inference with a sample post
def test_inference(model_dir):
    print("\nTesting model inference with sample posts...")

    # Load the fine-tuned model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model = model.to(device)  # Move model to GPU
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model.eval()

    # Sample posts to test
    sample_posts = [
        "I am very bullish on $GME! 🚀🚀🚀",
        "$TSLA is overvalued, I'm shorting it.",
        "Not sure about $AAPL, could go either way.",
    ]

    for post in sample_posts:
        # Tokenize the post
        inputs = tokenizer(
            post,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
        )

        # Move inputs to same device as model
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=1).item()

        # Map the predicted class back to sentiment
        sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
        sentiment = sentiment_map[predicted_class]
        confidence = predictions[0][predicted_class].item()

        print(f"\nPost: {post}")
        print(f"Sentiment: {sentiment} (Confidence: {confidence:.2f})")
        print(f"All probabilities: {predictions[0].tolist()}")


if __name__ == "__main__":
    main()

    # Free up GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    # Test the saved model with a few examples
    test_inference(OUTPUT_DIR)
