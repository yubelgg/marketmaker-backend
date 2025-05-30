import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from huggingface_hub import login

print("Loading local model...")
# Load your local model
model = AutoModelForSequenceClassification.from_pretrained("finbert_wsb_model")
tokenizer = AutoTokenizer.from_pretrained("finbert_wsb_model")

print("Model loaded successfully!")

# Set up repository info
repository_id = "yubelgg/marketmaker"

print(f"Pushing model to {repository_id}...")

try:
    # Push model and tokenizer
    model.push_to_hub(repository_id)
    tokenizer.push_to_hub(repository_id)
    
    print(f"✅ Model and tokenizer successfully pushed to {repository_id}")
    print(f"🔗 You can view your model at: https://huggingface.co/{repository_id}")
    
except Exception as e:
    print(f"❌ Error during upload: {e}")
    print("Please make sure:")
    print("1. You're logged in to Hugging Face CLI")
    print("2. The repository exists or you have permission to create it")
    print("3. Your internet connection is stable") 