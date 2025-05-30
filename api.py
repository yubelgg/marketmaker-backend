import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app) 

# Configure Hugging Face model ID
MODEL_ID = "yubelgg/marketmaker"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    """Load the model from Hugging Face Hub"""
    try:
        print(f"Loading model {MODEL_ID} from Hugging Face Hub...")
        # Get token from environment variable (optional for public models)
        token = os.environ.get("HUGGINGFACE_TOKEN")
        
        if token:
            print("Using authentication token...")
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, token=token)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
        else:
            print("No token found, attempting to load public model...")
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
            
        model = model.to(device)
        model.eval()
        print(f"Model loaded and running on {device}")
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e

# Load model at startup
model, tokenizer = load_model()

@app.route('/api/analyze', methods=['POST'])
def analyze_sentiment():
    # Get the text from the request
    data = request.json
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400
    
    text = data['text']
    
    try:
        # Tokenize
        inputs = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256
        )
        
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get prediction
        with torch.no_grad():
            outputs = model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(predictions, dim=1).item()
        
        # Map class to sentiment
        sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
        confidence = predictions[0][predicted_class].item()
        
        # Format the response
        result = {
            'text': text,
            'sentiment': sentiment_map[predicted_class],
            'confidence': float(confidence),
            'probabilities': {
                'negative': float(predictions[0][0]),
                'neutral': float(predictions[0][1]), 
                'positive': float(predictions[0][2])
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Internal server error during prediction'}), 500

# Add a health check endpoint
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model': MODEL_ID,
        'device': str(device)
    }), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 