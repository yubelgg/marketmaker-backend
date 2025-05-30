import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app)  # Enable CORS to allow requests from your frontend

# Configure Hugging Face model ID and API
MODEL_ID = "yubelgg/marketmaker"
HUGGINGFACE_API_URL = f"https://api-inference.huggingface.co/models/{MODEL_ID}"

def query_huggingface_api(text):
    """Query Hugging Face Inference API"""
    headers = {"Authorization": f"Bearer {os.environ.get('HUGGINGFACE_TOKEN')}"}
    data = {"inputs": text}
    
    response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=data)
    return response.json()

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model": MODEL_ID})

@app.route('/api/analyze', methods=['POST'])
def analyze_sentiment():
    """Analyze sentiment of the provided text"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "No text provided"}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({"error": "Empty text provided"}), 400
        
        # Query Hugging Face Inference API
        result = query_huggingface_api(text)
        
        # Handle API errors
        if 'error' in result:
            return jsonify({"error": f"Model API error: {result['error']}"}), 500
        
        # Process results
        predictions = result[0] if isinstance(result, list) and len(result) > 0 else result
        
        # Format response
        response = {
            "text": text,
            "predictions": predictions,
            "model": MODEL_ID
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    token = os.environ.get('HUGGINGFACE_TOKEN')
    if not token:
        print("⚠️  Warning: HUGGINGFACE_TOKEN not set!")
    else:
        print("✅ Hugging Face token configured")
    
    print(f"🤖 Using model: {MODEL_ID}")
    print(f"🌐 API URL: {HUGGINGFACE_API_URL}")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))