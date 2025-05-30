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
    
    try:
        response = requests.post(HUGGINGFACE_API_URL, headers=headers, json=data, timeout=30)
        print(f"HF API Response Status: {response.status_code}")
        print(f"HF API Response Text: {response.text}")
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"API returned status {response.status_code}: {response.text}"}
    except Exception as e:
        print(f"Error calling HF API: {e}")
        return {"error": f"Request failed: {str(e)}"}

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    token_status = "Set" if os.environ.get('HUGGINGFACE_TOKEN') else "Missing"
    return jsonify({
        "status": "healthy", 
        "model": MODEL_ID,
        "api_url": HUGGINGFACE_API_URL,
        "token_status": token_status
    })

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
        
        print(f"Analyzing text: {text}")
        
        # Query Hugging Face Inference API
        result = query_huggingface_api(text)
        
        # Handle API errors
        if 'error' in result:
            print(f"HF API Error: {result['error']}")
            return jsonify({"error": f"Model API error: {result['error']}"}), 500
        
        print(f"HF API Result: {result}")
        
        # Format response - the HF API returns classification results
        response = {
            "text": text,
            "predictions": result,
            "model": MODEL_ID
        }
        
        return jsonify(response)
    
    except Exception as e:
        print(f"Server error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500

if __name__ == '__main__':
    token = os.environ.get('HUGGINGFACE_TOKEN')
    if not token:
        print("Warning: HUGGINGFACE_TOKEN not set!")
    else:
        print("Hugging Face token configured")
    
    print(f"Using model: {MODEL_ID}")
    print(f"API URL: {HUGGINGFACE_API_URL}")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))