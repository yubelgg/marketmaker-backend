# MarketMaker - Text Classification API Backend

A Flask-based API backend that provides sentiment analysis using a custom-trained text classification model deployed on Hugging Face Spaces.

## Project Overview

MarketMaker is a text classification system designed for market sentiment analysis. This repository contains the backend API that interfaces with our deployed Hugging Face Space to provide real-time sentiment predictions.

### Hugging Face Spaces + Gradio Client
1. **Deployed to Hugging Face Spaces**: Created a Gradio app at `https://huggingface.co/spaces/yubelgg/marketmaker-sentiment`
2. **API Integration**: Used the official `gradio_client` library to programmatically call the Space
3. **Clean Architecture**: Flask backend serves as a clean API interface for frontend applications

## Architecture

```
Frontend (Next.js) ←→ Flask API (Backend) ←→ Hugging Face Space (Model)
    localhost:3000         localhost:5000      yubelgg/marketmaker-sentiment
```

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- pip or conda for package management

### 1. Clone the Repository
```bash
git clone <repository-url>
cd MarketMaker
```

### 2. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Environment Configuration
Create a `.env` file in the root directory:

### 5. Run the Backend
```bash
python api.py
```

The API will be available at `http://localhost:5000`

## API Endpoints

### Health Check
```http
GET /api/health
```
Returns the API status and model information.

### Sentiment Analysis
```http
POST /api/analyze
Content-Type: application/json

{
  "text": "The market is looking very bullish today!"
}
```

**Response:**
```json
{
  "text": "The market is looking very bullish today!",
  "predictions": [
    {"label": "POSITIVE", "score": 0.1191},
    {"label": "NEGATIVE", "score": 0.716},
    {"label": "NEUTRAL", "score": 0.1649}
  ],
  "model": "yubelgg/marketmaker",
  "source": "huggingface_space"
}
```

### Space Information
```http
GET /api/space-info
```
Returns information about the deployed Hugging Face Space.

## Testing the API

### Using curl
```bash
# Health check
curl http://localhost:5000/api/health

# Analyze sentiment
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "The market is looking very bullish today!"}'
```

### Using Python
```python
import requests

# Analyze sentiment
response = requests.post(
    "http://localhost:5000/api/analyze",
    json={"text": "The market is looking very bullish today!"}
)
print(response.json())
```

## Hugging Face Integration

### Current Model Deployment
- **Model**: `yubelgg/marketmaker`
- **Space**: `https://huggingface.co/spaces/yubelgg/marketmaker-sentiment`
- **API Method**: `gradio_client` library

### Key Files in the Space
- `app.py`: Gradio interface application
- `requirements.txt`: Python dependencies
- `README.md`: Space documentation

## How to Update the Model

### Option 1: Update Existing Model on Hugging Face Hub

1. **Prepare your improved model**:
   ```bash
   # Save your new model locally
   model.save_pretrained("./improved_marketmaker_model")
   tokenizer.save_pretrained("./improved_marketmaker_model")
   ```

2. **Push to Hugging Face Hub**:
   ```bash
   # Install huggingface_hub
   pip install huggingface_hub
   
   # Login to Hugging Face
   huggingface-cli login
   
   # Push the updated model
   python -c "
   from huggingface_hub import HfApi
   api = HfApi()
   api.upload_folder(
       folder_path='./improved_marketmaker_model',
       repo_id='yubelgg/marketmaker',
       repo_type='model'
   )
   "
   ```

3. **Space will automatically use the updated model** (no changes needed to the Space code)

### Option 2: Create a New Model Repository

1. **Create new model repository**:
   ```bash
   # Create new repo on Hugging Face Hub
   python -c "
   from huggingface_hub import HfApi
   api = HfApi()
   api.create_repo('yubelgg/marketmaker-v2', repo_type='model')
   "
   ```

2. **Upload the new model**:
   ```bash
   python -c "
   from huggingface_hub import HfApi
   api = HfApi()
   api.upload_folder(
       folder_path='./improved_marketmaker_model',
       repo_id='yubelgg/marketmaker-v2',
       repo_type='model'
   )
   "
   ```

3. **Update the Space to use the new model**:
   - Edit `marketmaker-sentiment/app.py`
   - Change `MODEL_ID = "yubelgg/marketmaker"` to `MODEL_ID = "yubelgg/marketmaker-v2"`
   - Push changes to the Space repository

4. **Update the backend API**:
   - Edit `api.py`
   - Update `MODEL_ID = "yubelgg/marketmaker-v2"`

### Option 3: Create a New Space for A/B Testing

1. **Create a new Space**:
   ```bash
   # Clone the existing space locally
   git clone https://huggingface.co/spaces/yubelgg/marketmaker-sentiment marketmaker-v2-sentiment
   cd marketmaker-v2-sentiment
   
   # Update the model ID in app.py
   sed -i 's/yubelgg\/marketmaker/yubelgg\/marketmaker-v2/g' app.py
   
   # Create new space repository
   python -c "
   from huggingface_hub import HfApi
   api = HfApi()
   api.create_repo('yubelgg/marketmaker-v2-sentiment', repo_type='space', space_sdk='gradio')
   "
   
   # Push to new space
   git remote set-url origin https://huggingface.co/spaces/yubelgg/marketmaker-v2-sentiment
   git push
   ```

2. **Update backend to use new Space**:
   ```python
   # In api.py, update the client initialization
   client = Client("yubelgg/marketmaker-v2-sentiment")
   ```

## Deployment Options

### Current Production Setup 
Your Flask backend is already deployed on Heroku:
```
Frontend → Heroku Backend (yubelgg-marketmaker-api.herokuapp.com) → HF Space
```

**Heroku App:** `yubelgg-marketmaker-api`  
**Dashboard:** https://dashboard.heroku.com/apps/yubelgg-marketmaker-api/deploy/github

### Development Setup
For local development, you can run the Flask backend locally:
```
Frontend (localhost:3000) → Flask API (localhost:5000) → HF Space
```

### Alternative: Direct Frontend → HF Space (Optional)
If you want to simplify further, you could skip the Flask backend entirely:
```
Frontend → HF Space (direct)
```

### Updating Your Heroku Deployment
Since your backend is connected to GitHub, updates are automatic:
1. Push changes to your GitHub repository
2. Heroku automatically deploys the latest version
3. Your production API updates without manual intervention

### Environment Variables on Heroku
If you need to add environment variables:
1. Go to your [Heroku dashboard](https://dashboard.heroku.com/apps/yubelgg-marketmaker-api/settings)
2. Click "Reveal Config Vars"
3. Add any required environment variables

## Project Structure

```
MarketMaker/
├── api.py                          # Main Flask API application
├── requirements.txt                # Python dependencies
├── .env                           # Environment variables (create this)
├── .gitignore                     # Git ignore rules
├── README.md                      # This file
├── marketmaker-sentiment/         # Hugging Face Space files
│   ├── app.py                     # Gradio application
│   ├── requirements.txt           # Space dependencies
│   └── README.md                  # Space documentation
└── frontend/                      # Frontend application (separate repo)
    └── ...
```

## Technical Details

### Dependencies
- **Flask**: Web framework for the API
- **flask-cors**: Cross-origin resource sharing
- **gradio_client**: Official client for calling Gradio apps
- **python-dotenv**: Environment variable management

### Model Information
- **Base Model**: BERT-based transformer (109M parameters)
- **Task**: Text classification
- **Labels**: POSITIVE, NEGATIVE, NEUTRAL
- **Training Data**: Market-related text data
