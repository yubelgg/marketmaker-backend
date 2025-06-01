import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from gradio_client import Client

load_dotenv()

app = Flask(__name__)
CORS(app)

MODEL_ID = "yubelgg/marketmaker"
SPACE_URL = "https://huggingface.co/spaces/yubelgg/marketmaker-sentiment"


def call_space_api(text):
    """Call the Hugging Face Space API using the official gradio_client"""
    try:
        client = Client("yubelgg/marketmaker-sentiment")
        result = client.predict(text, api_name="/analyze_text")

        print(f"Raw result: {result}")

        return parse_space_prediction(result, text)

    except Exception as e:
        print(f"gradio client error: {e}")
        return {"error": f"Space API error: {e}"}


def parse_space_prediction(prediction_text, original_text):
    """Parse the actual prediction from the Space"""
    try:
        print(f"Parsing prediction: {prediction_text}")

        if isinstance(prediction_text, str) and "**Predictions:**" in prediction_text:
            lines = prediction_text.split("\n")
            predictions = []

            for line in lines:
                line = line.strip()
                for label_lower in ["positive", "negative", "neutral"]:
                    if f"**{label_lower}**:" in line:
                        score_str = line.split(f"**{label_lower}**:")[1].strip()
                        if "%" in score_str:
                            percentage = score_str.replace("%", "").strip()
                            try:
                                score = float(percentage) / 100.0
                                label_upper = label_lower.upper()
                                predictions.append(
                                    {"label": label_upper, "score": score}
                                )
                                print(f"Parsed {label_upper}: {score}")
                            except ValueError:
                                continue

            if predictions:
                return {
                    "text": original_text,
                    "predictions": predictions,
                    "model": MODEL_ID,
                    "source": "huggingface_space",
                }

        return {"error": f"Could not parse prediction format: {prediction_text}"}

    except Exception as e:
        return {"error": f"Parse error: {e}"}


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "healthy", "model": MODEL_ID, "space_url": SPACE_URL})


@app.route("/api/analyze", methods=["POST"])
def analyze_sentiment():
    """Analyze sentiment using Hugging Face Space"""
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"error": "No text provided"}), 400

        text = data["text"].strip()
        if not text:
            return jsonify({"error": "Empty text provided"}), 400

        result = call_space_api(text)

        if "error" in result:
            return jsonify(result), 500

        return jsonify(result)

    except Exception as e:
        print(f"Server error: {str(e)}")
        return jsonify({"error": f"Server error: {str(e)}"}), 500


@app.route("/api/space-info", methods=["GET"])
def space_info():
    """Get information about the deployed Space"""
    return jsonify(
        {
            "space_url": SPACE_URL,
            "model": MODEL_ID,
            "status": "live",
            "description": "Your MarketMaker text classifier is deployed on Hugging Face Spaces",
            "features": [
                "Text classification for market sentiment",
                "Real-time predictions",
            ],
            "how_to_use": [
                f"1. Visit {SPACE_URL}",
                "2. Enter your text in the input box",
                "3. Click 'Analyze Text'",
                "4. View your classification results",
            ],
        }
    )


@app.route("/", methods=["GET"])
def home():
    """Home endpoint with Space information"""
    return jsonify(
        {
            "message": "MarketMaker API - powered by Hugging Face Spaces!",
            "space_url": SPACE_URL,
            "endpoints": {
                "/api/health": "Health check",
                "/api/analyze": "Analyze sentiment using gradio_client",
                "/api/space-info": "Space information",
            },
        }
    )


if __name__ == "__main__":
    print(f"MarketMaker API")
    print(f"Model: {MODEL_ID}")
    print(f"Space URL: {SPACE_URL}")
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
