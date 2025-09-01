import os
import joblib
from flask import Flask, request, jsonify
import numpy as np

# Config
MODEL_PATH = os.getenv('MODEL_PATH', 'model/iris_model.pkl')

# APP
app = Flask(__name__)

# Load once at startup
try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    # Fail fast with a helpful error message
    raise RuntimeError(f"Failed to load model from {MODEL_PATH}: {e}")

@app.get('/')
def index():
    return "✅ Flask Iris Prediction API is running! ", 200

@app.get('/health')
def health():
    return jsonify(status='ok'), 200

@app.post('/predict')
def predict():
    """Accepts either:
    {"input": [[...feature vectors...]]} or
    {"input": [1.0, 2.0, 3.0, 4.0]} (single vector)
    """
    try:
        payload = request.get_json(force=True)
        x = payload.get('input')

        if x is None:
            return jsonify(error="Missing 'input' in request"), 400

        # Normalize input to 2D array
        if isinstance(x, list) and all(isinstance(i, (int, float)) for i in x):
            x = [x]  # Wrap single feature vector

        x = np.array(x, dtype=float)

        preds = model.predict(x).tolist()
        return jsonify(predictions=preds), 200

    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
