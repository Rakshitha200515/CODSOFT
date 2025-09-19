# app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os

app = Flask(__name__)
CORS(app)  # allow frontend (React) to access

MODEL_PATH = 'model/churn_model.joblib'

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Please run train.py first!")

# Load the trained model
model = joblib.load(MODEL_PATH)

# Features used in training
FEATURES = ['CreditScore','Geography','Gender','Age','Tenure','Balance',
            'NumOfProducts','HasCrCard','IsActiveMember','EstimatedSalary']

@app.route('/')
def index():
    return "Churn Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if data is None:
        return jsonify({'error': 'No JSON data provided'}), 400

    # Accept single record or list of records
    if isinstance(data, dict) and 'instances' in data:
        records = data['instances']
    elif isinstance(data, dict) and all(f in data for f in FEATURES):
        records = [data]
    elif isinstance(data, list):
        records = data
    else:
        return jsonify({'error': 'Bad input format. Send features as dict or {"instances":[...]}'})

    df = pd.DataFrame(records)
    
    # Check for missing features
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        return jsonify({'error': 'Missing features', 'missing': missing}), 400

    # Ensure numeric features are numbers
    for col in ['CreditScore','Age','Tenure','Balance','NumOfProducts','EstimatedSalary','HasCrCard','IsActiveMember']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Predict
    preds = model.predict(df[FEATURES])
    probs = model.predict_proba(df[FEATURES])[:,1]

    result = [{'churn': int(p), 'churn_prob': float(prob)} for p, prob in zip(preds, probs)]
    return jsonify({'predictions': result})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
