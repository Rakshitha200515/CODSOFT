from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import numpy as np
import os

# 👇 Adjusted static folder path
app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app)

# Load model, vectorizer, label encoder
model = joblib.load("movie_genre_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")
label_encoder = joblib.load("label_encoder.pkl")

@app.route("/")
def index():
    # 👇 Make sure to serve index.html
    return send_from_directory(app.static_folder, "index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    plot = data.get("plot", "")
    if not plot:
        return jsonify({"error": "No plot provided"}), 400

    X = vectorizer.transform([plot])
    probs = model.predict_proba(X)[0]
    top_indices = np.argsort(probs)[::-1][:5]
    top_genres = label_encoder.inverse_transform(top_indices)

    result = {
        "genre": top_genres[0],
        "top": [{"genre": g, "prob": float(probs[i])}
                for g, i in zip(top_genres, top_indices)],
    }
    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True)
