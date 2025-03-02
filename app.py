import os
import pdfplumber  # Alternative to fitz for PDF handling
import pandas as pd
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask import Flask, render_template, request, jsonify

# Download NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Load the trained model and vectorizer
with open("model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Initialize Flask app
app = Flask(__name__)

# Preprocessing function
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Convert to lowercase
    text = text.lower()
    # Remove special characters & numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize and remove stopwords
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return " ".join(words)

# Prediction function
def predict_fake_news(text):
    processed_text = preprocess_text(text)
    vectorized_text = vectorizer.transform([processed_text])
    prediction = model.predict(vectorized_text)
    return "Fake News" if prediction == 1 else "Real News"

# Route for homepage
@app.route("/")
def home():
    return render_template("index.html")

# Route to handle text input
@app.route("/predict", methods=["POST"])
def predict():
    text = request.form["news_text"]
    prediction = predict_fake_news(text)
    return jsonify({"prediction": prediction})

# Route to handle file uploads
@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Read text from uploaded PDF file
    if file.filename.endswith(".pdf"):
        with pdfplumber.open(file) as pdf:
            text = "".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    else:
        return jsonify({"error": "Unsupported file format"}), 400

    if not text.strip():
        return jsonify({"error": "Empty file"}), 400

    prediction = predict_fake_news(text)
    return jsonify({"prediction": prediction})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
