import pickle
import numpy as np
from src.features.extract_features import extract_features

# Load model and scaler
model = pickle.load(open("models/best_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))

def predict_emotion_from_audio(audio_path):
    features = extract_features(audio_path)
    features = scaler.transform([features])
    prediction = model.predict(features)
    return prediction[0]