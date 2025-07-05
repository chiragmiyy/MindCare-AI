# streamlit_app.py

import streamlit as st
import pickle
import tempfile
import soundfile as sf
from src.features import extract_features

# Load model components
model = pickle.load(open("models/final_model.pkl", "rb"))
scaler = pickle.load(open("models/scaler.pkl", "rb"))
label_encoder = pickle.load(open("models/label_encoder.pkl", "rb"))

st.title("🎙️ Real-Time Speech Emotion Recognition")
st.markdown("Speak into your microphone or upload a recording and let the model predict your emotion.")

# File upload (recorded .wav)
uploaded_file = st.file_uploader("Upload a WAV file or record audio:", type=["wav"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(uploaded_file.read())
        tmpfile_path = tmpfile.name

    st.audio(tmpfile_path, format="audio/wav")

    try:
        features = extract_features(tmpfile_path)
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)[0]
        predicted_emotion = label_encoder.inverse_transform([prediction])[0]
        st.success(f"🎯 Predicted Emotion: **{predicted_emotion}**")
    except Exception as e:
        st.error(f"❌ Error processing audio: {e}")
