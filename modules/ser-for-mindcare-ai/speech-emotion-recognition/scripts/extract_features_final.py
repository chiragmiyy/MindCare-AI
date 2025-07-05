import os
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# Paths
AUDIO_DIR = "normalized_data"
FEATURES_CSV = "features/full_features.csv"

# Feature extractor
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_delta = librosa.feature.delta(mfcc)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        zcr = librosa.feature.zero_crossing_rate(y)
        rmse = librosa.feature.rms(y=y)

        features = np.hstack([
            np.mean(mfcc, axis=1),
            np.mean(mfcc_delta, axis=1),
            np.mean(chroma, axis=1),
            np.mean(spec_cent),
            np.mean(spec_bw),
            np.mean(zcr),
            np.mean(rmse)
        ])

        return features
    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return None

def get_emotion_from_filename(path):
    path = path.lower()
    for emo in ['angry', 'calm', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']:
        if emo in path:
            return emo
    return "unknown"

# Feature collection
data = []

for root, _, files in os.walk(AUDIO_DIR):
    for file in tqdm(files, desc="Extracting features"):
        if file.endswith(".wav"):
            path = os.path.join(root, file)
            emotion = get_emotion_from_filename(path)
            features = extract_features(path)
            if features is not None and emotion != "unknown":
                row = list(features) + [emotion]
                data.append(row)

# Save
columns = [f"mfcc_{i}" for i in range(13)] + \
          [f"delta_{i}" for i in range(13)] + \
          [f"chroma_{i}" for i in range(12)] + \
          ["spec_cent", "spec_bw", "zcr", "rmse", "emotion"]

df = pd.DataFrame(data, columns=columns)
os.makedirs("features", exist_ok=True)
df.to_csv(FEATURES_CSV, index=False)
logging.info(f"\n✅ Features saved to {FEATURES_CSV}")