# src/features.py

import librosa
import numpy as np

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)

    # 40 MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)

    # Zero Crossing Rate (1 feature)
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)

    # Root Mean Square Energy (1 feature)
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)

    # Combine all into 42 features
    combined = np.hstack([mfccs_mean, zcr_mean, rms_mean])

    return combined.reshape(1, -1)