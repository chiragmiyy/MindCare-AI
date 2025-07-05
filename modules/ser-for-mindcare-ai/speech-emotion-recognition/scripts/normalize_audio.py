import os
import librosa
import soundfile as sf
from tqdm import tqdm

# CONFIG
TARGET_SAMPLE_RATE = 22050
TARGET_DURATION = 3  # seconds
TARGET_LENGTH = TARGET_SAMPLE_RATE * TARGET_DURATION

# 📁 Input and output folders
RAW_DATA_DIR = "data"
OUTPUT_DIR = "normalized_data"

# Make output folders if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

def normalize_and_pad(filepath, output_path):
    try:
        y, sr = librosa.load(filepath, sr=TARGET_SAMPLE_RATE)
        y = librosa.util.normalize(y)

        # Trim silence
        y, _ = librosa.effects.trim(y)

        # Pad or truncate to TARGET_LENGTH
        if len(y) > TARGET_LENGTH:
            y = y[:TARGET_LENGTH]
        else:
            pad_width = TARGET_LENGTH - len(y)
            y = librosa.util.fix_length(y, size=TARGET_LENGTH)

        # Save as WAV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        sf.write(output_path, y, TARGET_SAMPLE_RATE)
    except Exception as e:
        print(f"❌ Error processing {filepath}: {e}")

# 🔁 Loop through all datasets
for dataset in os.listdir(RAW_DATA_DIR):
    dataset_path = os.path.join(RAW_DATA_DIR, dataset)
    for root, _, files in os.walk(dataset_path):
        for file in tqdm(files):
            if file.endswith(".wav"):
                input_file = os.path.join(root, file)
                rel_path = os.path.relpath(input_file, RAW_DATA_DIR)
                output_file = os.path.join(OUTPUT_DIR, rel_path)
                normalize_and_pad(input_file, output_file)

print("\n✅ Audio normalization complete. Check 'normalized_data/' folder.")
