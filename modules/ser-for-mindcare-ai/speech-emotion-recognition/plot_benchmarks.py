import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# === DUMMY DATA (for demo only) ===
emotions = [
    "neutral", "calm", "happy", "sad",
    "angry", "fear", "disgust", "pleasant_surprise", "boredom"
]

# Fake ground truth and predictions (pretend we had 100 samples)
np.random.seed(42)
y_true = np.random.choice(emotions, 100)
y_pred = np.random.choice(emotions, 100)

# === Confusion Matrix Plot ===
def plot_confusion_matrix(y_true, y_pred, labels, save_path="results/confusion_matrix.png"):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Emotion")
    plt.ylabel("True Emotion")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Confusion matrix saved at: {save_path}")
    plt.show()

# === Accuracy Comparison Plot ===
def plot_model_accuracy(model_scores, save_path="results/model_accuracy_comparison.png"):
    models = list(model_scores.keys())
    scores = list(model_scores.values())

    plt.figure(figsize=(10, 6))
    sns.barplot(x=scores, y=models, palette="coolwarm")
    plt.title("Model Accuracy Comparison")
    plt.xlabel("Accuracy (%)")
    plt.xlim(0, 100)
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"✅ Model accuracy comparison saved at: {save_path}")
    plt.show()

# === Example Model Accuracies ===
model_scores = {
    "Random Forest": 84.7,
    "Wav2Vec2": 88.9,
    "MLP Classifier": 81.2,
    "Best Combined Model": 93.96
}

# === Create Results Directory ===
import os
os.makedirs("results", exist_ok=True)

# === Plot Everything ===
plot_confusion_matrix(y_true, y_pred, emotions)
plot_model_accuracy(model_scores)
