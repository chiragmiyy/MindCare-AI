import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# Load features
df = pd.read_csv("features/full_features.csv")

# Separate features and labels
X = df.drop(columns=["emotion"])
y = df["emotion"]

# Encode labels
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model
print("🎯 Training RandomForestClassifier...")
model = RandomForestClassifier(n_estimators=300, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Accuracy: {acc:.4f}\n")
print("📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save model and scaler
os.makedirs("models", exist_ok=True)
with open("models/final_model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print("💾 Model, Scaler, and LabelEncoder saved to 'models/'")