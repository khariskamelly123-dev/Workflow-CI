import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# Lokasi folder modelling.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Membuat folder artifacts
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Load Dataset
DATA_PATH = os.path.join(BASE_DIR, "heart_processed.csv")

# Cek apakah file dataset ada
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset tidak ditemukan di: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded: {df.shape}")

# Pisahkan fitur dan target
X = df.drop("target", axis=1)
y = df["target"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training data: {X_train.shape}, Test data: {X_test.shape}")

# Training model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Prediksi
y_pred = model.predict(X_test)

# Evaluasi
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Simpan model
model_path = os.path.join(ARTIFACT_DIR, "best_model.pkl")
joblib.dump(model, model_path)
print(f"Model saved to {model_path}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

cm_path = os.path.join(BASE_DIR, "confusion_matrix.png")
plt.savefig(cm_path, bbox_inches="tight")
plt.close()
print(f"Confusion matrix saved to {cm_path}")

# Classification Report
report = classification_report(y_test, y_pred)

report_path = os.path.join(BASE_DIR, "classification_report.txt")
with open(report_path, "w") as f:
    f.write(report)
print(f"Classification report saved to {report_path}")

print("Training selesai!")