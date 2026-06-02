import os
import joblib
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

# =========================
# Setup MLflow
# =========================
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Heart Disease Experiment")

# =========================
# Setup Folder
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# =========================
# Load Dataset
# =========================
DATA_PATH = os.path.join(BASE_DIR, "heart_processed.csv")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset tidak ditemukan di: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded: {df.shape}")

# =========================
# Split Feature dan Target
# =========================
X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training data: {X_train.shape}")
print(f"Testing data : {X_test.shape}")

# =========================
# Training dengan MLflow
# =========================
with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")

    # Log parameter
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)

    # Log metric
    mlflow.log_metric("accuracy", accuracy)

    # =========================
    # Simpan Model
    # =========================
    model_path = os.path.join(ARTIFACT_DIR, "best_model.pkl")

    joblib.dump(model, model_path)

    print(f"Model saved to {model_path}")

    mlflow.log_artifact(model_path)

    # =========================
    # Confusion Matrix
    # =========================
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    cm_path = os.path.join(BASE_DIR, "confusion_matrix.png")

    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()

    print(f"Confusion Matrix saved to {cm_path}")

    mlflow.log_artifact(cm_path)

    # =========================
    # Classification Report
    # =========================
    report = classification_report(y_test, y_pred)

    report_path = os.path.join(
        BASE_DIR,
        "classification_report.txt"
    )

    with open(report_path, "w") as f:
        f.write(report)

    print(f"Classification Report saved to {report_path}")

    mlflow.log_artifact(report_path)

print("Training selesai!")