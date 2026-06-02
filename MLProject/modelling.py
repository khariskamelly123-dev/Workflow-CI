import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns

# ============ SETUP MLFLOW ============
# Set tracking URI ke local directory di dalam folder MLProject
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MLFLOW_DIR = os.path.join(BASE_DIR, "mlruns")
os.makedirs(MLFLOW_DIR, exist_ok=True)
mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")

print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Working directory: {BASE_DIR}")
# =======================================

# Membuat folder artifacts
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Load Dataset (pastikan file ada di folder yang sama)
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

# Set experiment
experiment_name = "Heart Disease Training"
try:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
        print(f"Created new experiment: {experiment_name} with ID: {experiment_id}")
    else:
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment: {experiment_name} with ID: {experiment_id}")
    
    mlflow.set_experiment(experiment_name)
except Exception as e:
    print(f"Error setting experiment: {e}")
    mlflow.set_experiment(experiment_name)

# Training model
with mlflow.start_run() as run:
    print(f"Run ID: {run.info.run_id}")
    print(f"Experiment ID: {run.info.experiment_id}")
    
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    # Log parameter
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)
    
    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    # Evaluasi
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    
    # Log metric
    mlflow.log_metric("accuracy", accuracy)

    # Simpan model
    model_path = os.path.join(ARTIFACT_DIR, "best_model.pkl")
    joblib.dump(model, model_path)

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

    # Classification Report
    report = classification_report(y_test, y_pred)

    report_path = os.path.join(BASE_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    # Log artifact ke MLflow
    mlflow.log_artifact(cm_path)
    mlflow.log_artifact(report_path)
    mlflow.log_artifact(model_path)
    
    # Log model menggunakan MLflow
    mlflow.sklearn.log_model(model, "model")

print("Training selesai!")