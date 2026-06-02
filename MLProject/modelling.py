import os
import joblib
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import matplotlib.pyplot as plt
import seaborn as sns

# ============ SETUP MLFLOW ============
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MLFLOW_DIR = os.path.join(BASE_DIR, "mlruns")
os.makedirs(MLFLOW_DIR, exist_ok=True)
mlflow.set_tracking_uri(f"file://{MLFLOW_DIR}")

print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
print(f"Working directory: {BASE_DIR}")
# =======================================

ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

DATA_PATH = os.path.join(BASE_DIR, "heart_processed.csv")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset tidak ditemukan di: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print(f"Dataset loaded: {df.shape}")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Training data: {X_train.shape}, Test data: {X_test.shape}")

# ============ EXPERIMENT / RUN SETUP ============
# When invoked via `mlflow run .`, MLFLOW_RUN_ID is already set.
# In that case, do NOT call set_experiment() — the run already belongs
# to the correct experiment. Only set the experiment when running standalone.
active_run_id = os.environ.get("MLFLOW_RUN_ID")

if not active_run_id:
    mlflow.set_experiment("Heart Disease Training")

with mlflow.start_run(run_id=active_run_id) as run:
    print(f"Run ID: {run.info.run_id}")
    print(f"Experiment ID: {run.info.experiment_id}")

    model = RandomForestClassifier(n_estimators=100, random_state=42)

    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("random_state", 42)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    mlflow.log_metric("accuracy", accuracy)

    model_path = os.path.join(ARTIFACT_DIR, "best_model.pkl")
    joblib.dump(model, model_path)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_path = os.path.join(BASE_DIR, "confusion_matrix.png")
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()

    report = classification_report(y_test, y_pred)
    report_path = os.path.join(BASE_DIR, "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    mlflow.log_artifact(cm_path)
    mlflow.log_artifact(report_path)
    mlflow.log_artifact(model_path)
    mlflow.sklearn.log_model(model, "model")

print("Training selesai!")