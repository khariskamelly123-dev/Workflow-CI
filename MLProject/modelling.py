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

# Lokasi folder modelling.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Membuat folder artifacts
ARTIFACT_DIR = os.path.join(BASE_DIR, "artifacts")
os.makedirs(ARTIFACT_DIR, exist_ok=True)

# Load Dataset
DATA_PATH = os.path.join(
    BASE_DIR,
    "heart_processed.csv"
)

df = pd.read_csv(DATA_PATH)

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

# Nama experiment MLflow
mlflow.set_experiment("Heart Disease Training")

# Aktifkan autolog
mlflow.sklearn.autolog()

# Training model
with mlflow.start_run():

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Prediksi
    y_pred = model.predict(X_test)

    # Evaluasi
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print(f"Accuracy: {accuracy:.4f}")

    # Simpan model
    model_path = os.path.join(
        ARTIFACT_DIR,
        "best_model.pkl"
    )

    joblib.dump(
        model,
        model_path
    )

    # Confusion Matrix
    cm = confusion_matrix(
        y_test,
        y_pred
    )

    plt.figure(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d"
    )

    plt.title("Confusion Matrix")

    cm_path = os.path.join(
        BASE_DIR,
        "confusion_matrix.png"
    )

    plt.savefig(
        cm_path,
        bbox_inches="tight"
    )

    plt.close()

    # Classification Report
    report = classification_report(
        y_test,
        y_pred
    )

    report_path = os.path.join(
        BASE_DIR,
        "classification_report.txt"
    )

    with open(
        report_path,
        "w"
    ) as f:
        f.write(report)

    # Log artifact ke MLflow
    mlflow.log_artifact(cm_path)

    mlflow.log_artifact(report_path)

    mlflow.log_artifact(model_path)

print("Training selesai.")