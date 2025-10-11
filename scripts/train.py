import os
import json
import time
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score


def load_data(embeddings_path: str):
    """Load precomputed embeddings and labels from .npy files."""
    print("[INFO] Loading embeddings...")
    X = np.load(os.path.join(embeddings_path, "embeddings.npy"))
    y = np.load(os.path.join(embeddings_path, "labels.npy"))
    print(f"[INFO] Loaded dataset: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    """Split into train/validation sets."""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"[INFO] Split: {len(X_train)} train / {len(X_val)} val samples")
    return X_train, X_val, y_train, y_val


def scale_data(X_train, X_val):
    """Standardize features using StandardScaler."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    print("[INFO] Features standardized.")
    return X_train_scaled, X_val_scaled, scaler


def train_classifier(X_train, y_train):
    """Train a Logistic Regression classifier."""
    print("[INFO] Training LogisticRegression model...")
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    print("[INFO] Model training complete.")
    return model


def evaluate_model(model, X_val, y_val):
    """Compute accuracy and AUC on the validation set."""
    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_prob)

    metrics = {
        "accuracy": round(acc, 4),
        "auc": round(auc, 4),
        "val_samples": len(y_val),
    }

    print(f"[INFO] Validation Accuracy: {acc:.4f}")
    print(f"[INFO] Validation AUC: {auc:.4f}")
    return metrics


def save_artifacts(model, scaler, metrics, output_dir: str):
    """Save trained model, scaler, and metrics to disk."""
    os.makedirs(output_dir, exist_ok=True)

    model_path = os.path.join(output_dir, "model.joblib")
    scaler_path = os.path.join(output_dir, "scaler.joblib")
    metrics_path = os.path.join(output_dir, "metrics.json")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"[INFO] Saved model: {model_path}")
    print(f"[INFO] Saved scaler: {scaler_path}")
    print(f"[INFO] Saved metrics: {metrics_path}")


def train_pipeline(embeddings_dir="data/embeddings", output_dir="data/models"):
    """Full training pipeline for face verification classifier."""
    initial_time = time.time()
    print("[START] Training pipeline initiated 🚀")
    X, y = load_data(embeddings_dir)
    X_train, X_val, y_train, y_val = split_data(X, y)
    X_train_scaled, X_val_scaled, scaler = scale_data(X_train, X_val)
    model = train_classifier(X_train_scaled, y_train)
    metrics = evaluate_model(model, X_val_scaled, y_val)
    save_artifacts(model, scaler, metrics, output_dir)
    print("[DONE] Training complete ✅")
    print(f"[INFO] Total training time: {time.time() - initial_time:.2f} seconds")


if __name__ == "__main__":
    train_pipeline()
