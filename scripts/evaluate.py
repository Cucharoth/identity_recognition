import os
import json
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    f1_score
)


def load_data(embeddings_dir="data/embeddings"):
    X = np.load(os.path.join(embeddings_dir, "embeddings.npy"))
    y = np.load(os.path.join(embeddings_dir, "labels.npy"))
    return X, y


def load_model(model_dir="data/models"):
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
    return model, scaler


def evaluate_model(X, y, model, scaler):
    X_scaled = scaler.transform(X)
    y_prob = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)

    # ROC + AUC
    fpr, tpr, _ = roc_curve(y, y_prob)
    auc = roc_auc_score(y, y_prob)

    # Precision–Recall + optimal t
    precision, recall, thresholds = precision_recall_curve(y, y_prob)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_tau = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]

    metrics = {
        "auc": round(auc, 4),
        "best_tau": round(float(best_tau), 4),
        "best_f1": round(float(best_f1), 4),
        "samples": len(y),
    }

    return metrics, cm, fpr, tpr, precision, recall


def save_plots(cm, fpr, tpr, precision, recall, metrics, output_dir="reports"):
    os.makedirs(output_dir, exist_ok=True)

    # Confusion Matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    # ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {metrics['auc']:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close()

    # Precision–Recall Curve
    plt.figure()
    plt.plot(recall, precision, label=f"Best τ={metrics['best_tau']:.2f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision–Recall Curve")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"))
    plt.close()

    # Save metrics
    with open(os.path.join(output_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"[INFO] Reports saved in {output_dir}/")


def main():
    print("[INFO] Starting evaluation...")
    X, y = load_data()
    model, scaler = load_model()
    metrics, cm, fpr, tpr, precision, recall = evaluate_model(X, y, model, scaler)
    save_plots(cm, fpr, tpr, precision, recall, metrics)
    print(f"[DONE] Evaluation complete. Best τ = {metrics['best_tau']:.3f}")


if __name__ == "__main__":
    main()
