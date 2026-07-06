import os
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

os.makedirs("assets", exist_ok=True)


def compute_metrics(pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print("\n---- Test Metrics -----")
    for k, v in metrics.items():
        print(f"{k:12s} : {v:.4f}")

    print("\n---- Confusion Matrix -----")
    print(confusion_matrix(y_test, y_pred))

    print("\n---- Classification report -----")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_proba)
    plot_precision_recall_curve(y_test, y_proba)
    plot_threshold_curve(y_test, y_proba)

    return metrics


def log_metrics(metrics: dict):
    mlflow.log_metrics(metrics)


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    labels = [["TN", "FP"], ["FN", "TP"]]
    annot = [[f"{labels[i][j]}\n{cm[i][j]}" for j in range(2)] for i in range(2)]

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"],
        linewidths=2,
        linecolor="white",
        cbar=False,
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig("assets/confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved assets/confusion_matrix.png")


def plot_roc_curve(y_test, y_proba):
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("assets/roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved assets/roc_curve.png")


def plot_precision_recall_curve(y_test, y_proba):
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    baseline = y_test.mean()

    plt.figure(figsize=(6, 5))
    plt.plot(recall_vals, precision_vals, lw=2, label=f"AP = {ap:.3f}")
    plt.axhline(
        baseline, linestyle="--", color="gray", label=f"Baseline ({baseline:.2f})"
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig("assets/precision_recall_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved assets/precision_recall_curve.png")


def plot_threshold_curve(y_test, y_proba):
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_test, y_proba)

    plt.figure(figsize=(7, 4))
    plt.plot(thresholds, precision_vals[:-1], lw=2, label="Precision")
    plt.plot(thresholds, recall_vals[:-1], lw=2, label="Recall")
    plt.axvline(0.5, linestyle="--", color="gray", label="Default (0.5)")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Precision & Recall vs Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig("assets/threshold_curve.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved assets/threshold_curve.png")
