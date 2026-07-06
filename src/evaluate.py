import mlflow
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


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

    print("---- Test Metrics -----\n")
    for k, v in metrics.items():
        print(f"{k:12s} : {v:.4f}")

    print("---- Confusion Matrix -----")
    print(confusion_matrix(y_test, y_pred))

    print("---- Classification report -----")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    return metrics


def log_metrics(metrics: dict):
    mlflow.log_metrics(metrics)
