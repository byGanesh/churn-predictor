import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

from config import (
    CV_FOLDS,
    DATA_PATH,
    MLFLOW_EXPERIMENT,
    MODEL_PARAMS,
    MODEL_PATH,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
)
from preprocess import build_preprocessor, load_data, split_features_target


def build_pipeline() -> Pipeline:
    preprocessor = build_preprocessor()
    model = LogisticRegression(**MODEL_PARAMS)

    return Pipeline([("preprocessor", preprocessor), ("model", model)])


def train():
    df = load_data(DATA_PATH)
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    pipeline = build_pipeline()

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    cv_results = cross_validate(
        pipeline, X_train, y_train, cv=cv, scoring=["precision", "f1", "roc_auc"]
    )

    print("----- CV RESULTS --------")
    for metric in ["precision", "f1", "roc_auc"]:
        scores = cv_results[f"test_{metric}"]
        print(f"{metric:12s} mean={scores.mean():.4f} std={scores.std():.4f}")

    pipeline.fit(X_train, y_train)

    metrics = compute_metrics(pipeline, X_test, y_test)

    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name="l2-balanced"):
        mlflow.log_params(MODEL_PARAMS)
        mlflow.log_param("cv_folds", CV_FOLDS)
        mlflow.log_param("test_size", TEST_SIZE)
        log_metrics(metrics)
        mlflow.log_metric("cv_f1_mean", cv_results["test_f1"].mean())
        mlflow.log_metric("cv_auc_mean", cv_results["test_roc_auc"].mean())
        mlflow.sklearn.log_model(pipeline, artifact_path="model")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"\n Pipeline Saved: {MODEL_PATH}")


if __name__ == "__main__":
    train()
