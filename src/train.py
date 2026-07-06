from pandas.core.common import random_state
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

    pipeline.fit(X_train, y_train)


train()
