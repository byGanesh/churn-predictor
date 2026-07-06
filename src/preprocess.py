import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_COLS, DATA_PATH, NUMERICAL_COLS


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.drop(columns=["customerID"], inplace=True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def split_features_target(df: pd.DataFrame):
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    return X, y


def build_preprocessor() -> ColumnTransformer:
    num_trans = Pipeline(
        [("impuster", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    cat_trans = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [("num", num_trans, NUMERICAL_COLS), ("cat", cat_trans, CATEGORICAL_COLS)]
    )
