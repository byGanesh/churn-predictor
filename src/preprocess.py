import pandas as pd

from config import CATEGORICAL_COLS, DATA_PATH, NUMERICAL_COLS


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df.drop(columns=["customerID"], inplace=True)
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


print(load_data(DATA_PATH).info())
