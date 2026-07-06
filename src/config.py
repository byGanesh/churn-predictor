DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = "models/churn.pkl"
TARGET_COL = "Churn"
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

NUMERICAL_COLS = ["tenure", "MonthlyCharges", "SeniorCitizen", "TotalCharges"]

CATEGORICAL_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

MODEL_PARAMS = {
    "l1_ratio": 0,  # L2 (ridge)
    "C": 1.0,  # max 1/lambda
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}

MLFLOW_EXPERIMENT = "churn-predictor"
