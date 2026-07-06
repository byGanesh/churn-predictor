DATA_PATH = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = "models/churn.pkl"
TARGET_COL = "Churn"
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

NUMERICAL_COLS = ["tenure", "MonthlyCharges", "SeniorCitizen"]

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
    "TotalCharges",
]

MODEL_PARAMS = {
    "penalty": "l2",
    "C": 1.0,
    "max_iter": 1000,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE,
}
