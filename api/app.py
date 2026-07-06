import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import MODEL_PATH

app = FastAPI(
    title="ChurnPredictor", description="Customer Churn Prediction API", version="1.0.0"
)

pipeline = joblib.load(MODEL_PATH)


class Customer(BaseModel):
    SeniorCitizen: int
    gender: str
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float


class Prediction(BaseModel):
    churn_probability: float
    prediction: str
    risk_level: str


@app.get("/")
def root():
    return {"status": "ok", "model": "Logistic Regression", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer):
    try:
        df = pd.DataFrame([customer.model_dump()])
        prob = pipeline.predict_proba(df)[0][1]
        pred = pipeline.predict(df)[0]

        risk = "High" if prob >= 0.7 else "medium" if prob >= 0.4 else "low"

        return Prediction(
            churn_probability=round(float(prob), 4),
            prediction="Churn" if pred == 1 else "No Churn",
            risk_level=risk,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
