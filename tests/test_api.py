import pytest
from fastapi.testclient import TestClient

from api.app import app

client = TestClient(app)

sample = {
    "SeniorCitizen": 0,
    "gender": "Male",
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 845.50,
}


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"


def test_predict():
    r = client.post("/predict", json=sample)
    assert r.status_code == 200
    body = r.json()
    assert "churn_probability" in body
    assert body["prediction"] in ["Churn", "No Churn"]
    assert body["risk_level"] in ["High", "Medium", "Low"]
    assert 0.0 <= body["churn_probability"] <= 1.0
