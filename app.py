from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI(title="Support Vector Machine API")


# =========================
# LOAD MODEL
# =========================

svc_model = joblib.load("svc_model.pkl")
svr_model = joblib.load("svr_model.pkl")

scaler_svc = joblib.load("scaler_svc.pkl")
scaler_svr = joblib.load("scaler_svr.pkl")


# =========================
# API KIỂM TRA
# =========================

@app.get("/")
def home():
    return {
        "message": "Support Vector Machine API is running"
    }


# =========================
# API SVC - PHÂN LOẠI
# =========================

@app.post("/predict-classification")
def predict_classification(features: list[float]):

    data = np.array(features).reshape(1, -1)

    data_scaled = scaler_svc.transform(data)

    prediction = svc_model.predict(data_scaled)

    return {
        "prediction": prediction.tolist()
    }


# =========================
# API SVR - DỰ ĐOÁN GIÁ
# =========================

@app.post("/predict-regression")
def predict_regression(features: list[float]):

    data = np.array(features).reshape(1, -1)

    data_scaled = scaler_svr.transform(data)

    prediction = svr_model.predict(data_scaled)

    return {
        "prediction": prediction.tolist()
    }