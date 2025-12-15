# src/api/main.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="Churn Prediction API")

# Load preprocessor and model (assumes they are in models/)
preprocessor = joblib.load("models/preprocessor.joblib")
model = joblib.load("models/model_rf.joblib")

# Example: construct model columns order if needed
# If your preprocessor requires raw DataFrame columns, pass them as DataFrame.

class Customer(BaseModel):
    # define required fields present in the original dataset (short example)
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    # (add the rest of fields as in dataset...)

@app.post("/predict")
def predict(customer: dict):
    # Accept raw dict, convert to single-row DataFrame
    df = pd.DataFrame([customer])
    # Ensure order and types as original if necessary
    X_proc = preprocessor.transform(df)
    proba = model.predict_proba(X_proc)[:,1][0]
    pred = int(proba >= 0.5)
    return {"prediction": pred, "probability": float(proba)}
