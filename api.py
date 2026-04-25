from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

model = joblib.load("models/forecast_model.pkl")

class ForecastInput(BaseModel):
    lag_1: float
    lag_2: float
    lag_3: float

@app.get("/")
def home():
    return {"message": "Forecasting API is running"}

@app.post("/predict")
def predict(data: ForecastInput):
    input_df = pd.DataFrame([{
        "lag_1": data.lag_1,
        "lag_2": data.lag_2,
        "lag_3": data.lag_3
    }])

    prediction = model.predict(input_df)[0]

    return {"forecast": round(float(prediction), 2)}