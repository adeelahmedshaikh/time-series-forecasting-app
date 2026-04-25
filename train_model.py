import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

df = pd.read_csv("data/sales.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

df["lag_1"] = df["value"].shift(1)
df["lag_2"] = df["value"].shift(2)
df["lag_3"] = df["value"].shift(3)

df = df.dropna()

X = df[["lag_1", "lag_2", "lag_3"]]
y = df["value"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/forecast_model.pkl")

print("Model trained and saved.")