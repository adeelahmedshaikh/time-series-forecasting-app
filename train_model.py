import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv("data/sales.csv")

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")

# Lag features
df["lag_1"] = df["value"].shift(1)
df["lag_2"] = df["value"].shift(2)
df["lag_3"] = df["value"].shift(3)

# Rolling features
df["rolling_mean_3"] = df["value"].rolling(3).mean()
df["rolling_std_3"] = df["value"].rolling(3).std()

df = df.dropna()

features = ["lag_1", "lag_2", "lag_3", "rolling_mean_3", "rolling_std_3"]

X = df[features]
y = df["value"]

split = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

rf_model = RandomForestRegressor(n_estimators=200, random_state=42)
xgb_model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

rf_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)
xgb_pred = xgb_model.predict(X_test)

metrics = {
    "Random Forest": {
        "MAE": mean_absolute_error(y_test, rf_pred),
        "MSE": mean_squared_error(y_test, rf_pred)
    },
    "XGBoost": {
        "MAE": mean_absolute_error(y_test, xgb_pred),
        "MSE": mean_squared_error(y_test, xgb_pred)
    }
}

os.makedirs("models", exist_ok=True)

joblib.dump(rf_model, "models/random_forest_model.pkl")
joblib.dump(xgb_model, "models/xgboost_model.pkl")
joblib.dump(features, "models/features.pkl")
joblib.dump(metrics, "models/metrics.pkl")

print("Models trained and saved.")
print(metrics)