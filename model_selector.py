import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def prepare_features(df):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    df["lag_1"] = df["value"].shift(1)
    df["lag_2"] = df["value"].shift(2)
    df["lag_3"] = df["value"].shift(3)
    df["rolling_mean_3"] = df["value"].rolling(3).mean()
    df["rolling_std_3"] = df["value"].rolling(3).std()

    return df.dropna()


def evaluate_models(df):
    df_model = prepare_features(df)

    features = ["lag_1", "lag_2", "lag_3", "rolling_mean_3", "rolling_std_3"]

    split = int(len(df_model) * 0.8)

    train = df_model.iloc[:split]
    test = df_model.iloc[split:]

    X_train = train[features]
    y_train = train["value"]

    X_test = test[features]
    y_test = test["value"]

    results = {}
    trained_models = {}

    # Naive baseline: tomorrow = today
    naive_pred = test["lag_1"]
    results["Naive Baseline"] = {
        "MAE": mean_absolute_error(y_test, naive_pred),
        "MSE": mean_squared_error(y_test, naive_pred)
    }

    # Moving average baseline
    ma_pred = test["rolling_mean_3"]
    results["Moving Average"] = {
        "MAE": mean_absolute_error(y_test, ma_pred),
        "MSE": mean_squared_error(y_test, ma_pred)
    }

    # Random Forest
    rf = RandomForestRegressor(n_estimators=200, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    results["Random Forest"] = {
        "MAE": mean_absolute_error(y_test, rf_pred),
        "MSE": mean_squared_error(y_test, rf_pred)
    }
    trained_models["Random Forest"] = rf

    # XGBoost
    xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
    xgb.fit(X_train, y_train)
    xgb_pred = xgb.predict(X_test)

    results["XGBoost"] = {
        "MAE": mean_absolute_error(y_test, xgb_pred),
        "MSE": mean_squared_error(y_test, xgb_pred)
    }
    trained_models["XGBoost"] = xgb

    best_model_name = min(results, key=lambda name: results[name]["MAE"])

    return results, trained_models, best_model_name, features


def forecast_next_days(df, model_name, trained_models, features, horizon=7):
    temp_df = df.copy()
    temp_df["date"] = pd.to_datetime(temp_df["date"])
    temp_df = temp_df.sort_values("date")

    predictions = []

    for _ in range(horizon):
        temp_features = prepare_features(temp_df)
        latest_row = temp_features.iloc[-1]

        if model_name == "Naive Baseline":
            next_pred = latest_row["lag_1"]

        elif model_name == "Moving Average":
            next_pred = latest_row["rolling_mean_3"]

        else:
            model = trained_models[model_name]
            X_latest = temp_features[features].iloc[[-1]]
            next_pred = model.predict(X_latest)[0]

        next_date = temp_df["date"].iloc[-1] + pd.Timedelta(days=1)

        predictions.append({
            "date": next_date,
            "forecast": float(next_pred)
        })

        new_row = pd.DataFrame({
            "date": [next_date],
            "value": [next_pred]
        })

        temp_df = pd.concat([temp_df, new_row], ignore_index=True)

    return pd.DataFrame(predictions)