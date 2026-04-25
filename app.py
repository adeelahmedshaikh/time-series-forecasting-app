import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.title("Time-Series Forecasting App")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

rf_model = joblib.load("models/random_forest_model.pkl")
xgb_model = joblib.load("models/xgboost_model.pkl")
features = joblib.load("models/features.pkl")
metrics = joblib.load("models/metrics.pkl")

model_choice = st.selectbox(
    "Choose forecasting model",
    ["Random Forest", "XGBoost"]
)

model = rf_model if model_choice == "Random Forest" else xgb_model

st.subheader("Model Performance")
st.write(metrics[model_choice])

st.subheader("Model Comparison")

comparison_df = pd.DataFrame(metrics).T
st.write(comparison_df)

best_model = min(metrics, key=lambda x: metrics[x]["MAE"])
st.info(f"Best model based on MAE: {best_model}")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.write(df.head())

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    if len(df) >= 5:
        horizon = 7
        predictions = []
        temp_df = df.copy()

        for _ in range(horizon):
            temp_df["lag_1"] = temp_df["value"].shift(1)
            temp_df["lag_2"] = temp_df["value"].shift(2)
            temp_df["lag_3"] = temp_df["value"].shift(3)
            temp_df["rolling_mean_3"] = temp_df["value"].rolling(3).mean()
            temp_df["rolling_std_3"] = temp_df["value"].rolling(3).std()

            temp_model_df = temp_df.dropna()

            latest_features = temp_model_df[features].iloc[[-1]]
            next_pred = model.predict(latest_features)[0]

            predictions.append(float(next_pred))

            next_date = temp_df["date"].iloc[-1] + pd.Timedelta(days=1)

            new_row = pd.DataFrame({
                "date": [next_date],
                "value": [next_pred]
            })

            temp_df = pd.concat([temp_df, new_row], ignore_index=True)

        forecast_dates = pd.date_range(
            start=df["date"].iloc[-1] + pd.Timedelta(days=1),
            periods=horizon
        )

        forecast_df = pd.DataFrame({
            "date": forecast_dates,
            "forecast": predictions
        })

        st.subheader("7-Day Forecast")
        st.write(forecast_df)

        st.success(
            f"{model_choice} predicted next 7 values successfully."
        )

        st.subheader("Forecast Visualization")

        fig, ax = plt.subplots()
        ax.plot(df["date"], df["value"], label="Actual")
        ax.plot(
            forecast_df["date"],
            forecast_df["forecast"],
            marker="o",
            label="7-Day Forecast"
        )
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        st.pyplot(fig)

    else:
        st.warning("Please upload at least 5 rows of data.")