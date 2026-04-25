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

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.write(df.head())

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    df["lag_1"] = df["value"].shift(1)
    df["lag_2"] = df["value"].shift(2)
    df["lag_3"] = df["value"].shift(3)
    df["rolling_mean_3"] = df["value"].rolling(3).mean()
    df["rolling_std_3"] = df["value"].rolling(3).std()

    df_model = df.dropna()

    if len(df_model) > 0:
        latest_features = df_model[features].iloc[[-1]]
        prediction = model.predict(latest_features)[0]

        st.subheader("Next Forecast")
        st.success(f"{model_choice} predicted next value: {prediction:.2f}")

        last_date = df["date"].iloc[-1]
        next_date = last_date + pd.Timedelta(days=1)

        forecast_df = pd.DataFrame({
            "date": [next_date],
            "value": [prediction]
        })

        st.subheader("Forecast Visualization")

        fig, ax = plt.subplots()
        ax.plot(df["date"], df["value"], label="Actual")
        ax.scatter(forecast_df["date"], forecast_df["value"], label="Forecast")
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
        ax.legend()
        st.pyplot(fig)

    else:
        st.warning("Not enough data after feature creation.")