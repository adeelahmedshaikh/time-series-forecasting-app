import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib

st.title("Time-Series Forecasting App")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

model = joblib.load("models/forecast_model.pkl")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.write(df.head())

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")

    st.subheader("Original Time Series")
    fig, ax = plt.subplots()
    ax.plot(df["date"], df["value"])
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    st.pyplot(fig)

    values = df["value"].tolist()

    if len(values) >= 3:
        lag_1 = values[-1]
        lag_2 = values[-2]
        lag_3 = values[-3]

        prediction = model.predict(pd.DataFrame([{
            "lag_1": lag_1,
            "lag_2": lag_2,
            "lag_3": lag_3
        }]))[0]

        st.subheader("Next Forecast")
        st.success(f"Predicted next value: {prediction:.2f}")
    else:
        st.warning("Please upload at least 3 rows of data.")