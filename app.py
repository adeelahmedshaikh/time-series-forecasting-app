import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from model_selector import evaluate_models, forecast_next_days
from data_utils import detect_date_column, detect_target_column, standardize_columns

st.title("Auto Time-Series Forecasting App")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data")
    st.write(df.head())

    detected_date = detect_date_column(df)
    detected_target = detect_target_column(df)

    st.subheader("Column Selection")

    date_col = st.selectbox(
        "Select date column",
        df.columns.tolist(),
        index=df.columns.tolist().index(detected_date) if detected_date in df.columns else 0
    )

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    if not numeric_cols:
        st.error("No numeric column found for forecasting.")
        st.stop()

    target_col = st.selectbox(
        "Select target column",
        numeric_cols,
        index=numeric_cols.index(detected_target) if detected_target in numeric_cols else 0
    )

    df = standardize_columns(df, date_col, target_col)

    st.subheader("Dataset Information")
    st.write(f"Number of rows: {len(df)}")

    if len(df) < 10:
        st.warning("Dataset is too small. Please upload at least 10 rows.")
        st.stop()

    results, trained_models, best_model, features = evaluate_models(df)

    st.subheader("Model Comparison")
    comparison_df = pd.DataFrame(results).T
    st.write(comparison_df)

    st.info(f"Best model based on MAE: {best_model}")

    st.subheader("Model Selection")

    mode = st.radio(
        "Choose forecasting mode",
        ["Automatic Best Model", "Manual Model Selection"]
    )

    if mode == "Automatic Best Model":
        selected_model = best_model
    else:
        selected_model = st.selectbox("Choose model", list(results.keys()))

    st.write(f"Selected model: **{selected_model}**")

    horizon = st.selectbox("Forecast horizon", [7, 14, 30])

    forecast_df = forecast_next_days(
        df=df,
        model_name=selected_model,
        trained_models=trained_models,
        features=features,
        horizon=horizon
    )

    st.subheader(f"{horizon}-Day Forecast")
    st.write(forecast_df)

    st.subheader("Forecast Visualization")

    fig, ax = plt.subplots()
    ax.plot(df["date"], df["value"], label="Actual")
    ax.plot(
        forecast_df["date"],
        forecast_df["forecast"],
        marker="o",
        label="Forecast"
    )
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Insight")

    first_forecast = forecast_df["forecast"].iloc[0]
    last_forecast = forecast_df["forecast"].iloc[-1]

    forecast_change = ((last_forecast - first_forecast) / first_forecast) * 100

    if forecast_change > 2:
        st.success(f"Trend: Forecast is increasing by about {forecast_change:.2f}%.")
    elif forecast_change < -2:
        st.warning(f"Trend: Forecast is decreasing by about {abs(forecast_change):.2f}%.")
    else:
        st.info(f"Trend: Forecast is mostly stable ({forecast_change:.2f}% change).")

    volatility = forecast_df["forecast"].std()

    if volatility > df["value"].std() * 0.5:
        st.warning("Risk: Forecast shows high volatility. Be careful with large decisions.")
    else:
        st.success("Risk: Forecast is relatively stable.")

    st.subheader("Recommendations")

    if forecast_change > 2:
        st.success("Recommendation: Demand is expected to rise. Consider increasing inventory or preparation.")
    elif forecast_change < -2:
        st.warning("Recommendation: Demand is expected to decline. Avoid overstocking.")
    else:
        st.info("Recommendation: Demand looks stable. Maintain current strategy.")

    st.subheader("Alerts")

    if forecast_change > 10:
        st.error("Alert: Strong upward movement expected.")
    elif forecast_change < -10:
        st.error("Alert: Strong downward movement expected.")
    else:
        st.success("No major alert detected.")