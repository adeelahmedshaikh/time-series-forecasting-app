import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from model_selector import evaluate_models, forecast_next_days
from data_utils import detect_date_column, detect_target_column, standardize_columns


st.set_page_config(
    page_title="Auto Time-Series Forecasting App",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Auto Time-Series Forecasting App")

st.write(
    "Upload a time-series CSV file, automatically compare forecasting models, "
    "select the best model, and generate forecasts with insights, alerts, and recommendations."
)

st.sidebar.header("Settings")

uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Upload a CSV file to start forecasting.")
    st.stop()

df_raw = pd.read_csv(uploaded_file)

st.subheader("Dataset Preview")
st.dataframe(df_raw.head())

detected_date = detect_date_column(df_raw)
detected_target = detect_target_column(df_raw)

st.sidebar.subheader("Column Mapping")

date_col = st.sidebar.selectbox(
    "Select date column",
    df_raw.columns.tolist(),
    index=df_raw.columns.tolist().index(detected_date) if detected_date in df_raw.columns else 0
)

numeric_cols = df_raw.select_dtypes(include=["number"]).columns.tolist()

if not numeric_cols:
    st.error("No numeric column found for forecasting.")
    st.stop()

target_col = st.sidebar.selectbox(
    "Select target column",
    numeric_cols,
    index=numeric_cols.index(detected_target) if detected_target in numeric_cols else 0
)

target_type = st.sidebar.selectbox(
    "What are you forecasting?",
    [
        "Sales / Demand",
        "Revenue",
        "Stock / Asset Price",
        "Energy / Usage",
        "Temperature / Weather",
        "Other Numeric Value"
    ]
)

df = standardize_columns(df_raw, date_col, target_col)

if len(df) < 10:
    st.warning("Dataset is too small. Please upload at least 10 valid rows.")
    st.stop()

results, trained_models, best_model, features = evaluate_models(df)

st.sidebar.subheader("Model Settings")

mode = st.sidebar.radio(
    "Choose forecasting mode",
    ["Automatic Best Model", "Manual Model Selection"]
)

if mode == "Automatic Best Model":
    selected_model = best_model
else:
    selected_model = st.sidebar.selectbox("Choose model", list(results.keys()))

horizon = st.sidebar.selectbox("Forecast horizon", [7, 14, 30])

forecast_df = forecast_next_days(
    df=df,
    model_name=selected_model,
    trained_models=trained_models,
    features=features,
    horizon=horizon
)

first_forecast = forecast_df["forecast"].iloc[0]
last_forecast = forecast_df["forecast"].iloc[-1]
forecast_change = ((last_forecast - first_forecast) / first_forecast) * 100
volatility = forecast_df["forecast"].std()

st.subheader("Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Rows", len(df))
col2.metric("Best Model", best_model)
col3.metric("Selected Model", selected_model)
col4.metric("Forecast Horizon", f"{horizon} days")

st.subheader("Model Leaderboard")

comparison_df = pd.DataFrame(results).T
st.dataframe(
    comparison_df.style.format({
        "MAE": "{:.3f}",
        "MSE": "{:.3f}"
    })
)

st.info(f"Best model based on MAE: **{best_model}**")

st.subheader("Forecast Results")
st.dataframe(forecast_df)

csv = forecast_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Forecast CSV",
    data=csv,
    file_name="forecast.csv",
    mime="text/csv",
)

st.subheader("Forecast Chart")

fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(df["date"], df["value"], label="Actual")
ax.plot(
    forecast_df["date"],
    forecast_df["forecast"],
    marker="o",
    label="Forecast"
)

ax.set_xlabel("Date")
ax.set_ylabel(target_col)
ax.legend()

st.pyplot(fig)

st.subheader("Executive Summary")

st.info(
    f"The selected model is **{selected_model}**. "
    f"The forecast change over the next **{horizon} days** is **{forecast_change:.2f}%**."
)

st.subheader("Insights")

if forecast_change > 2:
    st.success(f"Trend: Forecast is increasing by about {forecast_change:.2f}%.")
elif forecast_change < -2:
    st.warning(f"Trend: Forecast is decreasing by about {abs(forecast_change):.2f}%.")
else:
    st.info(f"Trend: Forecast is mostly stable ({forecast_change:.2f}% change).")

if volatility > df["value"].std() * 0.5:
    st.warning("Risk: Forecast shows high volatility. Be careful with large decisions.")
else:
    st.success("Risk: Forecast is relatively stable.")

st.subheader("Recommendations")

if target_type == "Sales / Demand":
    if forecast_change > 2:
        st.success("Recommendation: Demand is expected to rise. Consider increasing inventory or preparation.")
    elif forecast_change < -2:
        st.warning("Recommendation: Demand is expected to decline. Avoid overstocking.")
    else:
        st.info("Recommendation: Demand looks stable. Maintain current strategy.")

elif target_type == "Revenue":
    if forecast_change > 2:
        st.success("Recommendation: Revenue is expected to increase. Consider reinforcing successful channels.")
    elif forecast_change < -2:
        st.warning("Recommendation: Revenue may decline. Review pricing, marketing, or demand drivers.")
    else:
        st.info("Recommendation: Revenue looks stable.")

elif target_type == "Stock / Asset Price":
    if forecast_change > 2:
        st.info("Observation: Upward movement is forecasted. This is not financial advice.")
    elif forecast_change < -2:
        st.info("Observation: Downward movement is forecasted. This is not financial advice.")
    else:
        st.info("Observation: Price appears relatively stable. This is not financial advice.")

elif target_type == "Energy / Usage":
    if forecast_change > 2:
        st.success("Recommendation: Usage is expected to rise. Prepare additional capacity or resources.")
    elif forecast_change < -2:
        st.warning("Recommendation: Usage is expected to fall. Avoid over-allocation of resources.")
    else:
        st.info("Recommendation: Usage looks stable.")

elif target_type == "Temperature / Weather":
    if forecast_change > 2:
        st.info("Observation: Temperature/weather-related value is expected to increase.")
    elif forecast_change < -2:
        st.info("Observation: Temperature/weather-related value is expected to decrease.")
    else:
        st.info("Observation: Forecast looks stable.")

else:
    if forecast_change > 2:
        st.info("Observation: The target value is expected to increase.")
    elif forecast_change < -2:
        st.info("Observation: The target value is expected to decrease.")
    else:
        st.info("Observation: The target value looks stable.")

st.subheader("Risk Alerts")

if forecast_change > 10:
    st.error("Alert: Strong upward movement expected.")
elif forecast_change < -10:
    st.error("Alert: Strong downward movement expected.")
else:
    st.success("No major alert detected.")