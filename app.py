import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from charts import create_interactive_forecast, create_model_comparison_chart
from report_generator import generate_forecast_report

from model_selector import evaluate_models, forecast_next_days
from data_utils import (
    detect_date_column,
    detect_target_column,
    standardize_columns,
    fetch_yahoo_data
)
from validation import validate_data, get_data_quality_report, suggest_data_fixes
from cache_manager import ForecastCache

# Initialize cache
forecast_cache = ForecastCache()

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

data_source = st.sidebar.radio(
    "Choose data source",
    ["Upload CSV", "Yahoo Finance"]
)

df_raw = None

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is None:
        st.info("Upload a CSV file to start forecasting.")
        st.stop()

    df_raw = pd.read_csv(uploaded_file)

else:
    st.sidebar.subheader("Yahoo Finance Data")

    preset_assets = {
        "Apple (AAPL)": "AAPL",
        "Tesla (TSLA)": "TSLA",
        "Microsoft (MSFT)": "MSFT",
        "Amazon (AMZN)": "AMZN",
        "Google (GOOGL)": "GOOGL",
        "NVIDIA (NVDA)": "NVDA",
        "Gold Futures (GC=F)": "GC=F",
        "Bitcoin (BTC-USD)": "BTC-USD",
        "S&P 500 (^GSPC)": "^GSPC",
        "EUR/USD (EURUSD=X)": "EURUSD=X"
    }

    selected_asset = st.sidebar.selectbox(
        "Choose popular asset",
        list(preset_assets.keys())
    )

    manual_ticker = st.sidebar.text_input(
        "Enter ticker (e.g., AAPL for Apple, BLK for BlackRock)",
        value=preset_assets[selected_asset]
    )

    st.sidebar.info(
        "💡 Examples:\n"
        "Apple → AAPL\n"
        "Tesla → TSLA\n"
        "BlackRock → BLK\n"
        "Bitcoin → BTC-USD\n"
        "Gold → GC=F"
    )

    period = st.sidebar.selectbox(
        "Select data period",
        ["1y", "2y", "5y", "10y"]
    )

    if st.sidebar.button("Fetch Data"):
        with st.spinner("Fetching data from Yahoo Finance..."):
            df_raw = fetch_yahoo_data(manual_ticker, period)

        if df_raw is None or df_raw.empty:
            st.error(
                "❌ Invalid ticker symbol.\n\n"
                "Examples:\n"
                "AAPL (Apple), TSLA (Tesla), BLK (BlackRock), BTC-USD (Bitcoin), GC=F (Gold)\n\n"
                "Tip: Search on Google → 'company name + ticker'"
            )
            st.stop()
    else:
        st.info("Choose a Yahoo Finance asset and click Fetch Data.")
        st.stop()

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

# Standardize columns
df = standardize_columns(df_raw, date_col, target_col)

# ============ DATA VALIDATION ============
is_valid, error_msg = validate_data(df)
if not is_valid:
    st.error(f"❌ {error_msg}")
    
    suggestions = suggest_data_fixes(df)
    if suggestions:
        st.info("💡 **Suggestions to fix your data:**")
        for suggestion in suggestions:
            st.write(suggestion)
    st.stop()

# Show data quality report in sidebar
with st.sidebar.expander("📊 Data Quality Report"):
    quality_report = get_data_quality_report(df)
    if "error" not in quality_report:
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Rows", quality_report["total_rows"])
            st.metric("Missing Values", quality_report["missing_values"])
            st.metric("Date Range", quality_report["date_range"])
        with col2:
            st.metric("Mean Value", f"{quality_report['value_mean']:.2f}")
            st.metric("Std Dev", f"{quality_report['value_std']:.2f}")
            st.metric("Has Date Gaps", "Yes ⚠️" if quality_report["has_date_gaps"] else "No ✅")
        
        if quality_report["has_date_gaps"]:
            st.warning(f"Found {quality_report['gap_count']} gaps in dates. Forecast may be less accurate.")

if len(df) < 10:
    st.warning("Dataset is too small. Please upload at least 10 valid rows.")
    st.stop()

# Train models with progress indicator
with st.spinner("Training models and evaluating performance..."):
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

# ============ CACHED FORECAST WITH PROGRESS ============
cached_forecast = forecast_cache.get_cached_forecast(df, horizon, selected_model)

if cached_forecast is not None:
    forecast_df = cached_forecast
    st.info("📦 Using cached forecast (from previous calculation) - This is faster!")
else:
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    def update_progress(current, total):
        percent = int((current / total) * 100)
        progress_bar.progress(percent)
        status_text.text(f"🔄 Generating forecast: Day {current} of {total} ({percent}%)")
    
    forecast_df = forecast_next_days(
        df=df,
        model_name=selected_model,
        trained_models=trained_models,
        features=features,
        horizon=horizon,
        progress_callback=update_progress
    )
    
    progress_bar.empty()
    status_text.empty()
    
    forecast_cache.save_forecast(df, horizon, selected_model, forecast_df)
    st.success("✅ Forecast generated and cached for faster future access!")

# Show cache stats in sidebar
with st.sidebar.expander("⚡ Cache Info"):
    cache_stats = forecast_cache.get_cache_stats()
    st.write(f"📦 Cached forecasts: {cache_stats['total_cached']}")
    st.write(f"💾 Cache size: {cache_stats['cache_size_mb']:.2f} MB")
    
    oldest_cache = cache_stats.get('oldest_cache')
    if oldest_cache is not None:
        try:
            if hasattr(oldest_cache, 'strftime'):
                st.write(f"🕐 Oldest cache: {oldest_cache.strftime('%Y-%m-%d')}")
            else:
                st.write(f"🕐 Oldest cache: {oldest_cache}")
        except Exception:
            st.write(f"🕐 Oldest cache: {oldest_cache}")
    
    if st.button("🗑️ Clear Old Cache"):
        forecast_cache.clear_old_cache()
        st.success("Old cache cleared!")
        st.rerun()

first_forecast = forecast_df["forecast"].iloc[0]
last_forecast = forecast_df["forecast"].iloc[-1]
forecast_change = ((last_forecast - first_forecast) / first_forecast) * 100
volatility = forecast_df["forecast"].std()

# ============ OVERVIEW SECTION ============
st.subheader("Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Rows", len(df))
col2.metric("Best Model", best_model)
col3.metric("Selected Model", selected_model)
col4.metric("Forecast Horizon", f"{horizon} days")

# ============ MODEL LEADERBOARD ============
st.subheader("Model Leaderboard")

comparison_df = pd.DataFrame(results).T
st.dataframe(
    comparison_df.style.format({
        "MAE": "{:.3f}",
        "MSE": "{:.3f}"
    })
)

st.info(f"Best model based on MAE: **{best_model}**")

# ============ FORECAST RESULTS & DOWNLOADS ============
st.subheader("Forecast Results")

# Show the forecast table
st.dataframe(forecast_df)

# Download buttons side by side
col1, col2 = st.columns(2)

with col1:
    csv = forecast_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"forecast_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

with col2:
    # Simple direct PDF generation
    pdf_buffer = generate_forecast_report(
        df=df,
        forecast_df=forecast_df,
        results=results,
        best_model=best_model,
        forecast_change=forecast_change,
        target_type=target_type,
        target_col=target_col
    )
    
    st.download_button(
        label="📥 Download PDF Report",
        data=pdf_buffer,
        file_name=f"forecast_report_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf",
        mime="application/pdf",
        key="pdf_direct_btn"
    )

# # ============ FORECAST RESULTS & DOWNLOADS ============
# st.subheader("Forecast Results")

# # Show the forecast table
# st.dataframe(forecast_df)

# # Download buttons side by side
# col1, col2 = st.columns(2)

# with col1:
#     csv = forecast_df.to_csv(index=False).encode("utf-8")
#     st.download_button(
#         label="📥 Download CSV",
#         data=csv,
#         file_name=f"forecast_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
#         mime="text/csv",
#     )

# with col2:
#     # Initialize session state for PDF
#     if 'pdf_generated' not in st.session_state:
#         st.session_state.pdf_generated = False
#     if 'pdf_buffer' not in st.session_state:
#         st.session_state.pdf_buffer = None
    
#     if st.button("📄 Generate PDF Report", key="pdf_gen_btn"):
#         with st.spinner("Generating PDF report..."):
#             st.session_state.pdf_buffer = generate_forecast_report(
#                 df=df,
#                 forecast_df=forecast_df,
#                 results=results,
#                 best_model=best_model,
#                 forecast_change=forecast_change,
#                 target_type=target_type,
#                 target_col=target_col
#             )
#             st.session_state.pdf_generated = True
#             st.success("PDF Ready! Click download button below.")
    
#     # Show download button if PDF is generated
#     if st.session_state.pdf_generated and st.session_state.pdf_buffer is not None:
#         st.download_button(
#             label="📥 Download PDF",
#             data=st.session_state.pdf_buffer,
#             file_name=f"forecast_report_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf",
#             mime="application/pdf",
#             key="pdf_download_btn"
#         )

# ============ FORECAST CHART ============
st.subheader("Forecast Chart")

fig = create_interactive_forecast(df, forecast_df, f"{target_col} Forecast")
st.plotly_chart(fig, use_container_width=True)

# ============ MODEL PERFORMANCE CHART ============
st.subheader("Model Performance Comparison")
comparison_chart = create_model_comparison_chart(comparison_df)
st.plotly_chart(comparison_chart, use_container_width=True)

# ============ EXECUTIVE SUMMARY ============
st.subheader("Executive Summary")

st.info(
    f"The selected model is **{selected_model}**. "
    f"The forecast change over the next **{horizon} days** is **{forecast_change:.2f}%**."
)

# ============ INSIGHTS ============
st.subheader("Insights")

if forecast_change > 2:
    st.success(f"📈 Trend: Forecast is increasing by about {forecast_change:.2f}%.")
elif forecast_change < -2:
    st.warning(f"📉 Trend: Forecast is decreasing by about {abs(forecast_change):.2f}%.")
else:
    st.info(f"➡️ Trend: Forecast is mostly stable ({forecast_change:.2f}% change).")

if volatility > df["value"].std() * 0.5:
    st.warning("⚠️ Risk: Forecast shows high volatility. Be careful with large decisions.")
else:
    st.success("✅ Risk: Forecast is relatively stable.")

# ============ RECOMMENDATIONS ============
st.subheader("Recommendations")

if target_type == "Sales / Demand":
    if forecast_change > 2:
        st.success("📦 Recommendation: Demand is expected to rise. Consider increasing inventory or preparation.")
    elif forecast_change < -2:
        st.warning("📦 Recommendation: Demand is expected to decline. Avoid overstocking.")
    else:
        st.info("📦 Recommendation: Demand looks stable. Maintain current strategy.")

elif target_type == "Revenue":
    if forecast_change > 2:
        st.success("💰 Recommendation: Revenue is expected to increase. Consider reinforcing successful channels.")
    elif forecast_change < -2:
        st.warning("💰 Recommendation: Revenue may decline. Review pricing, marketing, or demand drivers.")
    else:
        st.info("💰 Recommendation: Revenue looks stable.")

elif target_type == "Stock / Asset Price":
    if forecast_change > 2:
        st.info("📊 Observation: Upward movement is forecasted. This is not financial advice.")
    elif forecast_change < -2:
        st.info("📊 Observation: Downward movement is forecasted. This is not financial advice.")
    else:
        st.info("📊 Observation: Price appears relatively stable. This is not financial advice.")

elif target_type == "Energy / Usage":
    if forecast_change > 2:
        st.success("⚡ Recommendation: Usage is expected to rise. Prepare additional capacity or resources.")
    elif forecast_change < -2:
        st.warning("⚡ Recommendation: Usage is expected to fall. Avoid over-allocation of resources.")
    else:
        st.info("⚡ Recommendation: Usage looks stable.")

elif target_type == "Temperature / Weather":
    if forecast_change > 2:
        st.info("🌡️ Observation: Temperature/weather-related value is expected to increase.")
    elif forecast_change < -2:
        st.info("🌡️ Observation: Temperature/weather-related value is expected to decrease.")
    else:
        st.info("🌡️ Observation: Forecast looks stable.")

else:
    if forecast_change > 2:
        st.info("📈 Observation: The target value is expected to increase.")
    elif forecast_change < -2:
        st.info("📉 Observation: The target value is expected to decrease.")
    else:
        st.info("➡️ Observation: The target value looks stable.")

# ============ RISK ALERTS ============
st.subheader("Risk Alerts")

if forecast_change > 10:
    st.error("🚨 ALERT: Strong upward movement expected (>10%).")
elif forecast_change < -10:
    st.error("🚨 ALERT: Strong downward movement expected (>10%).")
elif forecast_change > 5:
    st.warning("⚠️ Notice: Moderate upward movement expected (5-10%).")
elif forecast_change < -5:
    st.warning("⚠️ Notice: Moderate downward movement expected (5-10%).")
else:
    st.success("✅ No major alert detected.")