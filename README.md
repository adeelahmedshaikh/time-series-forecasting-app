# 📈 Auto Time-Series Forecasting & Decision App

A web-based application that automatically forecasts time-series data, selects the best-performing model, and provides actionable insights, alerts, and recommendations.

---

## 🚀 Live Demo

👉 https://time-series-forecasting-app-j6nv.onrender.com

---

## ⭐ Key Highlights

- Automatic model selection (no manual tuning required)
- Combines statistical + machine learning models
- Works with any time-series dataset (not limited to sales)
- Provides actionable insights, alerts, and recommendations
- Fully deployed and accessible online
- Download forecast results as CSV

---

## 🧠 How It Works

1. Upload your CSV file  
2. System automatically detects:
   - Date column  
   - Target (numeric) column  
3. Data is cleaned and transformed  
4. Multiple models are trained and evaluated:
   - Naive Baseline  
   - Moving Average  
   - ARIMA  
   - Random Forest  
   - XGBoost  
5. Best model is selected based on MAE  
6. Forecast is generated (7 / 14 / 30 days)  
7. Insights, recommendations, and alerts are displayed  

---

## 📊 Features

- 📂 Upload any time-series CSV
- 🔍 Automatic column detection
- ⚙️ Automatic + manual model selection
- 📈 Multi-horizon forecasting (7, 14, 30 days)
- 📊 Model comparison dashboard
- 📉 Trend analysis (increase / decrease / stable)
- ⚠️ Risk alerts based on forecast changes
- 💡 Domain-aware recommendations:
  - Sales / Demand  
  - Revenue  
  - Stock / Asset Price  
  - Energy / Usage  
  - Weather / Temperature  
- ⬇️ Download forecast results as CSV
- 🌐 Deployed on Render

---

## 🚀 What Makes This Different?

Unlike typical forecasting apps, this system:

- Automatically selects the best model based on data  
- Combines baseline, statistical, and ML models for fair comparison  
- Works across multiple domains (not just sales forecasting)  
- Focuses on decision-making (insights + recommendations), not just predictions  
- Includes simple baselines to ensure ML actually improves performance  

---

## 📊 Example Use Cases

- Sales forecasting for small businesses  
- Stock or asset price trend analysis  
- Revenue prediction  
- Energy demand forecasting  
- General time-series analysis  

---

## 🛠 Tech Stack

- Python  
- Streamlit  
- Pandas  
- Scikit-learn  
- XGBoost  
- Statsmodels (ARIMA)  
- Matplotlib  

---

## ⚙️ Installation (Local Setup)

```bash
git clone https://github.com/adeelahmedshaikh/time-series-forecasting-app.git
cd time-series-forecasting-app

python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

pip install -r requirements.txt
streamlit run app.py
