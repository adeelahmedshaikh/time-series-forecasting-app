import pandas as pd


def detect_date_column(df):
    date_keywords = ["date", "time", "day", "month", "year", "timestamp"]

    for col in df.columns:
        if any(keyword in col.lower() for keyword in date_keywords):
            return col

    for col in df.columns:
        try:
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() > 0.8:
                return col
        except Exception:
            continue

    return None


def detect_target_column(df):
    target_keywords = [
        "sales", "revenue", "value", "demand",
        "orders", "quantity", "units", "price", "close"
    ]

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

    for col in numeric_cols:
        if any(keyword in col.lower() for keyword in target_keywords):
            return col

    if numeric_cols:
        return numeric_cols[0]

    return None


def standardize_columns(df, date_col, target_col):
    clean_df = df[[date_col, target_col]].copy()
    clean_df.columns = ["date", "value"]

    clean_df["date"] = pd.to_datetime(clean_df["date"], errors="coerce")
    clean_df["value"] = pd.to_numeric(clean_df["value"], errors="coerce")

    clean_df = clean_df.dropna()
    clean_df = clean_df.sort_values("date")

    return clean_df

import yfinance as yf


def fetch_yahoo_data(ticker, period):
    data = yf.download(ticker, period=period)

    if data.empty:
        return None

    data = data.reset_index()

    if "Close" not in data.columns:
        return None

    df = data[["Date", "Close"]].copy()
    df.columns = ["date", "value"]

    df["date"] = pd.to_datetime(df["date"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna()

    return df