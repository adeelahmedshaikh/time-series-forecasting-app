import yfinance as yf

df = yf.download("AAPL", start="2020-01-01", end="2024-01-01")
df = df.reset_index()[["Date", "Close"]]
df.columns = ["date", "value"]
df.to_csv("data/sales.csv", index=False)

print("Saved real AAPL stock data to data/sales.csv")