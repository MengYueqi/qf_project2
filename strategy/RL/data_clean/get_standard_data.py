# fetch_sp500_and_spy.py
# pip install yfinance pandas

import datetime as dt
import pandas as pd
import yfinance as yf

def fetch_to_csv(ticker: str, out_csv: str, start_date: str = "2020-01-01"):
    """
    下载指定 ticker 的复权收盘价，从 start_date 到今天，保存为两列 CSV: Date, Price
    """
    end_date = dt.date.today().isoformat()

    # auto_adjust=True -> 返回的 Close 已经是分红/拆分调整后的价格（推荐用于回测/基准比较）
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
    )

    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}. Check ticker or connection.")

    out = (
        df[['Close']]
        .rename(columns={'Close': 'Price'})
        .reset_index()  # Date 作为列
    )

    # 统一日期格式为 YYYY-MM-DD
    out['Date'] = pd.to_datetime(out['Date']).dt.date.astype(str)

    out.to_csv(out_csv, index=False)
    print(f"Saved {ticker} -> {out_csv} | rows={len(out)} | first={out['Date'].iloc[0]} last={out['Date'].iloc[-1]}")

def main():
    # S&P 500 指数（Yahoo Finance 代码 ^GSPC）
    fetch_to_csv("^GSPC", "sp500.csv", start_date="2023-01-03")

    # SPY ETF
    fetch_to_csv("SPY", "spy.csv", start_date="2023-01-03")

if __name__ == "__main__":
    main()
