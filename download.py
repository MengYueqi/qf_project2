import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
import time
import os
from pathlib import Path
warnings.filterwarnings('ignore')


current_dir = Path(__file__).parent if '__file__' in globals() else Path.cwd()

download_dir = current_dir / 'download'
download_dir.mkdir(parents=True, exist_ok=True)

print(f"save: {download_dir.absolute()}\n")

tickers = ["AAPL", "MSFT", "GOOGL", "META", "AMZN", 
           "NVDA", "TSLA", "NFLX", "AVGO", "ORCL"]
benchmarks = ["SPY", "QQQ"]
all_symbols = tickers + benchmarks

start_date = "2010-01-01"
end_date = datetime.now().strftime("%Y-%m-%d")

BASE_DELAY = 2  
MAX_RETRIES = 3 

print("downloading...")
price_data = yf.download(
    all_symbols,
    start=start_date,
    end=end_date,
    interval='1d',
    auto_adjust=True,
    progress=True
)

close = price_data['Close']
open_price = price_data['Open']
high = price_data['High']
low = price_data['Low']
volume = price_data['Volume']
print("\n downloading basic data...")

def download_fundamental_with_retry(ticker, max_retries=MAX_RETRIES, base_delay=BASE_DELAY):

    for attempt in range(max_retries):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            time.sleep(base_delay)
            
            return {
                'PE_Ratio': info.get('trailingPE', np.nan),
                'Forward_PE': info.get('forwardPE', np.nan),
                'PB_Ratio': info.get('priceToBook', np.nan),
                'PS_Ratio': info.get('priceToSalesTrailing12Months', np.nan),
                'ROE': info.get('returnOnEquity', np.nan),
                'ROA': info.get('returnOnAssets', np.nan),
                'Debt_to_Equity': info.get('debtToEquity', np.nan),
                'Dividend_Yield': info.get('dividendYield', np.nan),
                'Earnings_Growth': info.get('earningsGrowth', np.nan),
                'Revenue_Growth': info.get('revenueGrowth', np.nan),
                'Profit_Margin': info.get('profitMargins', np.nan),
                'Operating_Margin': info.get('operatingMargins', np.nan),
                'Market_Cap': info.get('marketCap', np.nan),
                'Beta': info.get('beta', np.nan),
                '52Week_High': info.get('fiftyTwoWeekHigh', np.nan),
                '52Week_Low': info.get('fiftyTwoWeekLow', np.nan),
            }
            
        except Exception as e:
            error_msg = str(e)
            if "Too Many Requests" in error_msg or "Rate limited" in error_msg or "429" in error_msg:

                wait_time = (2 ** attempt) * base_delay
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                else:
                    return {}
            else:
                return {}
    
    return {}

fundamental_data = {}
for ticker in all_symbols:
    fundamental_data[ticker] = download_fundamental_with_retry(ticker)



fundamentals_df = pd.DataFrame(fundamental_data).T
fundamentals_df.to_csv(download_dir / "fundamental_data.csv")


def calculate_factors(close, high, low, volume, open_price):

    factors = pd.DataFrame(index=close.index)
    
    for ticker in close.columns:
        
        c = close[ticker]
        h = high[ticker]
        l = low[ticker]
        v = volume[ticker]
        o = open_price[ticker]
        

        factors[f'{ticker}_ret_1d'] = c.pct_change(1)
        factors[f'{ticker}_ret_5d'] = c.pct_change(5)
        factors[f'{ticker}_ret_10d'] = c.pct_change(10)
        factors[f'{ticker}_ret_20d'] = c.pct_change(20)
        factors[f'{ticker}_ret_60d'] = c.pct_change(60)
        factors[f'{ticker}_ret_120d'] = c.pct_change(120)
        

        mom_20 = c.pct_change(20)
        factors[f'{ticker}_mom_accel'] = mom_20 - mom_20.shift(20)
        
        factors[f'{ticker}_vol_20d'] = c.pct_change().rolling(20).std()
        factors[f'{ticker}_vol_60d'] = c.pct_change().rolling(60).std()
        

        hl_ratio = np.log(h / l)
        factors[f'{ticker}_parkinson_vol'] = hl_ratio.rolling(20).std() * np.sqrt(1/(4*np.log(2)))
        

        vol_short = c.pct_change().rolling(20).std()
        vol_long = c.pct_change().rolling(60).std()
        factors[f'{ticker}_vol_ratio'] = vol_short / vol_long
        

        factors[f'{ticker}_volume_ratio'] = v / v.rolling(20).mean()
        

        factors[f'{ticker}_volume_change'] = v.pct_change(5)
        

        factors[f'{ticker}_volume_mom'] = (v.rolling(5).mean() / 
                                           v.rolling(20).mean())
        

        price_mom = c.pct_change(10)
        vol_mom = v.pct_change(10)
        factors[f'{ticker}_price_volume_div'] = price_mom - vol_mom
        

        delta = c.diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        factors[f'{ticker}_rsi'] = 100 - (100 / (1 + rs))
        
        ma5 = c.rolling(5).mean()
        ma10 = c.rolling(10).mean()
        ma20 = c.rolling(20).mean()
        ma60 = c.rolling(60).mean()
        
        factors[f'{ticker}_ma5_bias'] = (c - ma5) / ma5
        factors[f'{ticker}_ma20_bias'] = (c - ma20) / ma20
        
        factors[f'{ticker}_ma_alignment'] = ((ma5 > ma10) & 
                                             (ma10 > ma20) & 
                                             (ma20 > ma60)).astype(int) * 2 - 1
        
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        factors[f'{ticker}_macd'] = macd
        factors[f'{ticker}_macd_signal'] = signal
        factors[f'{ticker}_macd_hist'] = macd - signal
        
        bb_middle = c.rolling(20).mean()
        bb_std = c.rolling(20).std()
        bb_upper = bb_middle + 2 * bb_std
        bb_lower = bb_middle - 2 * bb_std
        factors[f'{ticker}_bb_position'] = (c - bb_lower) / (bb_upper - bb_lower)
        
        tr1 = h - l
        tr2 = abs(h - c.shift(1))
        tr3 = abs(l - c.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        factors[f'{ticker}_atr'] = tr.rolling(14).mean()
        
        daily_ret = c.pct_change().abs()
        dollar_volume = v * c
        factors[f'{ticker}_amihud'] = (daily_ret / dollar_volume).rolling(20).mean()
        
        factors[f'{ticker}_hl_spread'] = (h - l) / c
        
        cov = c.diff().rolling(2).cov()
        factors[f'{ticker}_roll_spread'] = 2 * np.sqrt(-cov.clip(upper=0))
        
        high_52w = h.rolling(252).max()
        low_52w = l.rolling(252).min()
        factors[f'{ticker}_52w_position'] = (c - low_52w) / (high_52w - low_52w)
        
        factors[f'{ticker}_from_52w_high'] = (c / high_52w) - 1
        
        factors[f'{ticker}_overnight_ret'] = (o / c.shift(1)) - 1
        
        factors[f'{ticker}_intraday_ret'] = (c / o) - 1
        
        factors[f'{ticker}_gap'] = (o / c.shift(1)) - 1
        
    return factors

all_factors = calculate_factors(close, high, low, volume, open_price)

price_data.to_csv(download_dir / "price_data_full.csv")
close.to_csv(download_dir / "close_prices.csv")

all_factors.to_csv(download_dir / "technical_factors.csv")

for ticker in tickers:
    for col in fundamentals_df.columns:
        all_factors[f'{ticker}_{col}'] = fundamentals_df.loc[ticker, col]

all_factors.to_csv(download_dir / "all_factors_complete.csv")
print("download complete.")