# DSA 5205 Project 2
## Installation
Prerequisites

Python 3.10 or higher

```
pip install pandas numpy matplotlib scikit-learn requests
pip install yfinance pandas-datareader
pip install torch torchvision torchaudio
pip install pywavelets  # for wavelet denoising
```

## File Construction
```
../
├── download
├── download,py
├── GRU
├── LSTM
├── xlstm
├── Strategy(RL)
├── pairtrading
├── README.md
├── .gitclone
```


## Data Download

The data download module fetches historical stock price data from Yahoo Finance for the following 10 technology stocks:
Apple (AAPL)
Microsoft (MSFT)
Alphabet (GOOGL)
Amazon (AMZN)
NVIDIA (NVDA)
Meta Platforms (META)
Netflix (NFLX)
Broadcom (AVGO)
Oracle (ORCL)
Tesla (TSLA)

run the code download.py to download or the required data.

### Raw Data Structure
After running `download.py`, the generated `download` folder will contain:
```
download/
├── all_factors_complete.csv
├── close_price.csv
├── fundamental_data.csv
├── price_data_full.csv
├── technical_factors.csv
```

Each CSV file includes:
- Date
- Open, High, Low, Close prices
- Adjusted Close
- Volume
- Dividends (if any)
- Stock Splits (if any)

### Time Period

- **Start Date**: January 1, 2010
- **End Date**: October 23, 2025 (you will get most recent available date)
- **Frequency**: Daily

### Data Source

All data is sourced from Yahoo Finance via the `yfinance` Python library

## xLSTM Model

run the code ```xlstm.py``` to use the xLSTM-TS model and get the visualization

```
python xlstm.py
```

The code will generate a file named xlstm_output, containing predicted stock data and visualization

```
download/
├── AAPL.csv
├── AAPL_TEST_curve.png
├── MSFT.csv
├── MSFT_TEST_curve.png
...
```

## GRU Model

Our GRU module forecasts **next-day closing prices** for the tech stocks using a multi-feature, sequence model.

### Files and dependencies

- Main notebook: `GRU/prediction_split.ipynb`
- Input data: `download/price_data_full.csv` (two-level header `[PriceField, Ticker]`)
- Outputs: written to `GRU/output/`

In addition to the base packages listed above, the GRU notebook uses:

```bash
pip install torch torchvision torchaudio scikit-learn tqdm
````

### What the notebook does

For each ticker, the notebook:

1. **Builds features** from daily OHLCV data (returns, momentum/technical indicators, volatility and volume signals).
   All features are lagged by one trading day to avoid look-ahead.
2. **Splits the sample in time**

   * Train: up to 2020-12-31
   * Validation: 2021-01-01 to 2022-12-31
   * Test (out-of-sample): 2023-01-01 onwards
3. **Trains a 2-layer GRU** (sequence length ≈ 20 days) with an early-stopping rule on the validation loss.
4. **Produces a price forecast path** for the test window by:

   * predicting the next-day direction internally,
   * mapping that signal into a small next-day move,
   * chaining the moves into a synthetic close-price series.

   > We only use and report **price-level forecasts and metrics**; the intermediate return step is not the final target.

### How to run

Open `GRU/prediction_split.ipynb` in Jupyter/VS Code and run all cells in order.

### Outputs

Running the notebook populates `GRU/output/` with:

* `{TICKER}.csv` – predicted close prices for the test period.
* `{TICKER}_TEST_curve.png` – actual vs predicted close price plot.
* `OOS_summary.csv` – per-ticker out-of-sample metrics (price MSE, price-level (R^2), price-level IC, hit rate on price direction, and a simple directional Sharpe).
* `diag_price_all/` – optional diagnostic plots for rolling hit-rate, IC, MSE and relative price error.


