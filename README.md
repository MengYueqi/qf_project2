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
```
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
```

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

## CNN-BiLSTM Model
A deep learning system for stock price prediction using CNN-BiLSTM with online calibration.

### Quickstart

Run the code to train the model and generate predictions:
```bash
python LSTM.py
```
The code will process all tickers in the dataset and generate predictions with visualizations in the `output_final` directory.


### Output

The code generates the following files in `output_final/`:

```
output_final/
├── AAPL.csv                    # Predicted prices for AAPL
├── AAPL_TEST_curve.png         # Actual vs Predicted visualization
├── MSFT.csv                    # Predicted prices for MSFT
├── MSFT_TEST_curve.png         # Actual vs Predicted visualization
├── ...
└── OOS_summary.csv             # Performance summary for all tickers
```
### Performance Metrics

The `OOS_summary.csv` file contains:
- **MSE**: Mean squared error
- **R2_TEST**: Out-of-sample R² score
- **IC**: Information coefficient (correlation)
- **HitRate**: Directional accuracy
- **Sharpe**: Annualized Sharpe ratio


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

### Outputs

Running the notebook populates `GRU/output/` with:

* `{TICKER}.csv` – predicted close prices for the test period.
* `{TICKER}_TEST_curve.png` – actual vs predicted close price plot.
* `OOS_summary.csv` – per-ticker out-of-sample metrics (price MSE, price-level (R^2), price-level IC, hit rate on price direction, and a simple directional Sharpe).
* `diag_price_all/` – optional diagnostic plots for rolling hit-rate, IC, MSE and relative price error.
* 



## Pairs Trading

### Python files and notebook

The module is organized as one main notebook plus several small Python files:

- **`main.ipynb`**  
  Orchestrates the full workflow:
  - loads data,  
  - builds the cointegration universe,  
  - defines `pairs_to_use`,  
  - runs the backtest and benchmark,  
  - computes metrics and produces plots.

- **`loaddata.py`**  
  Helper functions to load and align:
  - historical close prices,  
  - ML-predicted next-day prices (mean of 3 models).

- **`selectpairs.py`**  
  Implements pair selection:
  - filters by overlapping dates and minimum sample length,  
  - runs the Engle–Granger cointegration test,  
  - outputs a `pair_df` with stock1, stock2, β, μ, σ, p-values.

- **`backtest.py`**  
  Implements the mean-reversion trading strategy

- **`buyandhold.py`**  
  Defines the static benchmark:
  - long 1 unit of stock1 and short β units of stock2,  
  - returns daily and cumulative log returns of this long–short portfolio.

- **`metrics.py`**  
  Collects performance metrics

- **`plotpairs.py`**  
   Plots strategy vs buy-and-hold benchmark,


### What the main notebook does

For each selected pair, the main notebook:

1. **Builds the cointegration universe** using daily log close prices  
   - keeps only overlapping dates for each pair,  
   - applies the Engle–Granger test,  
   - estimates the hedge ratio (β), spread mean (μ), and spread volatility (σ).

2. **Loads ML-predicted prices** for the out-of-sample period.  

3. **Computes predicted spreads and z-scores**, using:
   - predicted next-day prices,  
   - historical β, μ, σ from the training window.  

4. **Runs a rule-based trading strategy**:
   - short the spread when \(z > z_{\text{entry}}\),  
   - long the spread when \(z < -z_{\text{entry}}\),  
   - close positions when \(|z| < z_{\text{exit}}\).

5. **Constructs a static buy-and-hold benchmark**: long 1 unit of stock1 and short β units of stock2.

6. **Computes performance metrics**, including:
   - cumulative return,  
   - annualized return and volatility,  
   - Sharpe ratio,  
   - maximum drawdown,  
   - win rate.

7. **Generates plots** for strategy vs benchmark comparison.

### Outputs

Running `main.ipynb` produces:
   - `pair_df` – selected cointegrated pairs with β, μ, σ and p-values.
   - **Performance summary tables** – strategy vs benchmark metrics across pairs.  
   - **Plots** – return curves and visual comparisons of the strategy and benchmark.











