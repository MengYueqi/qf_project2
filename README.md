# DSA 5205 Project 2

## Installation

Prerequisites

Python 3.10 or higher

```
pip install pandas numpy matplotlib scikit-learn requests
pip install yfinance pandas-datareader
pip install torch torchvision torchaudio tqdm
pip install pywavelets  # for wavelet denoising

```



## File Construction

```
├── download (this file will be generated after running ```download.py```)
├── download.py
├── GRU/
│   └──prediction_spilt.ipynb
├──LSTM/
│   └── LSTM.py 
├── xlstm_ts/
│   └── xlstm.py
├── Strategy(RL)
│   ├── log
│   ├── model
│   │   ├── __pycache__
│   │   ├── data_loader.py        # data loading script
│   │   ├── eval_agent.py         # Evaluation agent code
│   │   ├── ppo_agent.py          # Core code of the agent model based on the PPO algorithm
│   │   ├── real_env.py           # Realistic environment simulation
│   │   └── train.py              # Model training script
│   └── test_data
├── pairstrading/
│   ├── loaddata.py               # load & align data
│   ├── selectpairs.py            # cointegration test
│   ├── backtest.py               # backtest using predicted spreads
│   ├── buyandhold.py             # long–short benchmark
│   ├── metrics.py                # performance statistics
│   ├── plotpairs.py              # visualization
│   ├── main.ipynb                # end-to-end pipeline notebook
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

## xLSTM-TS Model

run the code ```xlstm.py``` to use the xLSTM-TS model and get the visualization
* For better representation I put the code in the ```xlstm_ts file```, so I strongly suggest you to run ```download.py``` in this file or use ```mv``` and other copy method to move the ```download file``` into ```xlstm_ts file```
Then you can run the shell:

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

Every ticker-output csv include two columns: Date and PredictedPrice.
And a summary csv ```OOS_summary.csv``` performs the evaluation of the model, include:
```
Ticker | MES | R2_TEST | IC | HitRate | Sharpe | TEST_start | TEST_end | #TEST_days
```

## GRU Model

Our GRU module forecasts **next-day closing prices** for the tech stocks using a multi-feature, sequence model.

### What the notebook does

For each ticker, the notebook:

1. **Builds features** from daily OHLCV data (returns, momentum/technical indicators, volatility and volume signals).
   All features are lagged by one trading day to avoid look-ahead.
2. **Splits the sample in time**

   - Train: up to 2020-12-31
   - Validation: 2021-01-01 to 2022-12-31
   - Test (out-of-sample): 2023-01-01 onwards
3. **Trains a 2-layer GRU** (sequence length ≈ 20 days) with an early-stopping rule on the validation loss.
4. **Produces a price forecast path** for the test window by:

   - predicting the next-day direction internally,
   - mapping that signal into a small next-day move,
   - chaining the moves into a synthetic close-price series.

   > We only use and report **price-level forecasts and metrics**; the intermediate return step is not the final target.

### Outputs

Running the notebook populates `GRU/output/` with:

- `{TICKER}.csv` – predicted close prices for the test period.
- `{TICKER}_TEST_curve.png` – actual vs predicted close price plot.
- `OOS_summary.csv` – per-ticker out-of-sample metrics (price MSE, price-level (R^2), price-level IC, hit rate on price direction, and a simple directional Sharpe).
- `diag_price_all/` – optional diagnostic plots for rolling hit-rate, IC, MSE and relative price error.

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

The main notebook first loads two datasets:

- **`close.csv`**  
  Actual daily close prices for all tickers used in the project.  
  Contains a `Date` column and one column per stock.

- **`predicted.csv`**  
  Next-day predicted close prices generated from the model outputs in  
  `LSTM/outputs/`, 'xlstm/outputs/' and `GRU/outputs/`.  
  For each data input, predictions from the three models are averaged.

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

## Trading strategies based on the PPO algorithm

### Running method

To run the PPO-based trading strategies, follow the steps below.

### **1. Create and activate the Conda environment**

```bash
conda create -n ppo_trading python=3.10 -y
conda activate ppo_trading
```

### **2. Install all dependencies using pip**

You can find the requirements file in the `strategy/RL/requirements.txt`, you can use following command to install the requirements:

```bash
pip install -r requirements.txt
```

### **3. Navigate to the PPO strategy directory**

All RL modules are located in:

```
qf_project2/strategy/RL
```

However, **you must run all scripts from the project root** to ensure path correctness.

### **4. Run the PPO training script**

You can use following command to train the model:

```bash
python strategy/RL/model/train.py
```

We use the data at `strategy/RL/data_zjh` to train the RL model. You can use the redirct command to redirct the training log, like:

```bash
python strategy/RL/model/train.py > strategy/RL/log/zjh.log
```

During training, you will see the logs similar to:

```log
[Iter 01] avg_step_reward= -0.3954  policy_loss=0.0221  value_loss=92.2691  entropy=7.0913  ret_mean=-7.7123
[Iter 02] avg_step_reward= -0.5117  policy_loss=0.0064  value_loss=81.4336  entropy=7.0905  ret_mean=-10.6071
[Iter 03] avg_step_reward= -0.4817  policy_loss=0.0246  value_loss=86.3729  entropy=7.0924  ret_mean=-15.5884
[Iter 04] avg_step_reward= -0.5152  policy_loss=0.0037  value_loss=81.3029  entropy=7.0941  ret_mean=-20.5431
[Iter 05] avg_step_reward= -0.4649  policy_loss=0.0447  value_loss=52.9585  entropy=7.0959  ret_mean=-20.3484
[Iter 06] avg_step_reward= -0.4177  policy_loss=-0.0353  value_loss=69.9531  entropy=7.0977  ret_mean=-22.0404
[Iter 07] avg_step_reward= -0.2686  policy_loss=0.0336  value_loss=74.1395  entropy=7.0938  ret_mean=-22.2068
```

The PPO loop will continue printing each epoch’s policy loss, value loss, entropy, and ret_mean. At the end of training, a final summary line will appear, such as:

```log
=== EVAL RESULT (Strategy vs. Benchmarks) ===
Steps traded: 249
[Strategy]   CumPnL=0.200373 | Sharpe*=1.149 | MaxDD=0.182365
[Buy&Hold]          SPY | CumPnL=0.250830 | Sharpe*=1.769 | MaxDD=0.123319
[Buy&Hold] S&P500 (^GSPC) | CumPnL=0.270771 | Sharpe*=1.899 | MaxDD=0.120807
```

You can view the model performance by this matrix.
