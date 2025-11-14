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

run the code xlstm.py to use the xLSTM-TS model and get the visualization

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


