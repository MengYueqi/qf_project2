# DSA 5205 Project 2
## Installation
Prerequisites

Python 3.10 or higher

pip install pandas numpy matplotlib scikit-learn requests
pip install yfinance pandas-datareader
pip install torch torchvision torchaudio
pip install pywavelets  # for wavelet denoising

## Data download

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
After running `download.py`, the generated `download` folder will contain:
```
download/
├── AAPL.csv
├── MSFT.csv
├── GOOGL.csv
...
```
