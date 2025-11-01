import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pywt
from pathlib import Path
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
warnings.filterwarnings('ignore')

# ============================================
# 配置
# ============================================
class Config:
    # 路径
    DATA_DIR = Path("download")
    OUTPUT_DIR = Path("xlstm_pure_pred")
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    # 股票选择
    TICKERS = ["AAPL", "MSFT", "GOOGL", "META", "AMZN", 
               "NVDA", "TSLA", "NFLX", "AVGO", "ORCL"]
    
    # 时间分割
    TRAIN_END = '2020-12-31'
    # VAL_END = '2022-12-31'
    VAL_END = '2022-11-17'
    
    # 小波去噪
    WAVELET = 'db4'
    WAVELET_LEVEL = 1
    
    # xLSTM模型参数
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2
    DROPOUT = 0.2
    SEQUENCE_LENGTH = 30
    
    # 训练参数
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 50
    PATIENCE = 10
    
    # 设备
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42

# 设置随机种子
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(Config.SEED)

print(f"testing set: after 2022")

# ============================================
# sLSTM Cell 实现
# ============================================
class sLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(sLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x, states):
        h_prev, c_prev, n_prev, m_prev = states
        combined = torch.cat([x, h_prev], dim=1)
        
        i_t = torch.exp(self.W_i(combined))
        f_t = torch.sigmoid(self.W_f(combined))
        o_t = torch.sigmoid(self.W_o(combined))
        z_t = torch.tanh(self.W_z(combined))
        
        c_t = f_t * c_prev + i_t * z_t
        n_t = f_t * n_prev + i_t
        m_t = torch.max(f_t * m_prev, torch.abs(i_t * z_t))
        
        h_t = o_t * (c_t / (n_t + 1e-6))
        h_t = self.layer_norm(h_t)
        
        return h_t, (h_t, c_t, n_t, m_t)

class xLSTMLayer(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, dropout=0.0):
        super(xLSTMLayer, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            cell_input_size = input_size if i == 0 else hidden_size
            self.cells.append(sLSTMCell(cell_input_size, hidden_size))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        
    def forward(self, x, states=None):
        batch_size, seq_len, _ = x.size()
        
        if states is None:
            states = []
            for _ in range(self.num_layers):
                h = torch.zeros(batch_size, self.hidden_size, device=x.device)
                c = torch.zeros(batch_size, self.hidden_size, device=x.device)
                n = torch.zeros(batch_size, self.hidden_size, device=x.device)
                m = torch.zeros(batch_size, self.hidden_size, device=x.device)
                states.append((h, c, n, m))
        
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            
            for layer_idx, cell in enumerate(self.cells):
                h_t, states[layer_idx] = cell(x_t, states[layer_idx])
                x_t = h_t
                
                if self.dropout is not None and layer_idx < self.num_layers - 1:
                    x_t = self.dropout(x_t)
            
            outputs.append(h_t)
        
        output = torch.stack(outputs, dim=1)
        return output, states

class xLSTM_TS(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size=1):
        super(xLSTM_TS, self).__init__()
        
        self.xlstm = xLSTMLayer(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )
        
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        xlstm_out, _ = self.xlstm(x)
        out = self.fc(xlstm_out[:, -1, :])
        return out

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y, seq_length):
        self.X = X
        self.y = y
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.X) - self.seq_length
    
    def __getitem__(self, idx):
        X_seq = self.X[idx:idx+self.seq_length]
        y_val = self.y[idx+self.seq_length]
        
        X_tensor = torch.FloatTensor(X_seq)
        y_tensor = torch.FloatTensor([y_val])
        
        return X_tensor, y_tensor

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []
    
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item()
            predictions.extend(outputs.cpu().numpy().flatten())
            actuals.extend(y_batch.cpu().numpy().flatten())
    
    return total_loss / len(loader), np.array(predictions), np.array(actuals)

def wavelet_denoise(data, wavelet='db4', level=1):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(data)))
    coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)

def prepare_stock_data(ticker, all_factors, close_prices):
    print(f"preparing {ticker}  data...")
    
    ticker_cols = [col for col in all_factors.columns if col.startswith(f'{ticker}_')]
    
    if len(ticker_cols) == 0:
        print(f"warning: cannot find {ticker} features!")
        return None
    
    data = all_factors[ticker_cols].copy()
    
    if ticker not in close_prices.columns:
        print(f"warning: cannot find {ticker} close!")
        return None
    
    target_returns = close_prices[ticker].pct_change().shift(-1)
    data['target'] = target_returns
    data['close_price'] = close_prices[ticker]
    
    data = data.dropna()
    
    if len(data) < 100:
        print(f"low data quantity ({len(data)}days), try to use technical features...")
        
        fundamental_keywords = ['PE_', 'PB_', 'PS_', 'ROE', 'ROA', 'Debt', 'Dividend', 
                               'Earnings', 'Revenue', 'Profit', 'Operating', 'Market_Cap', 'Beta',
                               '52Week']
        
        tech_cols = [col for col in ticker_cols if not any(
            kw in col for kw in fundamental_keywords
        )]
        
        if len(tech_cols) > 0:
            data_tech = all_factors[tech_cols].copy()
            data_tech['target'] = target_returns
            data_tech['close_price'] = close_prices[ticker]
            data_tech = data_tech.dropna()
            
            if len(data_tech) >= 100:
                print(f"technical features are successfull")
                print(f"number of features: {len(tech_cols)} (only techinical features)")
                print(f"date range: {data_tech.index[0].date()} to {data_tech.index[-1].date()}")
                return data_tech
        
        return None
    
    return data

def train_single_stock(ticker, stock_data):
    print(f"train {ticker}'s xLSTM-TS model")
    
    train_data = stock_data[:Config.TRAIN_END]
    val_data = stock_data[Config.TRAIN_END:Config.VAL_END]
    test_data = stock_data[Config.VAL_END:]
    
    feature_cols = [col for col in stock_data.columns if col not in ['target', 'close_price']]
    
    X_train = train_data[feature_cols].values
    y_train = train_data['target'].values
    
    X_val = val_data[feature_cols].values
    y_val = val_data['target'].values
    
    X_test = test_data[feature_cols].values
    y_test = test_data['target'].values
    
    test_close_prices = test_data['close_price'].values
    test_dates = test_data.index
    
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    print("wavelet denoising on training targets...")
    y_train_denoised = wavelet_denoise(y_train_scaled, Config.WAVELET, Config.WAVELET_LEVEL)
    
    train_dataset = TimeSeriesDataset(X_train_scaled, y_train_denoised, Config.SEQUENCE_LENGTH)
    val_dataset = TimeSeriesDataset(X_val_scaled, y_val_scaled, Config.SEQUENCE_LENGTH)
    test_dataset = TimeSeriesDataset(X_test_scaled, y_test_scaled, Config.SEQUENCE_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    input_size = X_train_scaled.shape[1]
    model = xLSTM_TS(
        input_size=input_size,
        hidden_size=Config.HIDDEN_SIZE,
        num_layers=Config.NUM_LAYERS,
        dropout=Config.DROPOUT
    ).to(Config.DEVICE)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    print(f"\nstart training...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        val_loss, _, _ = validate(model, val_loader, criterion, Config.DEVICE)
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{Config.NUM_EPOCHS} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), Config.OUTPUT_DIR / f'{ticker}_best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= Config.PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    model.load_state_dict(torch.load(Config.OUTPUT_DIR / f'{ticker}_best_model.pth'))
    
    print("\n predicting on testing set...")
    test_loss, y_pred_scaled, y_test_actual_scaled = validate(model, test_loader, criterion, Config.DEVICE)
    
    y_pred_returns = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_test_returns = scaler_y.inverse_transform(y_test_actual_scaled.reshape(-1, 1)).flatten()
    
    actual_test_dates = test_dates[Config.SEQUENCE_LENGTH:]
    actual_test_close = test_close_prices[Config.SEQUENCE_LENGTH:]
    
    predicted_close = actual_test_close * (1 + y_pred_returns)
    actual_next_close = actual_test_close * (1 + y_test_returns)
    
    mse = mean_squared_error(y_test_returns, y_pred_returns)
    mae = mean_absolute_error(y_test_returns, y_pred_returns)
    r2 = r2_score(y_test_returns, y_pred_returns)
    direction_accuracy = np.mean((y_pred_returns > 0) == (y_test_returns > 0))
    
    print(f"\n evaluated index:")
    print(f"  MSE: {mse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R^2: {r2:.4f}")
    print(f"  Accuracy of direction: {direction_accuracy*100:.2f}%")
    
    results_df = pd.DataFrame({
        'Date': actual_test_dates,
        'Ticker': ticker,
        'Actual_Close': actual_next_close,
        'Predicted_Close': predicted_close,
    })
    
    output_path = Config.OUTPUT_DIR / f'{ticker}_predictions.csv'
    results_df.to_csv(output_path, index=False)
    print(f"\n✅ results saved: {output_path}")
    
    return {
        'dates': actual_test_dates,
        'actual_close': actual_next_close,
        'predicted_close': predicted_close,
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'direction_accuracy': direction_accuracy
    }

def main():
    
    all_factors = pd.read_csv(Config.DATA_DIR / "all_factors_complete.csv", 
                             index_col=0, parse_dates=True)
    close_prices = pd.read_csv(Config.DATA_DIR / "close_prices.csv",
                               index_col=0, parse_dates=True)
    
    results = {}
    
    for ticker in Config.TICKERS:
        try:
            stock_data = prepare_stock_data(ticker, all_factors, close_prices)
            
            if stock_data is None:
                print(f"jump {ticker}")
                continue
            
            result = train_single_stock(ticker, stock_data)
            results[ticker] = result
            
        except Exception as e:
            print(f"\n {ticker} fail to predict: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n successfully generate {len(results)}/{len(Config.TICKERS)} stocks predictions")
    
    all_predictions = []
    for ticker in results.keys():
        csv_path = Config.OUTPUT_DIR / f'{ticker}_predictions.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            all_predictions.append(df)
    
    all_predictions_df = pd.concat(all_predictions, ignore_index=True)
    all_predictions_path = Config.OUTPUT_DIR / 'ALL_STOCKS_predictions.csv'
    all_predictions_df.to_csv(all_predictions_path, index=False)

    pivot_actual = all_predictions_df.pivot_table(
        index='Date', 
        columns='Ticker', 
        values='Actual_Close'
    )
    pivot_actual_path = Config.OUTPUT_DIR / 'actual_close_matrix.csv'
    pivot_actual.to_csv(pivot_actual_path)
    
    pivot_predicted = all_predictions_df.pivot_table(
        index='Date', 
        columns='Ticker', 
        values='Predicted_Close'
    )
    pivot_predicted_path = Config.OUTPUT_DIR / 'predicted_close_matrix.csv'
    pivot_predicted.to_csv(pivot_predicted_path)
    
    report_data = []
    for ticker, result in results.items():
        report_data.append({
            'Ticker': ticker,
            'Direction_Accuracy': f"{result['direction_accuracy']*100:.2f}%",
            'R_Squared': f"{result['r2']:.4f}",
            'MAE': f"{result['mae']:.6f}",
            'MSE': f"{result['mse']:.6f}"
        })
    
    report_df = pd.DataFrame(report_data)
    report_path = Config.OUTPUT_DIR / 'prediction_quality_report.csv'
    report_df.to_csv(report_path, index=False)
    print(f"\n summary report saved: {report_path}")

if __name__ == "__main__":
    main()