#CNN-LSTM combination
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


class Config:
    # 路径
    DATA_DIR = Path("download")
    OUTPUT_DIR = Path("output_cnn_lstm")
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    
    # 股票选择
    TICKERS = ["AAPL", "MSFT", "GOOGL", "META", "AMZN", 
               "NVDA", "TSLA", "NFLX", "AVGO", "ORCL"]
    
    # 时间分割
    TRAIN_END = '2020-12-31'
    VAL_END = '2022-11-17'
    
    # 小波去噪
    WAVELET = 'db4'
    WAVELET_LEVEL = 1
    
    # CNN-LSTM模型参数
    CNN_FILTERS = [64, 128, 256]    # CNN卷积核数量
    KERNEL_SIZES = [3, 5, 7]        # 不同尺度的卷积核
    LSTM_HIDDEN = 128
    LSTM_LAYERS = 2
    DROPOUT = 0.3
    SEQUENCE_LENGTH = 30
    
    # 训练参数
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    PATIENCE = 15
    
    # 设备
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42

# 设置随机种子
torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(Config.SEED)

print(f"Using device: {Config.DEVICE}")
print(f"Testing set: after {Config.VAL_END}")

# ============================================
# Multi-Scale CNN Module
# ============================================
class MultiScaleCNN(nn.Module):
    """多尺度1D卷积神经网络 - 捕获不同时间尺度的特征"""
    def __init__(self, input_size, filters, kernel_sizes, dropout=0.2):
        super(MultiScaleCNN, self).__init__()
        
        self.convs = nn.ModuleList()
        
        for kernel_size in kernel_sizes:
            # 每个尺度使用多层卷积
            conv_block = nn.Sequential(
                nn.Conv1d(input_size, filters[0], kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(filters[0]),
                nn.ReLU(),
                nn.Dropout(dropout),
                
                nn.Conv1d(filters[0], filters[1], kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(filters[1]),
                nn.ReLU(),
                nn.Dropout(dropout),
                
                nn.Conv1d(filters[1], filters[2], kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(filters[2]),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.convs.append(conv_block)
        
        # 特征融合层
        self.fusion = nn.Conv1d(filters[2] * len(kernel_sizes), filters[2], 1)
        self.fusion_bn = nn.BatchNorm1d(filters[2])
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        
        # 多尺度特征提取
        conv_outputs = []
        for conv in self.convs:
            conv_out = conv(x)
            conv_outputs.append(conv_out)
        
        # 拼接多尺度特征
        concatenated = torch.cat(conv_outputs, dim=1)  # (batch, filters*len(kernels), seq_len)
        
        # 特征融合
        fused = self.fusion(concatenated)
        fused = self.fusion_bn(fused)
        fused = torch.relu(fused)
        
        fused = fused.transpose(1, 2)  # (batch, seq_len, filters)
        
        return fused

# ============================================
# CNN-BiLSTM 混合模型
# ============================================
class CNNBiLSTM(nn.Module):
    """
    1D-CNN + BiLSTM 混合架构
    - 多尺度CNN提取局部特征和模式
    - BiLSTM捕获双向时序依赖
    """
    def __init__(self, input_size, cnn_filters, kernel_sizes, 
                 lstm_hidden, lstm_layers, dropout, output_size=1):
        super(CNNBiLSTM, self).__init__()
        
        # Multi-scale CNN
        self.cnn = MultiScaleCNN(input_size, cnn_filters, kernel_sizes, dropout)
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=cnn_filters[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism on LSTM outputs
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.Tanh(),
            nn.Linear(lstm_hidden, 1)
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden // 2, output_size)
        )
        
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        
        # CNN feature extraction
        cnn_out = self.cnn(x)  # (batch, seq_len, cnn_filters[-1])
        
        # BiLSTM
        lstm_out, _ = self.lstm(cnn_out)  # (batch, seq_len, lstm_hidden*2)
        
        # Attention mechanism
        attention_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)
        
        # Apply attention
        context = torch.sum(attention_weights * lstm_out, dim=1)  # (batch, lstm_hidden*2)
        
        # Final prediction
        out = self.fc(context)
        
        return out

# ============================================
# 数据处理
# ============================================
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
        
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
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
    print(f"Preparing {ticker} data...")
    
    ticker_cols = [col for col in all_factors.columns if col.startswith(f'{ticker}_')]
    
    if len(ticker_cols) == 0:
        print(f"Warning: Cannot find {ticker} features!")
        return None
    
    data = all_factors[ticker_cols].copy()
    
    if ticker not in close_prices.columns:
        print(f"Warning: Cannot find {ticker} close!")
        return None
    
    target_returns = close_prices[ticker].pct_change().shift(-1)
    data['target'] = target_returns
    data['close_price'] = close_prices[ticker]
    
    data = data.dropna()
    
    if len(data) < 100:
        print(f"Low data quantity ({len(data)} days), trying technical features...")
        
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
                print(f"Technical features successful: {len(tech_cols)} features")
                return data_tech
        
        return None
    
    return data

def evaluate_predictions(y_true_returns, y_pred_returns, dataset_name):
    """评估预测性能"""
    mse = mean_squared_error(y_true_returns, y_pred_returns)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_returns, y_pred_returns)
    r2 = r2_score(y_true_returns, y_pred_returns)
    
    direction_accuracy = np.mean((y_pred_returns > 0) == (y_true_returns > 0)) * 100
    
    mape = np.mean(np.abs((y_true_returns - y_pred_returns) / (np.abs(y_true_returns) + 1e-8))) * 100
    
    metrics = {
        'Dataset': dataset_name,
        'Samples': len(y_true_returns),
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2,
        'MAPE': mape,
        'DirectionAccuracy': direction_accuracy
    }
    
    return metrics

def train_single_stock(ticker, stock_data):
    """训练单只股票并输出完整的性能报告"""
    print(f"\n{'='*80}")
    print(f"Training {ticker}'s CNN-BiLSTM model")
    print(f"{'='*80}")
    
    # 数据分割
    train_data = stock_data[:Config.TRAIN_END]
    val_data = stock_data[Config.TRAIN_END:Config.VAL_END]
    test_data = stock_data[Config.VAL_END:]
    
    print(f"Train: {len(train_data)} samples ({train_data.index[0].date()} to {train_data.index[-1].date()})")
    print(f"Val:   {len(val_data)} samples ({val_data.index[0].date()} to {val_data.index[-1].date()})")
    print(f"Test:  {len(test_data)} samples ({test_data.index[0].date()} to {test_data.index[-1].date()})")
    
    feature_cols = [col for col in stock_data.columns if col not in ['target', 'close_price']]
    
    # 提取特征和目标
    X_train = train_data[feature_cols].values
    y_train = train_data['target'].values
    
    X_val = val_data[feature_cols].values
    y_val = val_data['target'].values
    
    X_test = test_data[feature_cols].values
    y_test = test_data['target'].values
    
    # 保存价格信息
    train_dates = train_data.index
    val_dates = val_data.index
    test_dates = test_data.index
    
    train_close_prices = train_data['close_price'].values
    val_close_prices = val_data['close_price'].values
    test_close_prices = test_data['close_price'].values
    
    # 标准化特征
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)
    X_test_scaled = scaler_X.transform(X_test)
    
    # 标准化目标
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_val_scaled = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1, 1)).flatten()
    
    # 小波去噪
    print("\nApplying wavelet denoising on training targets...")
    y_train_denoised = wavelet_denoise(y_train_scaled, Config.WAVELET, Config.WAVELET_LEVEL)
    
    # 创建数据集
    train_dataset = TimeSeriesDataset(X_train_scaled, y_train_denoised, Config.SEQUENCE_LENGTH)
    val_dataset = TimeSeriesDataset(X_val_scaled, y_val_scaled, Config.SEQUENCE_LENGTH)
    test_dataset = TimeSeriesDataset(X_test_scaled, y_test_scaled, Config.SEQUENCE_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False)
    
    # 构建模型
    input_size = X_train_scaled.shape[1]
    model = CNNBiLSTM(
        input_size=input_size,
        cnn_filters=Config.CNN_FILTERS,
        kernel_sizes=Config.KERNEL_SIZES,
        lstm_hidden=Config.LSTM_HIDDEN,
        lstm_layers=Config.LSTM_LAYERS,
        dropout=Config.DROPOUT
    ).to(Config.DEVICE)
    
    print(f"\nModel architecture:")
    print(f"   Input features: {input_size}")
    print(f"   Multi-scale CNN: filters={Config.CNN_FILTERS}, kernels={Config.KERNEL_SIZES}")
    print(f"   BiLSTM: hidden={Config.LSTM_HIDDEN}, layers={Config.LSTM_LAYERS}")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, 
                                                     patience=5)
    
    # 训练
    print(f"\nStarting training...")
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(Config.NUM_EPOCHS):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, Config.DEVICE)
        val_loss, _, _ = validate(model, val_loader, criterion, Config.DEVICE)
        
        scheduler.step(val_loss)
        
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
    
    # 加载最佳模型
    model.load_state_dict(torch.load(Config.OUTPUT_DIR / f'{ticker}_best_model.pth'))
    
    # 评估所有数据集
    print("\n" + "="*80)
    print("Evaluating model performance on all datasets...")
    print("="*80)
    
    # 训练集评估
    _, y_train_pred_scaled, y_train_actual_scaled = validate(model, train_loader, criterion, Config.DEVICE)
    y_train_pred_returns = scaler_y.inverse_transform(y_train_pred_scaled.reshape(-1, 1)).flatten()
    y_train_actual_returns = scaler_y.inverse_transform(y_train_actual_scaled.reshape(-1, 1)).flatten()
    
    train_metrics = evaluate_predictions(y_train_actual_returns, y_train_pred_returns, "Train")
    
    # 验证集评估
    _, y_val_pred_scaled, y_val_actual_scaled = validate(model, val_loader, criterion, Config.DEVICE)
    y_val_pred_returns = scaler_y.inverse_transform(y_val_pred_scaled.reshape(-1, 1)).flatten()
    y_val_actual_returns = scaler_y.inverse_transform(y_val_actual_scaled.reshape(-1, 1)).flatten()
    
    val_metrics = evaluate_predictions(y_val_actual_returns, y_val_pred_returns, "Validation")
    
    # 测试集评估
    _, y_test_pred_scaled, y_test_actual_scaled = validate(model, test_loader, criterion, Config.DEVICE)
    y_test_pred_returns = scaler_y.inverse_transform(y_test_pred_scaled.reshape(-1, 1)).flatten()
    y_test_actual_returns = scaler_y.inverse_transform(y_test_actual_scaled.reshape(-1, 1)).flatten()
    
    test_metrics = evaluate_predictions(y_test_actual_returns, y_test_pred_returns, "Test")
    
    # 打印性能指标
    print(f"\n✓ Performance metrics:")
    print(f"   Training set   R²={train_metrics['R2']:.4f}, RMSE={train_metrics['RMSE']:.6f}, Direction Acc={train_metrics['DirectionAccuracy']:.2f}%")
    print(f"   Validation set R²={val_metrics['R2']:.4f}, RMSE={val_metrics['RMSE']:.6f}, Direction Acc={val_metrics['DirectionAccuracy']:.2f}%")
    print(f"   Test set       R²={test_metrics['R2']:.4f}, RMSE={test_metrics['RMSE']:.6f}, Direction Acc={test_metrics['DirectionAccuracy']:.2f}%")
    
    # 保存性能指标
    performance_df = pd.DataFrame([train_metrics, val_metrics, test_metrics])
    performance_file = Config.OUTPUT_DIR / f'{ticker}_performance.csv'
    performance_df.to_csv(performance_file, index=False)
    print(f"\n✓ Performance metrics saved: {performance_file}")
    
    # 保存完整对比
    train_actual_dates = train_dates[Config.SEQUENCE_LENGTH:]
    val_actual_dates = val_dates[Config.SEQUENCE_LENGTH:]
    test_actual_dates = test_dates[Config.SEQUENCE_LENGTH:]
    
    train_actual_close = train_close_prices[Config.SEQUENCE_LENGTH:]
    val_actual_close = val_close_prices[Config.SEQUENCE_LENGTH:]
    test_actual_close = test_close_prices[Config.SEQUENCE_LENGTH:]
    
    train_actual_price = train_actual_close * (1 + y_train_actual_returns)
    train_pred_price = train_actual_close * (1 + y_train_pred_returns)
    
    val_actual_price = val_actual_close * (1 + y_val_actual_returns)
    val_pred_price = val_actual_close * (1 + y_val_pred_returns)
    
    test_actual_price = test_actual_close * (1 + y_test_actual_returns)
    test_pred_price = test_actual_close * (1 + y_test_pred_returns)
    
    train_df = pd.DataFrame({
        'Date': train_actual_dates,
        'ActualPrice': train_actual_price,
        'PredictedPrice': train_pred_price,
        'Dataset': 'Train'
    })
    
    val_df = pd.DataFrame({
        'Date': val_actual_dates,
        'ActualPrice': val_actual_price,
        'PredictedPrice': val_pred_price,
        'Dataset': 'Validation'
    })
    
    test_df = pd.DataFrame({
        'Date': test_actual_dates,
        'ActualPrice': test_actual_price,
        'PredictedPrice': test_pred_price,
        'Dataset': 'Test'
    })
    
    full_comparison = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full_comparison_file = Config.OUTPUT_DIR / f'{ticker}_full_comparison.csv'
    full_comparison.to_csv(full_comparison_file, index=False)
    print(f"✓ Full comparison saved: {full_comparison_file}")
    
    # 保存测试集预测
    test_predictions = pd.DataFrame({
        'Date': test_actual_dates,
        'PredictedPrice': test_pred_price
    })
    test_pred_file = Config.OUTPUT_DIR / f'{ticker}_predictions.csv'
    test_predictions.to_csv(test_pred_file, index=False)
    print(f"✓ Test predictions saved: {test_pred_file}")
    
    return {
        'ticker': ticker,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'test_metrics': test_metrics,
        'test_dates': test_actual_dates,
        'test_actual_price': test_actual_price,
        'test_predicted_price': test_pred_price
    }

def main():
    """主函数 - 批量处理所有股票"""
    print("=" * 80)
    print(" " * 20 + "CNN-BiLSTM Stock Prediction System")
    print("=" * 80)
    
    # 加载数据
    print("\n[1/4] Loading data...")
    all_factors = pd.read_csv(Config.DATA_DIR / "all_factors_complete.csv", 
                             index_col=0, parse_dates=True)
    close_prices = pd.read_csv(Config.DATA_DIR / "close_prices.csv",
                               index_col=0, parse_dates=True)
    
    print(f"   Data range: {all_factors.index[0].date()} to {all_factors.index[-1].date()}")
    print(f"   Total trading days: {len(all_factors)}")
    print(f"   Found {len(Config.TICKERS)} stocks to process")
    
    # 批量处理
    print("\n[2/4] Training models for all stocks...")
    print("-" * 80)
    
    results = {}
    all_performance = []
    
    for i, ticker in enumerate(Config.TICKERS, 1):
        print(f"\n[{i}/{len(Config.TICKERS)}] Processing {ticker}...")
        
        try:
            stock_data = prepare_stock_data(ticker, all_factors, close_prices)
            
            if stock_data is None:
                print(f"Skipping {ticker}")
                continue
            
            result = train_single_stock(ticker, stock_data)
            results[ticker] = result
            
            for metrics in [result['train_metrics'], result['val_metrics'], result['test_metrics']]:
                all_performance.append({
                    'Ticker': ticker,
                    **metrics
                })
            
        except Exception as e:
            print(f"\n❌ {ticker} failed: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # 生成汇总报告
    print("\n[3/4] Generating summary reports...")
    print("-" * 80)
    
    if len(results) == 0:
        print("❌ No successful predictions!")
        return
    
    summary_data = []
    for ticker, result in results.items():
        test_metrics = result['test_metrics']
        summary_data.append({
            'Ticker': ticker,
            'Status': 'Success',
            'TestSamples': test_metrics['Samples'],
            'TestDateRange': f"{result['test_dates'][0].date()} to {result['test_dates'][-1].date()}",
            'TestR2': f"{test_metrics['R2']:.4f}",
            'TestRMSE': f"{test_metrics['RMSE']:.6f}",
            'TestDirectionAcc': f"{test_metrics['DirectionAccuracy']:.2f}%"
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_file = Config.OUTPUT_DIR / 'prediction_summary.csv'
    summary_df.to_csv(summary_file, index=False)
    
    performance_summary_df = pd.DataFrame(all_performance)
    performance_summary_file = Config.OUTPUT_DIR / 'all_performance_summary.csv'
    performance_summary_df.to_csv(performance_summary_file, index=False)
    
    print(f"\n✓ Summary reports saved:")
    print(f"   - {summary_file}")
    print(f"   - {performance_summary_file}")
    
    print("\n" + "=" * 80)
    print("Summary Statistics:")
    print("=" * 80)
    print(f"   Successfully processed: {len(results)}/{len(Config.TICKERS)} stocks")
    
    print(f"\n   ✓ Successful stocks:")
    for _, row in summary_df.iterrows():
        print(f"      {row['Ticker']}: R²={row['TestR2']}, RMSE={row['TestRMSE']}, Direction Acc={row['TestDirectionAcc']}")
    
    # 生成汇总预测文件
    print("\n[4/4] Generating aggregated prediction files...")
    
    all_predictions = []
    for ticker in results.keys():
        csv_path = Config.OUTPUT_DIR / f'{ticker}_predictions.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df['Ticker'] = ticker
            all_predictions.append(df)
    
    if all_predictions:
        all_predictions_df = pd.concat(all_predictions, ignore_index=True)
        all_predictions_path = Config.OUTPUT_DIR / 'ALL_STOCKS_predictions.csv'
        all_predictions_df.to_csv(all_predictions_path, index=False)
        print(f"✓ All predictions saved: {all_predictions_path}")
    
    print("\n" + "=" * 80)
    print(" " * 30 + "All Done!")
    print("=" * 80)
    print(f"\nModel: Multi-Scale CNN + Bidirectional LSTM with Attention")
    print(f"Successfully processed {len(results)} stocks")
    print(f"\nGenerated files per stock:")
    print(f"   1. {{TICKER}}_predictions.csv - Test set predictions")
    print(f"   2. {{TICKER}}_full_comparison.csv - Full comparison (Train/Val/Test)")
    print(f"   3. {{TICKER}}_performance.csv - Performance metrics")
    print(f"\nSummary files:")
    print(f"   - prediction_summary.csv")
    print(f"   - all_performance_summary.csv")

if __name__ == "__main__":
    main()