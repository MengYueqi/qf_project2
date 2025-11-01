"""
批量股票预测系统 - 完整版
为每只股票生成：
1. 测试集预测结果 (Date, PredictedPrice)
2. 完整预测结果 (Date, ActualPrice, PredictedPrice, Dataset)
3. 模型性能指标 (RMSE, MAE, R², Direction Accuracy)
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# 设置随机种子
np.random.seed(42)
tf.random.set_seed(42)

class StockPredictor:
    """股票预测器 - 完整版"""
    
    def __init__(self, ticker, lookback_window=30, 
                 train_end='2020-12-31', val_end='2022-12-31'):
        self.ticker = ticker
        self.lookback_window = lookback_window
        self.train_end = train_end
        self.val_end = val_end
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.model = None
        self.selected_features = None
        
        # 保存日期索引
        self.train_dates = None
        self.val_dates = None
        self.test_dates = None
        
    def select_features(self, all_factors_df):
        """选择特征"""
        ticker = self.ticker
        
        # 定义特征列表
        feature_names = [
            # 动量因子 (4个)
            f'{ticker}_ret_5d',
            f'{ticker}_ret_20d',
            f'{ticker}_ret_60d',
            f'{ticker}_mom_accel',
            
            # 技术指标 (5个)
            f'{ticker}_rsi',
            f'{ticker}_macd_hist',
            f'{ticker}_ma20_bias',
            f'{ticker}_bb_position',
            f'{ticker}_ma_alignment',
            
            # 波动率 (3个)
            f'{ticker}_vol_20d',
            f'{ticker}_vol_60d',
            f'{ticker}_atr',
            
            # 成交量 (2个)
            f'{ticker}_volume_ratio',
            f'{ticker}_volume_mom',
            
            # 价格位置 (2个)
            f'{ticker}_52w_position',
            f'{ticker}_from_52w_high',
            
            # 微观结构 (1个)
            f'{ticker}_hl_spread',
        ]
        
        # 检查哪些特征存在
        available_features = [f for f in feature_names if f in all_factors_df.columns]
        self.selected_features = available_features
        
        return available_features
    
    def prepare_data(self, close_df, factors_df):
        """准备数据"""
        # 选择特征
        features = self.select_features(factors_df)
        
        if len(features) == 0:
            raise ValueError(f"{self.ticker}: 没有可用的特征！")
        
        # 检查目标股票是否存在
        if self.ticker not in close_df.columns:
            raise ValueError(f"{self.ticker} 不在收盘价数据中")
        
        # 提取数据
        y = close_df[[self.ticker]].copy()
        X = factors_df[features].copy()
        
        # 处理缺失值
        X = X.fillna(method='ffill').fillna(method='bfill')
        y = y.fillna(method='ffill').fillna(method='bfill')
        
        # 删除仍有NaN的行
        valid_idx = X.notna().all(axis=1) & y.notna().all(axis=1)
        X = X[valid_idx]
        y = y[valid_idx]
        
        # 数据集分割（严格的时间顺序）
        # 训练集: <= train_end（例如：<= 2020-12-31）
        train_X = X[X.index <= self.train_end]
        train_y = y[y.index <= self.train_end]
        
        # 验证集: > train_end AND <= val_end（例如：2021-01-01 到 2022-12-31）
        val_X = X[(X.index > self.train_end) & (X.index <= self.val_end)]
        val_y = y[(y.index > self.train_end) & (y.index <= self.val_end)]
        
        # 测试集: > val_end（例如：>= 2023-01-03，第一个交易日）
        test_X = X[X.index > self.val_end]
        test_y = y[y.index > self.val_end]
        
        return train_X, train_y, val_X, val_y, test_X, test_y
    
    def create_sequences(self, X, y, lookback):
        """创建序列"""
        X_seq, y_seq = [], []
        for i in range(lookback, len(X)):
            X_seq.append(X[i-lookback:i])
            y_seq.append(y[i])
        return np.array(X_seq), np.array(y_seq)
    
    def prepare_sequences(self, train_X, train_y, val_X, val_y, test_X, test_y):
        """
        标准化并创建序列
        关键：允许测试集从第一天开始预测，使用验证集的历史数据
        """
        # 标准化
        X_train_scaled = self.scaler_X.fit_transform(train_X.values)
        X_val_scaled = self.scaler_X.transform(val_X.values)
        X_test_scaled = self.scaler_X.transform(test_X.values)
        
        y_train_scaled = self.scaler_y.fit_transform(train_y.values)
        y_val_scaled = self.scaler_y.transform(val_y.values)
        y_test_scaled = self.scaler_y.transform(test_y.values)
        
        # 创建训练集序列（正常）
        X_train_seq, y_train_seq = self.create_sequences(
            X_train_scaled, y_train_scaled, self.lookback_window)
        
        # 创建验证集序列（使用训练集+验证集的数据）
        # 这样验证集第一天就可以预测
        X_trainval_scaled = np.vstack([X_train_scaled, X_val_scaled])
        y_trainval_scaled = np.vstack([y_train_scaled, y_val_scaled])
        trainval_indices = train_X.index.append(val_X.index)
        
        # 从训练集最后lookback_window天开始，到验证集结束
        start_idx = len(X_train_scaled) - self.lookback_window
        X_val_seq_list, y_val_seq_list = [], []
        
        for i in range(len(X_val_scaled)):
            X_val_seq_list.append(X_trainval_scaled[start_idx + i:start_idx + i + self.lookback_window])
            y_val_seq_list.append(y_trainval_scaled[start_idx + i + self.lookback_window])
        
        X_val_seq = np.array(X_val_seq_list)
        y_val_seq = np.array(y_val_seq_list)
        
        # 创建测试集序列（使用验证集+测试集的数据）
        # 这样测试集第一天就可以预测
        X_valtest_scaled = np.vstack([X_val_scaled, X_test_scaled])
        y_valtest_scaled = np.vstack([y_val_scaled, y_test_scaled])
        
        # 从验证集最后lookback_window天开始，到测试集结束
        start_idx = len(X_val_scaled) - self.lookback_window
        X_test_seq_list, y_test_seq_list = [], []
        
        for i in range(len(X_test_scaled)):
            X_test_seq_list.append(X_valtest_scaled[start_idx + i:start_idx + i + self.lookback_window])
            y_test_seq_list.append(y_valtest_scaled[start_idx + i + self.lookback_window])
        
        X_test_seq = np.array(X_test_seq_list)
        y_test_seq = np.array(y_test_seq_list)
        
        # 保存日期索引
        # 训练集：跳过前lookback_window天
        self.train_dates = train_X.index[self.lookback_window:]
        # 验证集：从第一天开始（使用了训练集的历史数据）
        self.val_dates = val_X.index
        # 测试集：从第一天开始（使用了验证集的历史数据）
        self.test_dates = test_X.index
        
        return (X_train_seq, y_train_seq, X_val_seq, y_val_seq, 
                X_test_seq, y_test_seq)
    
    def build_model(self, input_shape):
        """构建LSTM模型"""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=input_shape,
                 kernel_regularizer=keras.regularizers.l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            
            LSTM(64, return_sequences=True,
                 kernel_regularizer=keras.regularizers.l2(0.001)),
            BatchNormalization(),
            Dropout(0.3),
            
            LSTM(32, return_sequences=False,
                 kernel_regularizer=keras.regularizers.l2(0.001)),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(16, activation='relu',
                  kernel_regularizer=keras.regularizers.l2(0.001)),
            Dropout(0.2),
            
            Dense(1)
        ])
        
        optimizer = Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        self.model = model
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
        """训练模型"""
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=0
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-7,
            verbose=0
        )
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop, reduce_lr],
            verbose=0
        )
        
        return history
    
    def predict(self, X):
        """预测并反标准化"""
        y_pred_scaled = self.model.predict(X, verbose=0)
        y_pred = self.scaler_y.inverse_transform(y_pred_scaled)
        return y_pred
    
    def evaluate(self, X, y_true_scaled, dataset_name="Dataset"):
        """
        评估模型性能
        
        返回:
        - metrics: 性能指标字典
        - predictions: 预测值 (反标准化后)
        - actuals: 真实值 (反标准化后)
        """
        # 预测
        y_pred = self.predict(X)
        
        # 反标准化真实值
        y_true = self.scaler_y.inverse_transform(y_true_scaled)
        
        # 计算指标
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # 方向准确率
        if len(y_true) > 1:
            direction_true = np.diff(y_true.flatten()) > 0
            direction_pred = np.diff(y_pred.flatten()) > 0
            direction_accuracy = np.mean(direction_true == direction_pred) * 100
        else:
            direction_accuracy = 0
        
        # MAPE
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        metrics = {
            'Dataset': dataset_name,
            'Samples': len(y_true),
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'DirectionAccuracy': direction_accuracy
        }
        
        return metrics, y_pred.flatten(), y_true.flatten()


def batch_predict_all_stocks(close_prices_path, all_factors_path, output_dir):
    """
    批量预测所有股票 - 完整版
    
    生成文件:
    1. {TICKER}_predictions.csv - 测试集预测 (Date, PredictedPrice)
    2. {TICKER}_full_comparison.csv - 完整对比 (Date, ActualPrice, PredictedPrice, Dataset)
    3. {TICKER}_performance.csv - 性能指标
    4. prediction_summary.csv - 总汇总
    5. all_performance_summary.csv - 所有股票性能汇总
    """
    print("=" * 80)
    print(" " * 20 + "批量股票预测系统 - 完整版")
    print("=" * 80)
    
    # 加载数据
    print("\n[1/5] 加载数据...")
    close_df = pd.read_csv(close_prices_path, index_col=0, parse_dates=True)
    factors_df = pd.read_csv(all_factors_path, index_col=0, parse_dates=True)
    
    # 确保索引对齐
    common_dates = close_df.index.intersection(factors_df.index)
    close_df = close_df.loc[common_dates]
    factors_df = factors_df.loc[common_dates]
    
    print(f"   数据日期范围: {close_df.index[0]} 到 {close_df.index[-1]}")
    print(f"   交易日数: {len(close_df)}")
    
    # 获取所有股票代码
    tickers = close_df.columns.tolist()
    print(f"\n   发现 {len(tickers)} 只股票: {tickers}")
    
    # 为每只股票进行预测
    print("\n[2/5] 开始批量预测...")
    print("-" * 80)
    
    results_summary = []
    all_performance = []
    
    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] 处理 {ticker}...")
        
        try:
            # 初始化预测器
            predictor = StockPredictor(
                ticker=ticker,
                lookback_window=30,
                train_end='2020-12-31',
                val_end='2022-12-31'
            )
            
            # 准备数据
            print(f"   - 准备数据...")
            train_X, train_y, val_X, val_y, test_X, test_y = predictor.prepare_data(
                close_df, factors_df)
            
            # 创建序列
            print(f"   - 创建序列...")
            X_train, y_train, X_val, y_val, X_test, y_test = predictor.prepare_sequences(
                train_X, train_y, val_X, val_y, test_X, test_y)
            
            # 构建模型
            print(f"   - 构建模型...")
            input_shape = (X_train.shape[1], X_train.shape[2])
            predictor.build_model(input_shape)
            
            # 训练模型
            print(f"   - 训练模型...")
            history = predictor.train(X_train, y_train, X_val, y_val, 
                                     epochs=100, batch_size=32)
            
            # ====================
            # 评估所有数据集
            # ====================
            print(f"   - 评估模型性能...")
            
            # 训练集评估
            train_metrics, train_pred, train_actual = predictor.evaluate(
                X_train, y_train, "Train")
            
            # 验证集评估
            val_metrics, val_pred, val_actual = predictor.evaluate(
                X_val, y_val, "Validation")
            
            # 测试集评估
            test_metrics, test_pred, test_actual = predictor.evaluate(
                X_test, y_test, "Test")
            
            # ====================
            # 保存性能指标
            # ====================
            performance_df = pd.DataFrame([train_metrics, val_metrics, test_metrics])
            performance_file = f"{output_dir}/{ticker}_performance.csv"
            performance_df.to_csv(performance_file, index=False)
            
            print(f"   ✓ 性能指标已保存: {performance_file}")
            print(f"      训练集 R²={train_metrics['R2']:.4f}, RMSE=${train_metrics['RMSE']:.2f}")
            print(f"      验证集 R²={val_metrics['R2']:.4f}, RMSE=${val_metrics['RMSE']:.2f}")
            print(f"      测试集 R²={test_metrics['R2']:.4f}, RMSE=${test_metrics['RMSE']:.2f}")
            
            # ====================
            # 保存完整对比结果
            # ====================
            print(f"   - 保存预测结果...")
            
            # 合并所有数据集的预测结果
            full_comparison = pd.DataFrame()
            
            # 训练集
            train_df = pd.DataFrame({
                'Date': predictor.train_dates,
                'ActualPrice': train_actual,
                'PredictedPrice': train_pred,
                'Dataset': 'Train'
            })
            
            # 验证集
            val_df = pd.DataFrame({
                'Date': predictor.val_dates,
                'ActualPrice': val_actual,
                'PredictedPrice': val_pred,
                'Dataset': 'Validation'
            })
            
            # 测试集
            test_df = pd.DataFrame({
                'Date': predictor.test_dates,
                'ActualPrice': test_actual,
                'PredictedPrice': test_pred,
                'Dataset': 'Test'
            })
            
            # 合并
            full_comparison = pd.concat([train_df, val_df, test_df], ignore_index=True)
            full_comparison_file = f"{output_dir}/{ticker}_full_comparison.csv"
            full_comparison.to_csv(full_comparison_file, index=False)
            
            print(f"   ✓ 完整对比已保存: {full_comparison_file}")
            
            # ====================
            # 保存测试集预测（原始格式）
            # ====================
            test_predictions = pd.DataFrame({
                'Date': predictor.test_dates,
                'PredictedPrice': test_pred
            })
            test_pred_file = f"{output_dir}/{ticker}_predictions.csv"
            test_predictions.to_csv(test_pred_file, index=False)
            
            print(f"   ✓ 测试集预测已保存: {test_pred_file}")
            
            # ====================
            # 记录汇总
            # ====================
            results_summary.append({
                'Ticker': ticker,
                'Status': 'Success',
                'TestSamples': len(test_pred),
                'TestDateRange': f"{predictor.test_dates[0].date()} to {predictor.test_dates[-1].date()}",
                'TestR2': f"{test_metrics['R2']:.4f}",
                'TestRMSE': f"${test_metrics['RMSE']:.2f}",
                'TestDirectionAcc': f"{test_metrics['DirectionAccuracy']:.2f}%"
            })
            
            # 添加到性能汇总
            for metrics in [train_metrics, val_metrics, test_metrics]:
                all_performance.append({
                    'Ticker': ticker,
                    **metrics
                })
            
        except Exception as e:
            print(f"   ✗ 错误: {str(e)}")
            results_summary.append({
                'Ticker': ticker,
                'Status': f'Failed: {str(e)}',
                'TestSamples': 0,
                'TestDateRange': 'N/A',
                'TestR2': 'N/A',
                'TestRMSE': 'N/A',
                'TestDirectionAcc': 'N/A'
            })
            continue
    
    # ====================
    # 生成汇总报告
    # ====================
    print("\n[3/5] 生成汇总报告...")
    print("-" * 80)
    
    # 预测结果汇总
    summary_df = pd.DataFrame(results_summary)
    summary_file = f"{output_dir}/prediction_summary.csv"
    summary_df.to_csv(summary_file, index=False)
    
    # 性能指标汇总
    performance_summary_df = pd.DataFrame(all_performance)
    performance_summary_file = f"{output_dir}/all_performance_summary.csv"
    performance_summary_df.to_csv(performance_summary_file, index=False)
    
    print("\n汇总统计:")
    success_count = len(summary_df[summary_df['Status'] == 'Success'])
    fail_count = len(summary_df) - success_count
    print(f"   成功: {success_count}/{len(tickers)}")
    print(f"   失败: {fail_count}/{len(tickers)}")
    
    if success_count > 0:
        print(f"\n   ✓ 成功股票:")
        for _, row in summary_df[summary_df['Status'] == 'Success'].iterrows():
            print(f"      {row['Ticker']}: R²={row['TestR2']}, RMSE={row['TestRMSE']}, 方向准确率={row['TestDirectionAcc']}")
    
    if fail_count > 0:
        print(f"\n   ✗ 失败股票:")
        for _, row in summary_df[summary_df['Status'] != 'Success'].iterrows():
            print(f"      {row['Ticker']}: {row['Status']}")
    
    print(f"\n   汇总报告已保存:")
    print(f"      - {summary_file}")
    print(f"      - {performance_summary_file}")
    
    # ====================
    # 完成
    # ====================
    print("\n[4/5] 生成的文件列表...")
    print("-" * 80)
    print(f"\n每只股票生成3个文件:")
    print(f"   1. {{TICKER}}_predictions.csv - 测试集预测 (Date, PredictedPrice)")
    print(f"   2. {{TICKER}}_full_comparison.csv - 完整对比 (Date, Actual, Predicted, Dataset)")
    print(f"   3. {{TICKER}}_performance.csv - 性能指标 (Train/Val/Test)")
    
    print(f"\n总汇总文件:")
    print(f"   - prediction_summary.csv - 所有股票预测汇总")
    print(f"   - all_performance_summary.csv - 所有股票性能汇总")
    
    print("\n[5/5] 完成！")
    print("=" * 80)
    print(f"\n所有结果已保存到: {output_dir}")
    print(f"\n成功处理 {success_count} 只股票，生成 {success_count * 3 + 2} 个文件")
    
    return summary_df, performance_summary_df


if __name__ == "__main__":
    # 配置路径
    CLOSE_PRICES_PATH = '/Users/daizhuolin/Desktop/NUS/5205/5205_project2/download/close_prices.csv'
    ALL_FACTORS_PATH = '/Users/daizhuolin/Desktop/NUS/5205/5205_project2/download/all_factors_complete.csv'
    OUTPUT_DIR = '/Users/daizhuolin/Desktop/NUS/5205/5205_project2/outputs'
    
    # 运行批量预测
    summary, performance = batch_predict_all_stocks(
        CLOSE_PRICES_PATH, 
        ALL_FACTORS_PATH, 
        OUTPUT_DIR
    )
    
    print("\n" + "=" * 80)
    print(" " * 30 + "全部完成!")
    print("=" * 80)
    print("\n提示: 查看 all_performance_summary.csv 了解所有股票的详细性能指标")