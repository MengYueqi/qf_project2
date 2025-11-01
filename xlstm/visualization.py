import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
sns.set_palette("husl")

class XLSTMVisualizer:
    def __init__(self, data_dir='xlstm_pure_pred', output_dir='xlstm_visualizations'):
        """
        Initialize the visualizer
        
        Parameters:
            data_dir: Directory containing CSV files
            output_dir: Directory for visualization outputs
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load data
        self.load_data()
        
    def load_data(self):
        """Load all CSV files"""
        print("📊 Loading data...")
        
        # Load aggregated predictions
        all_pred_path = self.data_dir / 'ALL_STOCKS_predictions.csv'
        if all_pred_path.exists():
            self.all_predictions = pd.read_csv(all_pred_path)
            self.all_predictions['Date'] = pd.to_datetime(self.all_predictions['Date'])
            print(f"  ✓ Loaded aggregated predictions: {len(self.all_predictions)} records")
        else:
            self.all_predictions = None
            print("  ✗ Aggregated predictions not found")
        
        # Load quality report
        report_path = self.data_dir / 'prediction_quality_report.csv'
        if report_path.exists():
            self.quality_report = pd.read_csv(report_path)
            print(f"  ✓ Loaded quality report: {len(self.quality_report)} stocks")
        else:
            self.quality_report = None
            print("  ✗ Quality report not found")
        
        # Load actual and predicted matrices
        actual_path = self.data_dir / 'actual_close_matrix.csv'
        pred_path = self.data_dir / 'predicted_close_matrix.csv'
        
        if actual_path.exists():
            self.actual_matrix = pd.read_csv(actual_path, index_col=0, parse_dates=True)
            print(f"  ✓ Loaded actual price matrix")
        else:
            self.actual_matrix = None
            
        if pred_path.exists():
            self.predicted_matrix = pd.read_csv(pred_path, index_col=0, parse_dates=True)
            print(f"  ✓ Loaded predicted price matrix")
        else:
            self.predicted_matrix = None
    
    def plot_individual_stocks(self, tickers=None, max_plots=10):
        """
        Plot actual vs predicted prices for individual stocks
        
        Parameters:
            tickers: List of tickers to plot, None for all
            max_plots: Maximum number of stocks to plot
        """
        if self.all_predictions is None:
            print("⚠️  No prediction data available")
            return
        
        print("\n📈 Generating individual stock prediction charts...")
        
        if tickers is None:
            tickers = self.all_predictions['Ticker'].unique()
        
        tickers = tickers[:max_plots]
        
        n_stocks = len(tickers)
        n_cols = 2
        n_rows = (n_stocks + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 5*n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for idx, ticker in enumerate(tickers):
            ax = axes[idx]
            
            # Filter data for this stock
            stock_data = self.all_predictions[self.all_predictions['Ticker'] == ticker].copy()
            stock_data = stock_data.sort_values('Date')
            
            # Plot actual and predicted prices
            ax.plot(stock_data['Date'], stock_data['Actual_Close'], 
                   label='Actual Close', linewidth=2, alpha=0.8)
            ax.plot(stock_data['Date'], stock_data['Predicted_Close'], 
                   label='Predicted Close', linewidth=2, alpha=0.8, linestyle='--')
            
            ax.set_title(f'{ticker} - Price Prediction', fontsize=14, fontweight='bold')
            ax.set_xlabel('Date', fontsize=11)
            ax.set_ylabel('Close Price ($)', fontsize=11)
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        # Hide extra subplots
        for idx in range(n_stocks, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        output_path = self.output_dir / 'individual_stock_predictions.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved to: {output_path}")
        plt.close()
    
    def plot_prediction_errors(self):
        """Plot prediction error analysis charts"""
        if self.all_predictions is None:
            print("⚠️  No prediction data available")
            return
        
        print("\n📊 Generating prediction error analysis...")
        
        # Calculate errors
        df = self.all_predictions.copy()
        df['Error'] = df['Predicted_Close'] - df['Actual_Close']
        df['Error_Percentage'] = (df['Error'] / df['Actual_Close']) * 100
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Error distribution histogram
        ax = axes[0, 0]
        for ticker in df['Ticker'].unique():
            ticker_errors = df[df['Ticker'] == ticker]['Error_Percentage']
            ax.hist(ticker_errors, bins=30, alpha=0.5, label=ticker)
        ax.set_xlabel('Prediction Error (%)', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Distribution of Prediction Errors', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # 2. Average absolute error by stock
        ax = axes[0, 1]
        avg_errors = df.groupby('Ticker')['Error_Percentage'].apply(lambda x: np.abs(x).mean())
        avg_errors = avg_errors.sort_values()
        colors = sns.color_palette("RdYlGn_r", len(avg_errors))
        ax.barh(avg_errors.index, avg_errors.values, color=colors)
        ax.set_xlabel('Mean Absolute Error (%)', fontsize=11)
        ax.set_title('Average Prediction Error by Stock', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # 3. Error time series
        ax = axes[1, 0]
        for ticker in df['Ticker'].unique():
            ticker_data = df[df['Ticker'] == ticker].sort_values('Date')
            ax.plot(ticker_data['Date'], ticker_data['Error_Percentage'], 
                   label=ticker, alpha=0.7, linewidth=1.5)
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Prediction Error (%)', fontsize=11)
        ax.set_title('Prediction Error Over Time', fontsize=14, fontweight='bold')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        # 4. Actual vs predicted scatter plot
        ax = axes[1, 1]
        for ticker in df['Ticker'].unique():
            ticker_data = df[df['Ticker'] == ticker]
            ax.scatter(ticker_data['Actual_Close'], ticker_data['Predicted_Close'], 
                      label=ticker, alpha=0.6, s=30)
        
        # Add perfect prediction line
        min_val = df[['Actual_Close', 'Predicted_Close']].min().min()
        max_val = df[['Actual_Close', 'Predicted_Close']].max().max()
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction', linewidth=2)
        
        ax.set_xlabel('Actual Close Price ($)', fontsize=11)
        ax.set_ylabel('Predicted Close Price ($)', fontsize=11)
        ax.set_title('Actual vs Predicted Prices', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / 'prediction_errors_analysis.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved to: {output_path}")
        plt.close()
    
    def plot_quality_metrics(self):
        """Plot prediction quality metrics comparison"""
        if self.quality_report is None:
            print("⚠️  No quality report available")
            return
        
        print("\n📊 Generating quality metrics comparison...")
        
        df = self.quality_report.copy()
        
        # Clean data (remove percentage signs and convert to numeric)
        df['Direction_Accuracy'] = df['Direction_Accuracy'].str.rstrip('%').astype(float)
        df['R_Squared'] = df['R_Squared'].astype(float)
        df['MAE'] = df['MAE'].astype(float)
        df['MSE'] = df['MSE'].astype(float)
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Direction accuracy
        ax = axes[0, 0]
        df_sorted = df.sort_values('Direction_Accuracy', ascending=False)
        colors = sns.color_palette("RdYlGn", len(df_sorted))
        ax.barh(df_sorted['Ticker'], df_sorted['Direction_Accuracy'], color=colors)
        ax.axvline(x=50, color='red', linestyle='--', label='Random Guess (50%)')
        ax.set_xlabel('Direction Accuracy (%)', fontsize=11)
        ax.set_title('Price Direction Prediction Accuracy', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='x')
        
        # 2. R² score
        ax = axes[0, 1]
        df_sorted = df.sort_values('R_Squared', ascending=False)
        colors = sns.color_palette("viridis", len(df_sorted))
        ax.barh(df_sorted['Ticker'], df_sorted['R_Squared'], color=colors)
        ax.set_xlabel('R² Score', fontsize=11)
        ax.set_title('Model Fit Quality (R²)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # 3. MAE comparison
        ax = axes[1, 0]
        df_sorted = df.sort_values('MAE')
        colors = sns.color_palette("coolwarm_r", len(df_sorted))
        ax.barh(df_sorted['Ticker'], df_sorted['MAE'], color=colors)
        ax.set_xlabel('Mean Absolute Error', fontsize=11)
        ax.set_title('Prediction Error (MAE)', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # 4. Radar chart for top 5 stocks
        ax = axes[1, 1]
        top_stocks = df.nlargest(5, 'Direction_Accuracy')
        
        # Normalize metrics to 0-1 range
        metrics = ['Direction_Accuracy', 'R_Squared']
        normalized = top_stocks[metrics].copy()
        normalized['Direction_Accuracy'] = normalized['Direction_Accuracy'] / 100
        
        # Create inverted normalization for MAE (lower is better)
        mae_max = df['MAE'].max()
        mae_min = df['MAE'].min()
        normalized['MAE_normalized'] = 1 - (top_stocks['MAE'] - mae_min) / (mae_max - mae_min)
        
        # Create radar chart
        categories = ['Direction\nAccuracy', 'R² Score', 'Error\n(Inverted)']
        num_vars = len(categories)
        
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]
        
        ax = plt.subplot(224, projection='polar')
        
        for idx, row in top_stocks.iterrows():
            values = [
                normalized.loc[idx, 'Direction_Accuracy'],
                normalized.loc[idx, 'R_Squared'],
                normalized.loc[idx, 'MAE_normalized']
            ]
            values += values[:1]
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['Ticker'])
            ax.fill(angles, values, alpha=0.15)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_title('Top 5 Stocks - Performance Radar', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
        ax.grid(True)
        
        plt.tight_layout()
        output_path = self.output_dir / 'quality_metrics_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved to: {output_path}")
        plt.close()
    
    def plot_portfolio_performance(self):
        """Plot portfolio performance analysis"""
        if self.all_predictions is None:
            print("⚠️  No prediction data available")
            return
        
        print("\n💼 Generating portfolio performance analysis...")
        
        df = self.all_predictions.copy()
        df = df.sort_values('Date')
        
        # Calculate average performance across all stocks by date
        daily_actual = df.groupby('Date')['Actual_Close'].mean()
        daily_predicted = df.groupby('Date')['Predicted_Close'].mean()
        
        # Calculate cumulative returns
        actual_returns = daily_actual.pct_change().fillna(0)
        predicted_returns = daily_predicted.pct_change().fillna(0)
        
        cumulative_actual = (1 + actual_returns).cumprod()
        cumulative_predicted = (1 + predicted_returns).cumprod()
        
        fig, axes = plt.subplots(2, 1, figsize=(16, 10))
        
        # 1. Average price trend
        ax = axes[0]
        ax.plot(daily_actual.index, daily_actual.values, 
               label='Actual Average Price', linewidth=2.5, alpha=0.8)
        ax.plot(daily_predicted.index, daily_predicted.values, 
               label='Predicted Average Price', linewidth=2.5, alpha=0.8, linestyle='--')
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Average Close Price ($)', fontsize=11)
        ax.set_title('Portfolio Average Price Trend', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        # 2. Cumulative returns comparison
        ax = axes[1]
        ax.plot(cumulative_actual.index, (cumulative_actual - 1) * 100, 
               label='Actual Cumulative Return', linewidth=2.5, alpha=0.8)
        ax.plot(cumulative_predicted.index, (cumulative_predicted - 1) * 100, 
               label='Predicted Cumulative Return', linewidth=2.5, alpha=0.8, linestyle='--')
        ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Cumulative Return (%)', fontsize=11)
        ax.set_title('Portfolio Cumulative Return Comparison', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        output_path = self.output_dir / 'portfolio_performance.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved to: {output_path}")
        plt.close()
    
    def plot_correlation_heatmap(self):
        """Plot correlation heatmap of prediction errors between stocks"""
        if self.all_predictions is None:
            print("⚠️  No prediction data available")
            return
        
        print("\n🔥 Generating correlation heatmap...")
        
        df = self.all_predictions.copy()
        df['Error_Percentage'] = ((df['Predicted_Close'] - df['Actual_Close']) / df['Actual_Close']) * 100
        
        # Create error matrix
        error_matrix = df.pivot_table(
            index='Date',
            columns='Ticker',
            values='Error_Percentage'
        )
        
        # Calculate correlation
        correlation = error_matrix.corr()
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(correlation, annot=True, fmt='.2f', cmap='coolwarm', 
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   ax=ax)
        
        ax.set_title('Correlation of Prediction Errors Between Stocks', 
                    fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        output_path = self.output_dir / 'error_correlation_heatmap.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved to: {output_path}")
        plt.close()
    
    def generate_summary_report(self):
        """Generate HTML format summary report"""
        print("\n📄 Generating HTML summary report...")
        
        if self.quality_report is None:
            print("⚠️  No quality report available")
            return
        
        df = self.quality_report.copy()
        
        # Clean data
        df['Direction_Accuracy_num'] = df['Direction_Accuracy'].str.rstrip('%').astype(float)
        df['R_Squared_num'] = df['R_Squared'].astype(float)
        df['MAE_num'] = df['MAE'].astype(float)
        
        # Calculate statistics
        avg_accuracy = df['Direction_Accuracy_num'].mean()
        avg_r2 = df['R_Squared_num'].mean()
        avg_mae = df['MAE_num'].mean()
        
        best_accuracy_stock = df.loc[df['Direction_Accuracy_num'].idxmax(), 'Ticker']
        best_r2_stock = df.loc[df['R_Squared_num'].idxmax(), 'Ticker']
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>xLSTM Stock Prediction Summary Report</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 40px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
        }}
        .summary-box {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .metric {{
            display: inline-block;
            margin: 10px 20px;
            padding: 15px;
            background-color: #3498db;
            color: white;
            border-radius: 5px;
            min-width: 200px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
        }}
        .metric-label {{
            font-size: 12px;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .highlight {{
            background-color: #2ecc71;
            color: white;
            padding: 2px 8px;
            border-radius: 3px;
        }}
        .image-grid {{
            display: grid;
            grid-template-columns: 1fr;
            gap: 20px;
            margin: 20px 0;
        }}
        .image-container {{
            text-align: center;
        }}
        .image-container img {{
            max-width: 100%;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .image-caption {{
            margin-top: 10px;
            font-style: italic;
            color: #7f8c8d;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 xLSTM Stock Prediction Summary Report</h1>
        
        <div class="summary-box">
            <h2>📊 Overall Performance Metrics</h2>
            <div class="metric">
                <div class="metric-value">{avg_accuracy:.2f}%</div>
                <div class="metric-label">Average Direction Accuracy</div>
            </div>
            <div class="metric">
                <div class="metric-value">{avg_r2:.4f}</div>
                <div class="metric-label">Average R² Score</div>
            </div>
            <div class="metric">
                <div class="metric-value">{avg_mae:.6f}</div>
                <div class="metric-label">Average MAE</div>
            </div>
        </div>
        
        <h2>🏆 Best Performing Stocks</h2>
        <div class="summary-box">
            <p><strong>Highest Direction Accuracy:</strong> <span class="highlight">{best_accuracy_stock}</span></p>
            <p><strong>Best R² Score:</strong> <span class="highlight">{best_r2_stock}</span></p>
        </div>
        
        <h2>📈 Detailed Results by Stock</h2>
        <table>
            <thead>
                <tr>
                    <th>Ticker</th>
                    <th>Direction Accuracy</th>
                    <th>R² Score</th>
                    <th>MAE</th>
                    <th>MSE</th>
                </tr>
            </thead>
            <tbody>
"""
        
        # Sort by accuracy and add table rows
        df_sorted = df.sort_values('Direction_Accuracy_num', ascending=False)
        for _, row in df_sorted.iterrows():
            html_content += f"""
                <tr>
                    <td><strong>{row['Ticker']}</strong></td>
                    <td>{row['Direction_Accuracy']}</td>
                    <td>{row['R_Squared']}</td>
                    <td>{row['MAE']}</td>
                    <td>{row['MSE']}</td>
                </tr>
"""
        
        html_content += """
            </tbody>
        </table>
        
        <h2>📸 Visualization Results</h2>
        <div class="image-grid">
            <div class="image-container">
                <img src="individual_stock_predictions.png" alt="Individual Stock Predictions">
                <div class="image-caption">Individual Stock Price Predictions</div>
            </div>
            <div class="image-container">
                <img src="prediction_errors_analysis.png" alt="Prediction Errors Analysis">
                <div class="image-caption">Comprehensive Error Analysis</div>
            </div>
            <div class="image-container">
                <img src="quality_metrics_comparison.png" alt="Quality Metrics">
                <div class="image-caption">Model Quality Metrics Comparison</div>
            </div>
            <div class="image-container">
                <img src="portfolio_performance.png" alt="Portfolio Performance">
                <div class="image-caption">Portfolio Performance Analysis</div>
            </div>
            <div class="image-container">
                <img src="error_correlation_heatmap.png" alt="Error Correlation">
                <div class="image-caption">Error Correlation Between Stocks</div>
            </div>
        </div>
        
        <div class="summary-box" style="margin-top: 40px;">
            <h3>💡 Key Insights</h3>
            <ul>
                <li>The xLSTM model achieved an average direction accuracy of <strong>{avg_accuracy:.2f}%</strong> across all stocks</li>
                <li>Best performing stock: <strong>{best_accuracy_stock}</strong> with the highest direction prediction accuracy</li>
                <li>The model shows {"good" if avg_r2 > 0.5 else "moderate"} overall fit with an average R² of <strong>{avg_r2:.4f}</strong></li>
                <li>Average prediction error (MAE) is <strong>{avg_mae:.6f}</strong></li>
            </ul>
        </div>
    </div>
</body>
</html>
"""
        
        output_path = self.output_dir / 'summary_report.html'
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"  ✓ Saved to: {output_path}")
    
    def run_all(self):
        """Execute all visualization tasks"""
        print("\n" + "="*60)
        print("🎨 xLSTM Prediction Results Visualization System")
        print("="*60)
        
        self.plot_individual_stocks()
        self.plot_prediction_errors()
        self.plot_quality_metrics()
        self.plot_portfolio_performance()
        self.plot_correlation_heatmap()
        self.generate_summary_report()
        
        print("\n" + "="*60)
        print("✅ All visualization tasks completed!")
        print(f"📁 Results saved in: {self.output_dir}")
        print("="*60)


def main():
    # Create visualizer instance
    visualizer = XLSTMVisualizer(
        data_dir='xlstm_pure_pred',
        output_dir='xlstm_visualizations'
    )
    
    # Execute all visualizations
    visualizer.run_all()


if __name__ == "__main__":
    main()