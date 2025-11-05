import pandas as pd
import matplotlib.pyplot as plt

def plot_predicted_prices(file1, file2, label1='Model A', label2='Model B'):
    # 自动读取列名
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # 自动识别日期列（模糊匹配）
    date_col1 = [c for c in df1.columns if 'date' in c.lower()][0]
    date_col2 = [c for c in df2.columns if 'date' in c.lower()][0]

    # 自动识别价格列（排除日期列）
    value_col1 = [c for c in df1.columns if c != date_col1][0]
    value_col2 = [c for c in df2.columns if c != date_col2][0]

    # 转换日期类型并排序
    df1[date_col1] = pd.to_datetime(df1[date_col1])
    df2[date_col2] = pd.to_datetime(df2[date_col2])
    df1 = df1.sort_values(date_col1)
    df2 = df2.sort_values(date_col2)

    # 绘图
    plt.figure(figsize=(10, 5))
    plt.plot(df1[date_col1], df1[value_col1], label=f"{label1} ({value_col1})", linewidth=2)
    plt.plot(df2[date_col2], df2[value_col2], label=f"{label2} ({value_col2})", linewidth=2)

    plt.title('Predicted Price Comparison')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 修改为你的两个文件名
    file1 = "strategy/RL/data_zjh/predict/AMZN.csv"
    file2 = "strategy/RL/data_zjh/close_price/AMZN.csv"
    plot_predicted_prices(file1, file2, label1='Prediction', label2='Ground Truth')
