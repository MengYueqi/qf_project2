import os
import pandas as pd
import re

# ========= 配置区域 =========
input_path = "data/all_factors_complete.csv"   # 你的这个带 AAPL_ret_1d 之类表头的csv
output_dir = "strategy/RL/data/factors"
os.makedirs(output_dir, exist_ok=True)

# ========= 辅助函数 =========
def split_features_by_ticker(df: pd.DataFrame, output_dir: str):
    """
    作用：
    - 发现所有形如 TICKER_xxx 的列（例如 AAPL_ret_1d, AMZN_ret_5d, TSLA_mom_accel）
    - 按 ticker 分组导出成多个 CSV
    - 每个 CSV: 第一列 Date，后面是去掉前缀后的列名 (ret_1d, ret_5d, mom_accel, ...)
    """
    # 拿到所有列名，除了 Date
    cols = [c for c in df.columns if c != "Date"]

    # 用正则解析列名，把股票代码和后缀拆开
    # 假设列名格式是  <Ticker>_<RestOfName>
    # 例如:  AAPL_ret_1d  -> ticker="AAPL", rest="ret_1d"
    pattern = re.compile(r"^([A-Z]+)_(.+)$")

    ticker_to_cols = {}  # ticker -> list of (original_col, new_col_name)

    for col in cols:
        m = pattern.match(col)
        if not m:
            # 如果这一列不符合“前缀_后缀”的格式，直接跳过或以后再决定怎么处理
            continue
        ticker, rest = m.group(1), m.group(2)
        ticker_to_cols.setdefault(ticker, []).append((col, rest))

    # 针对每个ticker生成并写csv
    for ticker, col_pairs in ticker_to_cols.items():
        # 取出原始列名
        original_cols = [c[0] for c in col_pairs]

        # 构造一个子DataFrame：Date + 这些列
        sub_df = df[["Date"] + original_cols].copy()

        # 重命名列，把 ticker_ 去掉
        rename_map = {orig: new for (orig, new) in col_pairs}
        sub_df = sub_df.rename(columns=rename_map)

        # 输出文件名，比如 AAPL_features.csv
        out_path = os.path.join(output_dir, f"{ticker}.csv")
        sub_df.to_csv(out_path, index=False)
        print(f"✅ Saved: {out_path}")


# ========= 主流程 =========
if __name__ == "__main__":
    df = pd.read_csv(input_path)
    split_features_by_ticker(df, output_dir)

    print("🎯 Done.")
