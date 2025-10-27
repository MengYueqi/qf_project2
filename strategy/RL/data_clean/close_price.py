import os
import pandas as pd

# 输入文件路径
input_path = "data/close_prices.csv"

# 输出目录
output_dir = "strategy/RL/data/close_price"

# 如果不存在就创建
os.makedirs(output_dir, exist_ok=True)

# 读取原始 CSV
df = pd.read_csv(input_path)

# 拆分并保存
for col in df.columns:
    if col == "Date":
        continue
    sub_df = df[["Date", col]]
    out_path = os.path.join(output_dir, f"{col}.csv")
    sub_df.to_csv(out_path, index=False)
    print(f"✅ Saved: {out_path}")

print("🎯 All files saved successfully.")
