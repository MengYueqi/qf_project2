import glob
import os
import pandas as pd
def load_model_predictions_with_tag(folder_path: str, tag: str) -> pd.DataFrame:
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    prediction_files = [
        f for f in csv_files
        if all(kw not in os.path.basename(f).lower()
               for kw in ["performance", "comparison", "summary", "test_curve"])
    ]

    if not prediction_files:
        print(f"[WARN] No prediction-like csv files in {folder_path}")
        return None

    rows = []
    for file in prediction_files:
        base = os.path.splitext(os.path.basename(file))[0]
        if "all_stocks" in base.lower():
            continue

        stock_plain = base.replace("_predictions", "")
        df = pd.read_csv(file)

        if not {"Date", "PredictedPrice"}.issubset(df.columns):
            print(f"[SKIP] {file} (no Date/PredictedPrice columns)")
            continue

        df = df[["Date", "PredictedPrice"]].copy()
        df["stock_tag"] = f"{stock_plain}_{tag}"
        rows.append(df)

    if not rows:
        print(f"[WARN] No valid prediction files in {folder_path}")
        return None

    all_predictions = pd.concat(rows, ignore_index=True)

    wide = (
        all_predictions
        .pivot_table(
            index="Date",
            columns="stock_tag",
            values="PredictedPrice",
            aggfunc="last"
        )
        .reset_index()
    )

    print(f"[OK] Loaded {len(rows)} stock files from {folder_path} ({tag})")
    return wide