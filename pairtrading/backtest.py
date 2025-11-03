import numpy as np
import pandas as pd

def backtest_pair(pred_df, actual_df, pair_row, z_entry=1.0, z_exit=0.5):
    """
    pred_df  : DataFrame with columns ['Date', s1, s2]
    actual_df: DataFrame with columns ['Date', s1, s2]
    pair_row : {'stock1','stock2','beta','mu','sigma'}

    Returns a DataFrame indexed by Date with columns
    """
    s1 = pair_row["stock1"]; s2 = pair_row["stock2"]
    beta = float(pair_row["beta"]); mu = float(pair_row["mu"]); sigma = float(pair_row["sigma"])

    pred = pred_df[["Date", s1, s2]].dropna().copy()
    act  = actual_df[["Date", s1, s2]].dropna().copy()
    pred["Date"] = pd.to_datetime(pred["Date"]); act["Date"] = pd.to_datetime(act["Date"])
    pred = pred.set_index("Date").sort_index()
    act  = act.set_index("Date").sort_index()
    idx = pred.index.intersection(act.index)
    pred = pred.loc[idx]; act = act.loc[idx]

    # spread and z
    spread_pred = np.log(pred[s1]) - beta * np.log(pred[s2])
    z = (spread_pred - mu) / sigma

    pos1, pos2 = [], []
    cur1 = 0.0; cur2 = 0.0
    for zi in z:
        if zi > z_entry:          
            cur1, cur2 = -1.0, +beta
        elif zi < -z_entry:     
            cur1, cur2 = +1.0, -beta
        elif abs(zi) < z_exit:   
            cur1, cur2 = 0.0, 0.0
        pos1.append(cur1); pos2.append(cur2)

    out = pd.DataFrame({
        "price_"+s1: act[s1],
        "price_"+s2: act[s2],
        "spread": spread_pred,
        "z": z,
        "pos_"+s1: pos1,
        "pos_"+s2: pos2,
    }, index=idx)

    r1 = np.log(act[s1]).diff()
    r2 = np.log(act[s2]).diff()

    pos1 = out["pos_"+s1].fillna(0.0)
    pos2 = out["pos_"+s2].fillna(0.0)

    strat_ret = (pos1 * r1 + pos2 * r2).fillna(0.0)
    out["strategy_ret"] = strat_ret
    out["cumret"] = out["strategy_ret"].cumsum()

    return out