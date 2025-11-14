import os, warnings
warnings.filterwarnings("ignore")
os.environ["PYTHONWARNINGS"] = "ignore"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from collections import deque
import pywt

tqdm.monitor_interval = 0

# ---------------- Config ----------------
START_FROM_IDX = 0

DATA_DIR   = "./download"  
OUTPUT_DIR = "./xlstm_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

ALL_FACTORS_FILE = os.path.join(DATA_DIR, "all_factors_complete.csv")
CLOSE_PRICES_FILE = os.path.join(DATA_DIR, "close_prices.csv")

START = "2010-01-01"
END   = None  

TRAIN_END = "2020-12-31"
VAL_END   = "2022-12-31" 
SEQ_LEN        = 30  # Use 30 like original xLSTM code
MIN_TRAIN_DAYS = 40

BATCH_SIZE   = 32
LR           = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS   = 50
PATIENCE     = 10
HIDDEN_SIZE  = 128
NUM_LAYERS   = 2
DROPOUT      = 0.2
CLIP_NORM    = 1.0

# Calibration and stability
RIDGE_L2      = 5e-6
B_CAP         = 0.17
RHAT_CAP      = 0.023
CAL_WIN       = 40
MIN_CAL_SAMPLES = 40

# Online mean matching
MEAN_MATCH_WIN = 40
MEAN_MATCH_MIN = 20

# Display rebasing (for visualization only)
REBASE_DAYS_FOR_CURVE = 20

# Wavelet denoising (optional, set to None to disable)
USE_WAVELET = True
WAVELET = 'db4'
WAVELET_LEVEL = 1

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if device.type == "cuda":
    try: 
        torch.set_float32_matmul_precision("high")
    except: 
        pass
    torch.backends.cudnn.benchmark = True

def last_complete_day():
    return (pd.Timestamp.today().normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

if END is None: 
    END = last_complete_day()

# ==== Simple AdamW (same as reference) ====
class SimpleAdamW:
    def __init__(self, params, lr=1e-3, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8):
        plist = list(params)
        self.params = [p for p in plist if isinstance(p, torch.Tensor) and p.requires_grad]
        if not self.params:
            raise ValueError("Model has no trainable parameters.")
        self.lr, self.wd, self.betas, self.eps = lr, weight_decay, betas, eps
        self.state = {}

    def zero_grad(self, set_to_none=True):
        for p in self.params:
            if p.grad is not None:
                if set_to_none: 
                    p.grad = None
                else: 
                    p.grad.detach_()
                    p.grad.zero_()

    @torch.no_grad()
    def step(self):
        b1, b2 = self.betas
        for p in self.params:
            g = p.grad
            if g is None: 
                continue
            if self.wd:
                p.data.mul_(1 - self.lr * self.wd)
            st = self.state.get(p)
            if st is None:
                st = self.state[p] = {
                    "t": 0,
                    "m": torch.zeros_like(p, memory_format=torch.preserve_format),
                    "v": torch.zeros_like(p, memory_format=torch.preserve_format),
                }
            st["t"] += 1
            t = st["t"]
            st["m"].mul_(b1).add_(g, alpha=1 - b1)
            st["v"].mul_(b2).addcmul_(g, g, value=1 - b2)
            m_hat = st["m"] / (1 - b1**t)
            v_hat = st["v"] / (1 - b2**t)
            p.data.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)

# ---------------- Tech Indicators ----------------
def rsi(series, n=14):
    d = series.diff()
    up = d.clip(lower=0)
    down = -d.clip(upper=0)
    ma_up = up.rolling(n, min_periods=n).mean()
    ma_dn = down.rolling(n, min_periods=n).mean()
    rs = ma_up / ma_dn
    return 100 - (100/(1+rs))

def macd(close, fast=12, slow=26, signal=9):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    line  = ema_f - ema_s
    sig   = line.ewm(span=signal, adjust=False).mean()
    hist  = line - sig
    return line, sig, hist

# Online mean matching
def online_mean_match(curr_rhat, ret_hist: deque, rhat_hist: deque):
    if MEAN_MATCH_WIN is None or len(ret_hist) < MEAN_MATCH_MIN or len(rhat_hist) == 0:
        return float(curr_rhat)
    mu_true = float(np.mean(ret_hist))
    mu_pred = float(np.mean(rhat_hist))
    return float(curr_rhat + (mu_true - mu_pred))

# Wavelet denoising
def wavelet_denoise(data, wavelet='db4', level=1):
    coeffs = pywt.wavedec(data, wavelet, level=level)
    threshold = np.std(coeffs[-1]) * np.sqrt(2 * np.log(len(data)))
    coeffs[1:] = [pywt.threshold(c, threshold, mode='soft') for c in coeffs[1:]]
    return pywt.waverec(coeffs, wavelet)

# ---------------- sLSTM Cell (xLSTM) ----------------
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

# ---------------- xLSTM Classification Model ----------------
class xLSTMTrend(nn.Module):
    """xLSTM for binary classification (up/down)"""
    def __init__(self, in_dim, hidden=128, layers=2, dropout=0.2):
        super().__init__()
        self.xlstm = xLSTMLayer(input_size=in_dim, hidden_size=hidden, 
                                num_layers=layers, dropout=dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden, 1))
    
    def forward(self, x):
        y, _ = self.xlstm(x)  # (B,T,H)
        last = y[:, -1, :]
        logit = self.head(last)  # (B,1)
        return logit.squeeze(1)

def make_seq_2d_to_3d(X2d: np.ndarray, seq_len: int):
    Xs = []
    for i in range(seq_len-1, len(X2d)):
        Xs.append(X2d[i-seq_len+1:i+1, :])
    return np.stack(Xs) if len(Xs)>0 else np.zeros((0, seq_len, X2d.shape[1]), dtype=X2d.dtype)

def align_y(y: np.ndarray, seq_len: int):
    return y[seq_len-1:]

def get_by_key(series: pd.Series, key):
    return series.iloc[key] if isinstance(key, (int, np.integer)) else series.loc[key]

# ----------- Online Calibrator -----------
class OnlineCalibrator:
    """
    Maintains window of (z, r), returns closed-form ridge regression (a_t, b_t).
    z = p-0.5, r is next-day actual return (revealed yesterday).
    """
    def __init__(self, ridge=1e-6, cap_b=0.10, win=252):
        self.ridge = ridge
        self.cap_b = cap_b
        self.win   = win if (win is None or win > 0) else None

        self.buf = deque()

        # Cumulative sums
        self.n = 0
        self.sz = 0.0
        self.sr = 0.0
        self.szz = 0.0
        self.szr = 0.0

    def _push(self, z, r):
        self.buf.append((z, r))
        self.n  += 1
        self.sz += z
        self.sr += r
        self.szz += z*z
        self.szr += z*r

    def _pop_left(self):
        if self.win is None: 
            return
        while self.n > self.win:
            oldz, oldr = self.buf.popleft()
            self.n  -= 1
            self.sz -= oldz
            self.sr -= oldr
            self.szz -= oldz*oldz
            self.szr -= oldz*oldr

    def add(self, z, r):
        self._push(float(z), float(r))
        self._pop_left()

    def fit_from_arrays(self, z_arr, r_arr):
        """Initialize from historical arrays (train+valid), trim to window."""
        if len(z_arr) != len(r_arr):
            raise ValueError("z_arr and r_arr must have the same length")

        # Only take last win samples
        if self.win is not None and len(z_arr) > self.win:
            z_arr = z_arr[-self.win:]
            r_arr = r_arr[-self.win:]

        self.buf.clear()
        self.buf.extend((float(z), float(r)) for z, r in zip(z_arr, r_arr))

        self.n = len(self.buf)
        if self.n == 0:
            self.sz = self.sr = self.szz = self.szr = 0.0
            return

        zs = [z for z, _ in self.buf]
        rs = [r for _, r in self.buf]
        self.sz  = float(np.sum(zs))
        self.sr  = float(np.sum(rs))
        self.szz = float(np.dot(zs, zs))
        self.szr = float(np.dot(zs, rs))

    def coef(self):
        if self.n < MIN_CAL_SAMPLES:
            return None
        n, sz, sr, szz, szr = self.n, self.sz, self.sr, self.szz, self.szr
        a00 = n + self.ridge
        a01 = sz
        a11 = szz + self.ridge
        b0, b1 = sr, szr
        det = a00*a11 - a01*a01
        if det <= 1e-18:
            return 0.0, 0.0
        a = ( b0*a11 - b1*a01) / det
        b = ( a00*b1 - a01*b0) / det
        b = float(np.clip(b, -self.cap_b, self.cap_b))
        return float(a), b

# ---------------- Training (Train+Valid only) ----------------
def train_with_val(X_train3, y_train1, X_val3, y_val1, n_feat_for_model):
    ds_tr = torch.utils.data.TensorDataset(
        torch.from_numpy(X_train3.astype(np.float32)),
        torch.from_numpy(y_train1.astype(np.float32)))
    ds_va = torch.utils.data.TensorDataset(
        torch.from_numpy(X_val3.astype(np.float32)),
        torch.from_numpy(y_val1.astype(np.float32)))
    
    pin = (device.type == "cuda")
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=BATCH_SIZE, shuffle=True,
                                        drop_last=False, pin_memory=pin)
    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=2048, shuffle=False,
                                        drop_last=False, pin_memory=pin)

    model = xLSTMTrend(n_feat_for_model, hidden=HIDDEN_SIZE, layers=NUM_LAYERS, dropout=DROPOUT).to(device)
    opt = SimpleAdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # Weighted BCE loss for imbalanced classes
    pos_weight = max(1.0, (len(y_train1)-y_train1.sum())/max(1.0, y_train1.sum()))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    best_state, best_val, bad = None, float("inf"), 0
    
    for epoch in range(MAX_EPOCHS):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            opt.step()

        model.eval()
        vs = []
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                vs.append(loss_fn(model(xb), yb).item())
        v = float(np.mean(vs))
        
        if v < best_val - 1e-9:
            best_val, bad = v, 0
            best_state = {k: w.detach().clone() for k, w in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE: 
                break

    if best_state is not None: 
        model.load_state_dict(best_state)
    model.eval()
    return model

@torch.no_grad()
def predict_proba(model, X3):
    if len(X3) == 0: 
        return np.array([], dtype=np.float32)
    tens = torch.from_numpy(X3.astype(np.float32)).to(device)
    return 1/(1+np.exp(-model(tens).cpu().numpy()))

# ---------------- Load all data ----------------
if not os.path.exists(ALL_FACTORS_FILE):
    raise FileNotFoundError(f"File not found: {ALL_FACTORS_FILE}")
if not os.path.exists(CLOSE_PRICES_FILE):
    raise FileNotFoundError(f"File not found: {CLOSE_PRICES_FILE}")

print("Loading factor data...")
all_factors = pd.read_csv(ALL_FACTORS_FILE, index_col=0, parse_dates=True)
all_factors = all_factors.sort_index()
all_factors = all_factors.loc[(all_factors.index >= START) & (all_factors.index <= END)]

print("Loading close price data...")
close_prices = pd.read_csv(CLOSE_PRICES_FILE, index_col=0, parse_dates=True)
close_prices = close_prices.sort_index()
close_prices = close_prices.loc[(close_prices.index >= START) & (close_prices.index <= END)]

# Get all tickers from close_prices
all_tickers = close_prices.columns.tolist()
print(f"Tickers detected: {len(all_tickers)}; start from #{START_FROM_IDX+1}")

summary = []

# ---------------- Main loop: process each ticker ----------------
for TICKER in tqdm(all_tickers[START_FROM_IDX:], desc=f"All tickers [{START_FROM_IDX+1}→{len(all_tickers)}]"):
    try:
        # Get close price series for this ticker
        if TICKER not in close_prices.columns:
            print(f"\n[{TICKER}] not found in close_prices, skip.")
            continue
        
        close_s = pd.to_numeric(close_prices[TICKER], errors="coerce")
        
        # Check if we have enough data
        if close_s.dropna().shape[0] < (SEQ_LEN + MIN_TRAIN_DAYS + 50):
            print(f"\n[{TICKER}] too few points, skip.")
            continue

        # ----- Targets -----
        y_ret = close_s.shift(-1)/close_s - 1.0
        y_cls = (y_ret > 0).astype(int)

        selected_features = [

            "ret_5d", "ret_20d", "ret_60d", "mom_accel",

            "RSI", "MACD_hist", "MA20_bias", "BB_position", "MA_align",

            "vol_20d", "vol_60d", "ATR",

            "volume_ratio", "volume_mom",

            "52w_position", "from_52w_high",

            "hl_spread"
        ]


        ticker_cols = [col for col in all_factors.columns if col.startswith(f"{TICKER}_")]

        if len(ticker_cols) == 0:
            print(f"\n[{TICKER}] no features found, skip.")
        else:

            selected_cols = [f"{TICKER}_{feat}" for feat in selected_features if f"{TICKER}_{feat}" in ticker_cols]
    
            feat = all_factors[selected_cols].copy()
    
            print(f"[{TICKER}] Selected {len(selected_cols)} features: {selected_cols}")

        feat = feat.shift(1)

        # ----- Align -----
        data = pd.concat([feat, y_ret.rename("y_ret"), y_cls.rename("y_cls"), close_s.rename("close")], axis=1).dropna()
        
        if data.shape[0] < (SEQ_LEN + MIN_TRAIN_DAYS + 10):
            print(f"\n[{TICKER}] not enough aligned samples, skip.")
            continue

        X_all   = data.drop(columns=['y_ret','y_cls','close'])
        ret_all = data['y_ret']
        cls_all = data['y_cls'].astype(int)
        close_all = data['close']
        dates = X_all.index
        n_feat = X_all.shape[1]

        # ----- Split by date -----
        train_idx = dates[dates <= pd.Timestamp(TRAIN_END)]
        val_idx   = dates[(dates > pd.Timestamp(TRAIN_END)) & (dates <= pd.Timestamp(VAL_END))]
        test_idx  = dates[dates > pd.Timestamp(VAL_END)]

        if len(train_idx) < SEQ_LEN + MIN_TRAIN_DAYS or len(val_idx) < SEQ_LEN//2 or len(test_idx) < SEQ_LEN:
            print(f"\n[{TICKER}] split too short (train/val/test), skip.")
            continue

        X_train = X_all.loc[train_idx]
        y_train_cls = cls_all.loc[train_idx]
        y_train_ret = ret_all.loc[train_idx]
        
        X_val   = X_all.loc[val_idx]
        y_val_cls   = cls_all.loc[val_idx]
        y_val_ret   = ret_all.loc[val_idx]
        
        X_test  = X_all.loc[test_idx]
        y_test_ret  = ret_all.loc[test_idx]

        # ----- Standardization (fit on train only) -----
        scaler = StandardScaler().fit(X_train.values)

        Xtr2 = scaler.transform(X_train.values)
        Xva2 = scaler.transform(X_val.values)
        
        Xtr3 = make_seq_2d_to_3d(Xtr2, SEQ_LEN)
        Xva3 = make_seq_2d_to_3d(Xva2, SEQ_LEN)
        
        ytr1 = align_y(y_train_cls.values.astype(np.float32), SEQ_LEN)
        yva1 = align_y(y_val_cls.values.astype(np.float32), SEQ_LEN)
        
        # Apply wavelet denoising to training targets (optional)
        if USE_WAVELET:
            ytr1 = wavelet_denoise(ytr1, WAVELET, WAVELET_LEVEL)

        # ----- Train on Train+Valid with early stopping -----
        model = train_with_val(Xtr3, ytr1, Xva3, yva1, n_feat_for_model=n_feat)

        # ----- Initialize online calibrator with Train+Valid z,r -----
        z_seed, r_seed = [], []
        
        # Seed with train
        p_tr  = predict_proba(model, make_seq_2d_to_3d(Xtr2, SEQ_LEN))
        r_tr  = align_y(y_train_ret.values.astype(np.float32), SEQ_LEN)
        z_seed.append(p_tr - 0.5)
        r_seed.append(r_tr)
        
        # Seed with valid
        p_va  = predict_proba(model, make_seq_2d_to_3d(Xva2, SEQ_LEN))
        r_va  = align_y(y_val_ret.values.astype(np.float32), SEQ_LEN)
        z_seed.append(p_va - 0.5)
        r_seed.append(r_va)

        z_seed = np.concatenate(z_seed) if len(z_seed)>0 else np.zeros(0)
        r_seed = np.concatenate(r_seed) if len(r_seed)>0 else np.zeros(0)

        cal = OnlineCalibrator(ridge=RIDGE_L2, cap_b=B_CAP, win=CAL_WIN)
        if len(z_seed) > 0: 
            cal.fit_from_arrays(z_seed, r_seed)

        # ======= Test period: fixed model parameters, rolling prediction =======
        pred_prob = pd.Series(index=test_idx, dtype=float)
        pred_rhat = pd.Series(index=test_idx, dtype=float)

        ret_hist  = deque(maxlen=MEAN_MATCH_WIN)
        rhat_hist = deque(maxlen=MEAN_MATCH_WIN)

        z_prev, prev_j = None, None
        
        for j in test_idx:
            X_hist = X_all.loc[:j].values
            if len(X_hist) < SEQ_LEN: 
                continue
            
            X_hist_sc = scaler.transform(X_hist)
            X_last3   = make_seq_2d_to_3d(X_hist_sc, SEQ_LEN)[-1:,:,:]

            p = float(predict_proba(model, X_last3)[0])
            z = p - 0.5

            coef = cal.coef()
            a_t, b_t = (0.0, 0.0) if coef is None else coef

            rhat = a_t + b_t * z
            rhat = online_mean_match(rhat, ret_hist, rhat_hist)
            rhat = float(np.clip(rhat, -RHAT_CAP, RHAT_CAP))

            pred_prob.loc[j] = p
            pred_rhat.loc[j] = rhat

            # Update with yesterday's actual r and r̂
            if z_prev is not None and prev_j is not None:
                cal.add(z_prev, float(get_by_key(ret_all, prev_j)))
                ret_hist.append(float(get_by_key(ret_all, prev_j)))
                rhat_hist.append(float(pred_rhat.loc[prev_j]))

            z_prev, prev_j = z, j

        # ---------------- OOS alignment and "synthetic price" (test period) ----------------
        valid = pred_prob.dropna().index
        if len(valid)==0:
            print(f"\n[{TICKER}] no valid test predictions, skip.")
            continue

        ret_oos   = y_test_ret.loc[valid]
        close_oos = close_all.loc[valid]

        # Price takes effect next day: shift(1); anchor to first day actual close
        P0 = float(close_oos.iloc[0])
        pred_factor = (1.0 + pred_rhat.loc[valid]).cumprod().shift(1).fillna(1.0)
        pred_price  = pd.Series(P0, index=pred_factor.index) * pred_factor

        # Scheme A: block rebasing (for visualization/statistics only)
        dfp = pd.DataFrame({"actual": close_oos, "pred": pred_price}).dropna().copy()
        blocks_curve = np.arange(len(dfp)) // REBASE_DAYS_FOR_CURVE
        
        def _rebase_block(g: pd.DataFrame):
            s = float(g["actual"].iloc[0] / g["pred"].iloc[0])
            g["pred_rb"] = g["pred"] * s
            return g
        
        dfp = dfp.groupby(blocks_curve, group_keys=False).apply(_rebase_block)
        pred_price_rb = dfp["pred_rb"]

        # ---------------- Output CSV (test period: Date, PredictedPrice) ----------------
        out = pd.DataFrame({"Date": valid, "PredictedPrice": pred_price.loc[valid].values})
        out.to_csv(os.path.join(OUTPUT_DIR, f"{TICKER}.csv"), index=False)
        print(f"\n[{TICKER}] saved -> {TICKER}.csv  (test period)")

        # ---------------- OOS metrics (returns) ----------------
        mse_model = float(((ret_oos - pred_rhat.loc[valid])**2).mean())
        mse_base  = float(((ret_oos - ret_oos.mean())**2).mean())
        r2_oos    = 1 - mse_model/mse_base if mse_base>0 else np.nan
        ic        = float(pd.Series(pred_rhat.loc[valid]).corr(ret_oos))
        hit       = float((np.sign(pred_rhat.loc[valid])==np.sign(ret_oos)).mean())

        # Directional strategy annualized Sharpe (for reference)
        str_ret = np.sign(pred_rhat.loc[valid]) * ret_oos
        ann_ret = (1+str_ret.mean())**252 - 1
        ann_vol = str_ret.std(ddof=0) * np.sqrt(252)
        sharpe  = float(ann_ret/ann_vol) if ann_vol>0 else np.nan

        print(f"[{TICKER}] TEST  MSE={mse_model:.3e} | R2={r2_oos: .3f} | IC={ic: .3f} | Hit={hit: .3f} | Sharpe={sharpe: .2f}")

        summary.append([TICKER, mse_model, r2_oos, ic, hit, sharpe, valid[0].date(), valid[-1].date(), len(valid)])

        # ---------------- Plot: Actual vs Predicted (rebased) ----------------
        plt.figure(figsize=(10,3.2))
        plt.plot(close_oos.index, close_oos.values, label="Actual Close (USD)")
        plt.plot(pred_price_rb.index, pred_price_rb.values,
                 label=f"Predicted (rebased {REBASE_DAYS_FOR_CURVE}d)")
        plt.title(f"{TICKER} | Actual vs Predicted (TEST)")
        plt.ylabel("USD")
        plt.legend()
        plt.grid(True, alpha=0.3)
        fn = os.path.join(OUTPUT_DIR, f"{TICKER}_TEST_curve.png")
        plt.tight_layout()
        plt.savefig(fn, dpi=140)
        plt.close()
        print(f"[{TICKER}] saved -> {os.path.basename(fn)}")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"\n[{TICKER}] error: {e}")
        import traceback
        traceback.print_exc()
        continue

# ---------------- Summary output ----------------
sum_df = pd.DataFrame(summary, columns=[
    "Ticker","MSE","R2_TEST","IC","HitRate","Sharpe","TEST_start","TEST_end","#TEST_days"
])
sum_df = sum_df.sort_values("R2_TEST", ascending=False)
sum_df.to_csv(os.path.join(OUTPUT_DIR, "OOS_summary.csv"), index=False)
print("\nSaved OOS_summary.csv")
print(sum_df.head(20).to_string(index=False))