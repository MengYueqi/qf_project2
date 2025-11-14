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

tqdm.monitor_interval = 0

# ---------------- Configuration ----------------
START_FROM_IDX = 0

DATA_DIR   = "./download"  
OUTPUT_DIR = "./output_final"
PRICE_FILE = os.path.join(DATA_DIR, "price_data_full.csv")

START      = "2010-01-01"
END        = None  

# Fixed split boundaries
TRAIN_END = "2020-12-31"
VAL_END   = "2022-12-31"   # After VAL_END is Testing

# Sequence and rolling window parameters
SEQ_LEN        = 30  # Match your original SEQUENCE_LENGTH
MIN_TRAIN_DAYS = 40

# Training hyperparameters
BATCH_SIZE   = 32
LR           = 1e-3
WEIGHT_DECAY = 1e-4
MAX_EPOCHS   = 100
PATIENCE     = 15
DROPOUT      = 0.3

# CNN-BiLSTM architecture (matching your original config)
CNN_FILTERS = [64, 128, 256]
KERNEL_SIZES = [3, 5, 7]
LSTM_HIDDEN = 128
LSTM_LAYERS = 2

CLIP_NORM    = 1.0

# Calibration and stability (safety threshold)
RIDGE_L2      = 5e-6
B_CAP         = 0.17
RHAT_CAP      = 0.023
CAL_WIN       = 40
MIN_CAL_SAMPLES = 40

# Online mean matching
MEAN_MATCH_WIN = 40
MEAN_MATCH_MIN = 20

# Display rebasing scheme
REBASE_DAYS_FOR_CURVE = 20

np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if device.type == "cuda":
    try: torch.set_float32_matmul_precision("high")
    except: pass
    torch.backends.cudnn.benchmark = True

def last_complete_day():
    return (pd.Timestamp.today().normalize() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
if END is None: END = last_complete_day()

# ==== Pure PyTorch AdamW (no dependency on torch.optim.*) ====
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
                if set_to_none: p.grad = None
                else: p.grad.detach_(); p.grad.zero_()

    @torch.no_grad()
    def step(self):
        b1, b2 = self.betas
        for p in self.params:
            g = p.grad
            if g is None: continue
            if self.wd:
                p.data.mul_(1 - self.lr * self.wd)
            st = self.state.get(p)
            if st is None:
                st = self.state[p] = {
                    "t": 0,
                    "m": torch.zeros_like(p, memory_format=torch.preserve_format),
                    "v": torch.zeros_like(p, memory_format=torch.preserve_format),
                }
            st["t"] += 1; t = st["t"]
            st["m"].mul_(b1).add_(g, alpha=1 - b1)
            st["v"].mul_(b2).addcmul_(g, g, value=1 - b2)
            m_hat = st["m"] / (1 - b1**t)
            v_hat = st["v"] / (1 - b2**t)
            p.data.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)

# ---------------- Technical Indicators ----------------
def rsi(series, n=14):
    """Calculate Relative Strength Index"""
    d = series.diff()
    up = d.clip(lower=0); down = -d.clip(upper=0)
    ma_up = up.rolling(n, min_periods=n).mean()
    ma_dn = down.rolling(n, min_periods=n).mean()
    rs = ma_up / ma_dn
    return 100 - (100/(1+rs))

def macd(close, fast=12, slow=26, signal=9):
    """Calculate MACD (Moving Average Convergence Divergence)"""
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    line  = ema_f - ema_s
    sig   = line.ewm(span=signal, adjust=False).mean()
    hist  = line - sig
    return line, sig, hist

# Online mean matching
def online_mean_match(curr_rhat, ret_hist: deque, rhat_hist: deque):
    """Adjust predicted returns by matching historical means"""
    if MEAN_MATCH_WIN is None or len(ret_hist) < MEAN_MATCH_MIN or len(rhat_hist) == 0:
        return float(curr_rhat)
    mu_true = float(np.mean(ret_hist))
    mu_pred = float(np.mean(rhat_hist))
    return float(curr_rhat + (mu_true - mu_pred))

# ---------------- CNN-BiLSTM Model ----------------
class MultiScaleCNN(nn.Module):
    """Multi-scale Convolutional Neural Network for feature extraction"""
    def __init__(self, input_size, filters, kernel_sizes, dropout=0.2):
        super(MultiScaleCNN, self).__init__()
        self.convs = nn.ModuleList()
        for kernel_size in kernel_sizes:
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

        self.fusion = nn.Conv1d(filters[2] * len(kernel_sizes), filters[2], 1)
        self.fusion_bn = nn.BatchNorm1d(filters[2])

    def forward(self, x):
        # x: (B, T, F) -> transpose to (B, F, T)
        x = x.transpose(1, 2)
        conv_outputs = []
        for conv in self.convs:
            conv_outputs.append(conv(x))
        concatenated = torch.cat(conv_outputs, dim=1)
        fused = self.fusion(concatenated)
        fused = self.fusion_bn(fused)
        fused = torch.relu(fused)
        # (B, F, T) -> (B, T, F)
        fused = fused.transpose(1, 2)
        return fused

class CNNBiLSTMClassifier(nn.Module):
    """
    CNN-BiLSTM for binary classification (up/down)
    Output: single logit for BCEWithLogitsLoss
    """
    def __init__(self, input_size, cnn_filters, kernel_sizes,
                 lstm_hidden, lstm_layers, dropout):
        super(CNNBiLSTMClassifier, self).__init__()
        self.cnn = MultiScaleCNN(input_size, cnn_filters, kernel_sizes, dropout)
        self.lstm = nn.LSTM(input_size=cnn_filters[-1],
                            hidden_size=lstm_hidden,
                            num_layers=lstm_layers,
                            batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0,
                            bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden * 2, lstm_hidden),
            nn.Tanh(),
            nn.Linear(lstm_hidden, 1)
        )
        # Output layer: single logit
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden * 2, 1)
        )

    def forward(self, x):
        # x: (B, T, F)
        cnn_out = self.cnn(x)              # (B, T, F')
        lstm_out, _ = self.lstm(cnn_out)   # (B, T, 2*H)
        
        # Attention pooling
        att_w = self.attention(lstm_out)  # (B, T, 1)
        att_w = torch.softmax(att_w, dim=1)
        context = torch.sum(att_w * lstm_out, dim=1)  # (B, 2*H)
        
        logit = self.head(context)         # (B, 1)
        return logit.squeeze(1)            # (B,)

def make_seq_2d_to_3d(X2d: np.ndarray, seq_len: int):
    """Convert 2D array to 3D sequences for LSTM input"""
    Xs = []
    for i in range(seq_len-1, len(X2d)):
        Xs.append(X2d[i-seq_len+1:i+1, :])
    return np.stack(Xs) if len(Xs)>0 else np.zeros((0, seq_len, X2d.shape[1]), dtype=X2d.dtype)

def align_y(y: np.ndarray, seq_len: int):
    """Align target array with sequences"""
    return y[seq_len-1:]

def get_by_key(series: pd.Series, key):
    """Get value from series by index or label"""
    return series.iloc[key] if isinstance(key, (int, np.integer)) else series.loc[key]

# ----------- Online Calibrator -----------
class OnlineCalibrator:
    """
    Maintain window of (z, r), return closed-form ridge regression solution for (a_t, b_t).
    z = p-0.5, r is next-day actual return (revealed yesterday).
    """
    def __init__(self, ridge=1e-6, cap_b=0.10, win=252):
        self.ridge = ridge
        self.cap_b = cap_b
        self.win   = win if (win is None or win > 0) else None

        # Important: no maxlen to avoid passive pop causing cumulative sum mismatch
        self.buf = deque()

        # Cumulative quantities
        self.n = 0
        self.sz = 0.0; self.sr = 0.0; self.szz = 0.0; self.szr = 0.0

    def _push(self, z, r):
        """Add new observation to buffer"""
        self.buf.append((z, r))
        self.n  += 1
        self.sz += z
        self.sr += r
        self.szz += z*z
        self.szr += z*r

    def _pop_left(self):
        """Remove oldest observations to maintain window size"""
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
        """Add single observation and maintain window"""
        self._push(float(z), float(r))
        self._pop_left()

    def fit_from_arrays(self, z_arr, r_arr):
        """Initialize with historical arrays (train+valid), trim to window"""
        if len(z_arr) != len(r_arr):
            raise ValueError("z_arr and r_arr must have the same length")

        # Only take last win entries to avoid huge history
        if self.win is not None and len(z_arr) > self.win:
            z_arr = z_arr[-self.win:]
            r_arr = r_arr[-self.win:]

        # Directly rebuild buffer and cumulative sums
        self.buf.clear()
        self.buf.extend((float(z), float(r)) for z, r in zip(z_arr, r_arr))

        self.n = len(self.buf)
        if self.n == 0:
            self.sz = self.sr = self.szz = self.szr = 0.0
            return

        # Recalculate cumulative quantities (without changing buf)
        zs = [z for z, _ in self.buf]
        rs = [r for _, r in self.buf]
        self.sz  = float(np.sum(zs))
        self.sr  = float(np.sum(rs))
        self.szz = float(np.dot(zs, zs))
        self.szr = float(np.dot(zs, rs))

    def coef(self):
        """Return calibration coefficients (a, b) using ridge regression"""
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
    """Train model on training set with validation for early stopping"""
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

    model = CNNBiLSTMClassifier(
        input_size=n_feat_for_model,
        cnn_filters=CNN_FILTERS,
        kernel_sizes=KERNEL_SIZES,
        lstm_hidden=LSTM_HIDDEN,
        lstm_layers=LSTM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    
    opt = SimpleAdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([max(1.0, (len(y_train1)-y_train1.sum())/max(1.0, y_train1.sum()))], device=device)
    )

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

        model.eval(); vs=[]
        with torch.no_grad():
            for xb, yb in dl_va:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                vs.append(loss_fn(model(xb), yb).item())
        v = float(np.mean(vs)) if len(vs) > 0 else float("inf")
        
        if v < best_val - 1e-9:
            best_val, bad = v, 0
            best_state = {k: w.detach().clone() for k, w in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE: 
                break

    if best_state is not None: model.load_state_dict(best_state)
    model.eval()
    return model

@torch.no_grad()
def predict_proba(model, X3):
    """Predict probability of upward movement"""
    if len(X3) == 0: return np.array([], dtype=np.float32)
    tens = torch.from_numpy(X3.astype(np.float32)).to(device)
    return 1/(1+np.exp(-model(tens).cpu().numpy()))

# ---------------- Load all data ----------------
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

if not os.path.exists(PRICE_FILE):
    raise FileNotFoundError(f"File not found: {PRICE_FILE}")

df = pd.read_csv(PRICE_FILE, header=[0,1], index_col=0)
if isinstance(df.index[0], str) and str(df.index[0]).strip().lower() == "date":
    df = df.iloc[1:]
df.index = pd.to_datetime(df.index, errors="coerce")
df = df[~df.index.isna()].sort_index()
df = df.loc[(df.index >= START) & (df.index <= END)]

all_tickers = df.columns.get_level_values(1).unique().tolist()
print(f"Tickers detected: {len(all_tickers)}; start from #{START_FROM_IDX+1}")

summary = []

# ---------------- Main loop: iterate through tickers ----------------
for TICKER in tqdm(all_tickers[START_FROM_IDX:], desc=f"All tickers [{START_FROM_IDX+1}→{len(all_tickers)}]"):
    try:
        close_s = pd.to_numeric(df[("Close",  TICKER)], errors="coerce")
        open_s  = pd.to_numeric(df[("Open",   TICKER)], errors="coerce")
        high_s  = pd.to_numeric(df[("High",   TICKER)], errors="coerce")
        low_s   = pd.to_numeric(df[("Low",    TICKER)], errors="coerce")
        vol_s   = pd.to_numeric(df[("Volume", TICKER)], errors="coerce")
    except KeyError:
        print(f"\n[{TICKER}] missing fields, skip."); continue

    # Need sufficient history
    if close_s.dropna().shape[0] < (SEQ_LEN + MIN_TRAIN_DAYS + 50):
        print(f"\n[{TICKER}] too few points, skip."); continue

    # ----- Targets -----
    y_ret = close_s.shift(-1)/close_s - 1.0
    y_cls = (y_ret > 0).astype(int)

    # ----- Features: Only keep specified 17 features (all shift(1) to prevent leakage) -----
    feat = pd.DataFrame(index=close_s.index)
    
    # 1. Momentum factors (4)
    feat['ret_5d']    = close_s.pct_change(5)
    feat['ret_20d']   = close_s.pct_change(20)
    feat['ret_60d']   = close_s.pct_change(60)
    ret5 = close_s.pct_change(5)
    ret20 = close_s.pct_change(20)
    feat['mom_accel'] = ret5 - ret20
    
    # 2. Technical indicators (5)
    feat['RSI']        = rsi(close_s, 14) / 100.0
    m_line, m_sig, m_hist = macd(close_s)
    feat['MACD_hist']  = m_hist
    ma20 = close_s.rolling(20).mean()
    feat['MA20_bias']  = (close_s - ma20) / ma20
    
    # Bollinger Bands
    bb_std = close_s.rolling(20).std()
    bb_upper = ma20 + 2*bb_std
    bb_lower = ma20 - 2*bb_std
    bb_range = bb_upper - bb_lower
    feat['BB_position'] = np.where(bb_range>0, (close_s - bb_lower)/bb_range, 0.5)
    
    ma5  = close_s.rolling(5).mean()
    ma10 = close_s.rolling(10).mean()
    ma60 = close_s.rolling(60).mean()
    feat['MA_align'] = ((ma5>ma10) & (ma10>ma20) & (ma20>ma60)).astype(float)
    
    # 3. Volatility (3)
    feat['vol_20d'] = close_s.pct_change().rolling(20).std()
    feat['vol_60d'] = close_s.pct_change().rolling(60).std()
    hl_range = np.log(high_s/low_s)
    feat['ATR']     = hl_range.rolling(14).mean()
    
    # 4. Volume (2)
    vol_ma20 = vol_s.rolling(20).mean()
    feat['volume_ratio'] = np.where(vol_ma20>0, vol_s/vol_ma20, 1.0)
    feat['volume_mom']   = vol_s.pct_change(5)
    
    # 5. Price position (2)
    high_52w = high_s.rolling(252, min_periods=60).max()
    low_52w  = low_s.rolling(252, min_periods=60).min()
    range_52w = high_52w - low_52w
    feat['52w_position'] = np.where(range_52w>0, (close_s - low_52w)/range_52w, 0.5)
    feat['from_52w_high'] = (close_s - high_52w)/high_52w
    
    # 6. Microstructure (1)
    feat['hl_spread'] = (high_s - low_s)/close_s

    # Keep only these 17 features
    selected_features = [
        # Momentum factors (4)
        "ret_5d", "ret_20d", "ret_60d", "mom_accel",
        # Technical indicators (5)
        "RSI", "MACD_hist", "MA20_bias", "BB_position", "MA_align",
        # Volatility (3)
        "vol_20d", "vol_60d", "ATR",
        # Volume (2)
        "volume_ratio", "volume_mom",
        # Price position (2)
        "52w_position", "from_52w_high",
        # Microstructure (1)
        "hl_spread"
    ]
    feat = feat[selected_features].copy()

    feat = feat.shift(1)  # Prevent leakage

    # ----- Align data -----
    data = pd.concat([feat, y_ret.rename("y_ret"), y_cls.rename("y_cls"), close_s.rename("close")], axis=1).dropna()
    if data.shape[0] < (SEQ_LEN + MIN_TRAIN_DAYS + 10):
        print(f"\n[{TICKER}] not enough aligned samples, skip."); continue

    X_all   = data.drop(columns=['y_ret','y_cls','close'])
    ret_all = data['y_ret']
    cls_all = data['y_cls'].astype(int)
    close_all = data['close']
    dates = X_all.index
    n_feat = X_all.shape[1]

    # ----- Split data by date -----
    train_idx = dates[dates <= pd.Timestamp(TRAIN_END)]
    val_idx   = dates[(dates > pd.Timestamp(TRAIN_END)) & (dates <= pd.Timestamp(VAL_END))]
    test_idx  = dates[dates > pd.Timestamp(VAL_END)]

    if len(train_idx) < SEQ_LEN + MIN_TRAIN_DAYS or len(val_idx) < SEQ_LEN//2 or len(test_idx) < SEQ_LEN:
        print(f"\n[{TICKER}] split too short (train/val/test), skip."); continue

    X_train = X_all.loc[train_idx]; y_train_cls = cls_all.loc[train_idx]; y_train_ret = ret_all.loc[train_idx]
    X_val   = X_all.loc[val_idx];   y_val_cls   = cls_all.loc[val_idx];   y_val_ret   = ret_all.loc[val_idx]
    X_test  = X_all.loc[test_idx];  y_test_ret  = ret_all.loc[test_idx]   # for metrics

    # ----- Standardization (fit only on training set) -----
    scaler = StandardScaler().fit(X_train.values)

    Xtr2 = scaler.transform(X_train.values); Xva2 = scaler.transform(X_val.values)
    Xtr3 = make_seq_2d_to_3d(Xtr2, SEQ_LEN); Xva3 = make_seq_2d_to_3d(Xva2, SEQ_LEN)
    ytr1 = align_y(y_train_cls.values.astype(np.float32), SEQ_LEN)
    yva1 = align_y(y_val_cls.values.astype(np.float32),   SEQ_LEN)

    # ----- Train on Train+Valid with early stopping -----
    model = train_with_val(Xtr3, ytr1, Xva3, yva1, n_feat_for_model=n_feat)

    # ----- Initialize online calibrator with Train+Valid z,r (model is frozen) -----
    # (Calibrator is online post-processing during inference, doesn't change model weights)
    z_seed, r_seed = [], []
    # seed with train
    p_tr  = predict_proba(model, make_seq_2d_to_3d(Xtr2, SEQ_LEN))
    r_tr  = align_y(y_train_ret.values.astype(np.float32), SEQ_LEN)
    z_seed.append(p_tr - 0.5); r_seed.append(r_tr)
    # seed with valid
    p_va  = predict_proba(model, make_seq_2d_to_3d(Xva2, SEQ_LEN))
    r_va  = align_y(y_val_ret.values.astype(np.float32), SEQ_LEN)
    z_seed.append(p_va - 0.5); r_seed.append(r_va)

    z_seed = np.concatenate(z_seed) if len(z_seed)>0 else np.zeros(0)
    r_seed = np.concatenate(r_seed) if len(r_seed)>0 else np.zeros(0)

    cal = OnlineCalibrator(ridge=RIDGE_L2, cap_b=B_CAP, win=CAL_WIN)
    if len(z_seed) > 0: cal.fit_from_arrays(z_seed, r_seed)

    # ======= Test period: fixed model parameters, rolling forecast by time =======
    pred_prob = pd.Series(index=test_idx, dtype=float)
    pred_rhat = pd.Series(index=test_idx, dtype=float)

    ret_hist  = deque(maxlen=MEAN_MATCH_WIN)
    rhat_hist = deque(maxlen=MEAN_MATCH_WIN)

    z_prev, prev_j = None, None
    for j in test_idx:
        X_hist = X_all.loc[:j].values
        if len(X_hist) < SEQ_LEN: continue
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

        # Update with "yesterday's" actual r and r̂ (only use revealed history, no future peeking)
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

    # Price takes effect next day: shift(1); anchor to first test day's actual close
    P0 = float(close_oos.iloc[0])
    pred_factor = (1.0 + pred_rhat.loc[valid]).cumprod().shift(1).fillna(1.0)
    pred_price  = pd.Series(P0, index=pred_factor.index) * pred_factor

    # Scheme A: rebasing by blocks (for display/statistics only)
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

    # Directional strategy annualized Sharpe (for reference only)
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
    plt.ylabel("USD"); plt.legend(); plt.grid(True, alpha=0.3)
    fn = os.path.join(OUTPUT_DIR, f"{TICKER}_TEST_curve.png")
    plt.tight_layout(); plt.savefig(fn, dpi=140); plt.close()
    print(f"[{TICKER}] saved -> {os.path.basename(fn)}")

    if device.type == "cuda":
        torch.cuda.empty_cache()

# ---------------- Summary output ----------------
sum_df = pd.DataFrame(summary, columns=[
    "Ticker","MSE","R2_TEST","IC","HitRate","Sharpe","TEST_start","TEST_end","#TEST_days"
])
sum_df = sum_df.sort_values("R2_TEST", ascending=False)
sum_df.to_csv(os.path.join(OUTPUT_DIR, "OOS_summary.csv"), index=False)
print("\nSaved OOS_summary.csv")
print(sum_df.head(20).to_string(index=False))
print("\n" + "="*80)
print("All done! Output saved to:", OUTPUT_DIR)
print("="*80)
print("\nKey points:")
print("1. ✓ Strict time series: no data leakage")
print("2. ✓ Fixed split: Train → Valid → Test")
print("3. ✓ Online calibration + mean matching")
print("4. ✓ CNN-BiLSTM classification → calibrated returns")
print("5. ✓ Rolling forecast on test set only")
