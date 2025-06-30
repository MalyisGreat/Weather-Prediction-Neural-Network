#!/usr/bin/env python3
"""
weather_unified.py

A unified, resumable script for training an advanced TCN+MDN forecaster.
- Automatically starts from scratch or resumes from a checkpoint.
- Trains until the EarlyStopping condition is met.
- Saves the best model weights continuously.
- Creates a checkpoint after every epoch for maximum robustness.
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn import __version__ as skl_ver
from packaging import version
from sklearn.metrics import mean_squared_error, r2_score
from torch.distributions import Categorical, Normal, Independent, MixtureSameFamily
from joblib import dump, load
import matplotlib.pyplot as plt

# 1. CONFIGURATION
SEQ_LEN         = 72
HORIZON         = 24
BATCH_SIZE      = 64
MAX_EPOCHS      = 300    # A high number; early stopping will finish the job
LR              = 5e-4
WEIGHT_DECAY    = 1e-5
PATIENCE        = 15     # More patience for fine-tuning phase
K_MIX           = 5
DEVICE          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# File Paths
MODEL_DIR       = "models"
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "weather_tcn_mdn_checkpoint.pth")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "weather_tcn_mdn_best.pth")
FULL_MODEL_PATH = os.path.join(MODEL_DIR, "weather_tcn_mdn_full.pth")
SCALER_PATH     = os.path.join(MODEL_DIR, "weather_scaler.pkl")
ENCODER_PATH    = os.path.join(MODEL_DIR, "weather_encoder.pkl")

os.makedirs(MODEL_DIR, exist_ok=True)
print(f"Using device: {DEVICE}")

# --- DATA LOADING AND PREPROCESSING (This section runs every time) ---
# 2. LOAD DATA
url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00381/"
    "PRSA_data_2010.1.1-2014.12.31.csv"
)
df = pd.read_csv(url)
df["datetime"] = pd.to_datetime(df[["year","month","day","hour"]])
df.set_index("datetime", inplace=True)

# 3. FEATURE ENGINEERING
df["pm2.5_lag1"] = df["pm2.5"].shift(1)
df["RH"] = 100 * np.exp(
    (17.27 * df["DEWP"]) / (df["DEWP"] + 237.3)
    - (17.27 * df["TEMP"]) / (df["TEMP"] + 237.3)
)
df["hour"]  = df.index.hour
df["dow"]   = df.index.dayofweek
df["month"] = df.index.month
for col, period in [("hour",24), ("dow",7), ("month",12)]:
    df[f"{col}_sin"] = np.sin(2 * np.pi * df[col] / period)
    df[f"{col}_cos"] = np.cos(2 * np.pi * df[col] / period)
for feat in ["TEMP","DEWP"]:
    df[f"{feat}_rm3"]   = df[feat].rolling(3,  min_periods=1).mean()
    df[f"{feat}_std3"]  = df[feat].rolling(3,  min_periods=1).std().fillna(0)
    df[f"{feat}_rm24"]  = df[feat].rolling(24, min_periods=1).mean()
    df[f"{feat}_std24"] = df[feat].rolling(24, min_periods=1).std().fillna(0)

# 4. SELECT FEATURES & DROP NA
num_feats = [
    "DEWP","TEMP","PRES","Iws","Is","Ir", "pm2.5_lag1","RH",
    "hour_sin","hour_cos","dow_sin","dow_cos","month_sin","month_cos",
    "TEMP_rm3","TEMP_std3","TEMP_rm24","TEMP_std24",
    "DEWP_rm3","DEWP_std3","DEWP_rm24","DEWP_std24"
]
cat_feats = ["cbwd"]
df = df[num_feats + cat_feats].dropna()

# 5. TRAIN/TEST SPLIT & PREPROCESSING
n      = len(df)
split  = int(n * 0.8)
train_df = df.iloc[:split]
test_df  = df.iloc[split - (SEQ_LEN + HORIZON):]

# Handle Scalers/Encoders: Fit or Load
if not os.path.exists(SCALER_PATH):
    print("Fitting and saving new scaler/encoder.")
    scaler = MinMaxScaler()
    train_num = scaler.fit_transform(train_df[num_feats])
    dump(scaler, SCALER_PATH)
    if version.parse(skl_ver) >= version.parse("1.2"):
        encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    else:
        encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
    train_cat = encoder.fit_transform(train_df[cat_feats])
    dump(encoder, ENCODER_PATH)
else:
    print("Loading existing scaler/encoder.")
    scaler = load(SCALER_PATH)
    encoder = load(ENCODER_PATH)
    train_num = scaler.transform(train_df[num_feats])
    train_cat = encoder.transform(train_df[cat_feats])

test_num  = scaler.transform(test_df[num_feats])
test_cat  = encoder.transform(test_df[cat_feats])
train_mat = np.hstack([train_num, train_cat])
test_mat  = np.hstack([test_num,  test_cat])
TOTAL_FEATS = train_mat.shape[1]

# 6. DATASET & DATALOADER
def create_windows(mat, seq_len, horizon):
    X, y = [], []
    for i in range(len(mat) - seq_len - horizon + 1):
        X.append(mat[i:i+seq_len])
        y.append(mat[i+seq_len+horizon-1])
    return np.array(X), np.array(y)
X_train, y_train = create_windows(train_mat, SEQ_LEN, HORIZON)
X_val,   y_val   = create_windows(test_mat,  SEQ_LEN, HORIZON)

class TSData(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]
train_loader = DataLoader(TSData(X_train,y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TSData(X_val,  y_val), batch_size=BATCH_SIZE)

# --- MODEL DEFINITION (Identical to previous script) ---
class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv1   = weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation))
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.downsample = (nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None)
    def forward(self, x):
        x_pad = F.pad(x, (self.padding, 0))
        out   = self.conv1(x_pad)
        out   = self.activation(out)
        out   = self.dropout(out)
        res   = x if self.downsample is None else self.downsample(x)
        return out + res

class TCN_MDN(nn.Module):
    def __init__(self, n_feats, channels=[64, 64, 64], kernel_size=3, drop=0.2, n_mix=K_MIX):
        super().__init__()
        layers, in_ch = [], n_feats
        for i,ch in enumerate(channels):
            layers.append(TCNBlock(in_ch, ch, kernel_size, dilation=2**i, dropout=drop))
            in_ch = ch
        self.tcn = nn.Sequential(*layers)
        self.ln = nn.LayerNorm(in_ch)
        self.drop = nn.Dropout(drop)
        self.pi, self.mu, self.sig = nn.Linear(in_ch, n_mix), nn.Linear(in_ch, n_mix*n_feats), nn.Linear(in_ch, n_mix*n_feats)
        self.n_mix = n_mix
    def forward(self, x):
        x = x.transpose(1,2)
        y = self.tcn(x)
        h = self.ln(y[:,:,-1])
        h = self.drop(h)
        pi_logits = self.pi(h)
        mu = self.mu(h).view(-1,self.n_mix,x.size(1))
        sigma = F.softplus(self.sig(h).view(-1,self.n_mix,x.size(1))) + 1e-6
        cat, comp = Categorical(logits=pi_logits), Independent(Normal(mu, sigma), 1)
        return MixtureSameFamily(cat, comp)

# --- INITIALIZE OR RESUME ---
model     = TCN_MDN(TOTAL_FEATS).to(DEVICE)
optimizer = Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=PATIENCE // 2)
nll_loss  = lambda dist,y: -dist.log_prob(y).mean()

start_epoch = 1
best_val = float("inf")
history = {"train_nll":[], "val_nll":[], "val_rmse":[], "val_r2":[]}

if os.path.exists(CHECKPOINT_PATH):
    print("--- Checkpoint found! Resuming training. ---")
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    history = checkpoint['history']
    best_val = checkpoint['best_val_loss']
    print(f"Resuming from epoch {start_epoch}. Best validation loss so far: {best_val:.4f}")
    print(f"Current learning rate: {optimizer.param_groups[0]['lr']:.2e}")
else:
    print("--- No checkpoint found. Starting training from scratch. ---")

# --- TRAINING LOOP ---
no_imp = 0
for ep in range(start_epoch, MAX_EPOCHS + 1):
    model.train()
    tot_nll = 0.0
    for xb,yb in train_loader:
        xb,yb = xb.to(DEVICE), yb.to(DEVICE)
        dist  = model(xb)
        loss  = nll_loss(dist,yb)
        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        tot_nll += loss.item()*xb.size(0)
    train_nll = tot_nll/len(train_loader.dataset)

    model.eval()
    tot_nll, preds, trues = 0.0, [], []
    with torch.no_grad():
        for xb,yb in val_loader:
            xb,yb = xb.to(DEVICE), yb.to(DEVICE)
            dist  = model(xb)
            tot_nll += nll_loss(dist,yb).item()*xb.size(0)
            preds.append(dist.mean.cpu().numpy())
            trues.append(yb.cpu().numpy())
    val_nll = tot_nll/len(val_loader.dataset)
    y_true, y_pred = np.vstack(trues), np.vstack(preds)
    val_rmse, val_r2 = np.sqrt(mean_squared_error(y_true,y_pred)), r2_score(y_true,y_pred)

    history["train_nll"].append(train_nll)
    history["val_nll"].append(val_nll)
    history["val_rmse"].append(val_rmse)
    history["val_r2"].append(val_r2)

    print(f"Ep{ep:3d} ▶ train_NLL={train_nll:.4f}  "
          f"val_NLL={val_nll:.4f}  RMSE={val_rmse:.3f}  R2={val_r2:.3f}  "
          f"LR={optimizer.param_groups[0]['lr']:.2e}")

    # Save checkpoint after every epoch
    checkpoint = {
        'epoch': ep,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'history': history,
        'best_val_loss': best_val
    }
    torch.save(checkpoint, CHECKPOINT_PATH)

    scheduler.step(val_nll)
    if val_nll < best_val:
        best_val, no_imp = val_nll, 0
        torch.save(model.state_dict(), BEST_MODEL_PATH)
        print("  ✓ Saved new best weights")
    else:
        no_imp += 1
        if no_imp >= PATIENCE:
            print(f"\nStopping early at epoch {ep} due to no improvement for {PATIENCE} epochs.")
            break

print("\n--- Training Finished ---")
# --- FINAL SAVE & PLOT ---
torch.save(model, FULL_MODEL_PATH)
print(f"✓ Final model saved to {FULL_MODEL_PATH}")

eps = list(range(1, len(history["train_nll"]) + 1))
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(eps, history["train_nll"], label="train NLL")
plt.plot(eps, history["val_nll"],   label="val NLL")
if start_epoch > 1:
    plt.axvline(x=start_epoch-1, color='r', linestyle='--', label=f'Resume @ Ep {start_epoch}')
plt.title("Negative Log‑Likelihood"); plt.xlabel("Epoch"); plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(1,2,2)
plt.plot(eps, history["val_r2"], label="val R²", color='green')
if start_epoch > 1:
    plt.axvline(x=start_epoch-1, color='r', linestyle='--', label=f'Resume @ Ep {start_epoch}')
plt.title("Validation R²"); plt.xlabel("Epoch"); plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout();
plt.show()