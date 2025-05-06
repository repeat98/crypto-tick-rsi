#!/usr/bin/env python3
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pyts.image import GramianAngularField
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader

# ---------------------- Argument Parsing ----------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Preprocess, label & train 6-channel GAF with custom ConvNet on 1m K-line CSVs"
    )
    p.add_argument('input_dir',  help='Directory of 1m K-line CSVs')
    p.add_argument('data_dir',   help='Where to save/load .npz tensors')
    p.add_argument('labels_dir', help='Where to save label CSVs')
    p.add_argument('--freq',      default='1T',  help="Resample frequency")
    p.add_argument('--window',    default='5T',  help="Window length")
    p.add_argument('--stride',    default='1T',  help="Stride between windows")
    p.add_argument('--horizon',   default='5T',  help="Future horizon for labeling")
    p.add_argument('--q_low',   type=float, default=0.2, help="Lower quantile")
    p.add_argument('--q_high',  type=float, default=0.8, help="Upper quantile")
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--epochs',     type=int, default=20)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--wd',         type=float, default=1e-5)
    p.add_argument('--workers',    type=int, default=os.cpu_count())
    p.add_argument('--model_out',  default='best_Ind_klines.pth')
    return p.parse_args()

# ---------------------- Utilities ----------------------

def normalize(x: np.ndarray):
    mn, mx = x.min(), x.max()
    return np.zeros_like(x) if mx == mn else ((x - mn)/(mx - mn))*2 - 1


def load_kline_csv(fp):
    with open(fp, 'r') as f:
        hdr = f.readline().strip().split(',')
    return pd.read_csv(fp, names=hdr, header=0)

# ---------------------- Preprocessing & Labeling ----------------------

def preprocess(input_dir, data_dir, freq, window, stride, image_size):
    """
    Build 6-channel GAF per sliding window: [close, MA8, MA20, RSI14, volume, ATR14]
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    gasf = GramianAngularField(method='summation')

    w_bars = int(pd.Timedelta(window) / pd.Timedelta(freq))
    s_bars = max(int(pd.Timedelta(stride) / pd.Timedelta(freq)), 1)

    for fp in tqdm(sorted(Path(input_dir).glob("*.csv")), desc="Preprocess"):
        df = load_kline_csv(fp)
        tcol = df.columns[0]
        df[tcol] = pd.to_datetime(df[tcol], unit="ms")
        df = df.set_index(tcol).sort_index()

        # resample OHLCV
        close = df['close'].resample(freq).last().ffill()
        vol   = df['volume'].resample(freq).sum().fillna(0.0)
        high  = df['high'].resample(freq).max().ffill()
        low   = df['low'].resample(freq).min().ffill()

        # compute indicators
        feat = pd.DataFrame({'close': close, 'volume': vol})
        feat['MA8']  = feat['close'].rolling(window=8,  min_periods=1).mean()
        feat['MA20'] = feat['close'].rolling(window=20, min_periods=1).mean()
        delta = feat['close'].diff().fillna(0.0)
        gain  = delta.clip(lower=0)
        loss  = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        feat['RSI14'] = (100 - 100/(1+rs))

        prev = feat['close'].shift(1).fillna(feat['close'])
        tr1 = high - low
        tr2 = (high - prev).abs()
        tr3 = (low - prev).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        feat['ATR14'] = tr.rolling(window=14, min_periods=1).mean().fillna(0.0)

        arrays = {k: feat[k].to_numpy(dtype=np.float32) for k in ['close','MA8','MA20','RSI14','volume','ATR14']}
        n = len(feat)

        for i in range(0, n - w_bars + 1, s_bars):
            out = data_dir / f"{fp.stem}_win{i:05d}.npz"

            channels = []
            for key in ['close','MA8','MA20','RSI14','volume','ATR14']:
                win = arrays[key][i:i + w_bars]
                a = normalize(win)
                g = gasf.fit_transform(a.reshape(1, -1))[0]
                if image_size and g.shape[0] != image_size:
                    from PIL import Image
                    im = Image.fromarray(((g+1)/2*255).astype(np.uint8))
                    im = im.resize((image_size, image_size), Image.BILINEAR)
                    g = (np.array(im, np.float32)/127.5) - 1.0
                channels.append(g.astype(np.float32))

            tensor = np.stack(channels, axis=0)  # (6, H, W)
            np.savez_compressed(str(out), data=tensor)


def label(input_dir, data_dir, labels_dir, freq, window, stride, horizon, ql, qh):
    """
    Save continuous returns as labels for regression.
    """
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    w_bars = int(pd.Timedelta(window) / pd.Timedelta(freq))
    s_bars = max(int(pd.Timedelta(stride) / pd.Timedelta(freq)), 1)
    h_bars = int(pd.Timedelta(horizon)/pd.Timedelta(freq))

    for fp in tqdm(sorted(Path(input_dir).glob("*.csv")), desc="Label"):
        out = labels_dir / f"{fp.stem}_labels.csv"
        if out.exists(): continue

        df = load_kline_csv(fp)
        tcol = df.columns[0]
        df[tcol] = pd.to_datetime(df[tcol], unit="ms")
        df = df.set_index(tcol).sort_index()

        series = df['close'].resample(freq).last().ffill().to_numpy(np.float32)
        idxs, rets = [], []
        for i in range(0, len(series)-w_bars+1, s_bars):
            e = i + w_bars - 1
            if e + h_bars >= len(series): break
            p0, pf = series[e], series[e+h_bars]
            rets.append((pf-p0)/p0 if p0 else 0.0)
            idxs.append(i)

        rows = []
        for r, i in zip(rets, idxs):
            rows.append({'filename':f"{fp.stem}_win{i:05d}.npz", 'label':r})
        pd.DataFrame(rows).to_csv(out, index=False)

# ---------------------- Dataset ----------------------

class KlineDataset(Dataset):
    def __init__(self, df, data_dir):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        arr = np.load(self.data_dir/row.filename)['data']
        # ensure channels-first ordering: if arr is (H, W, C), move C to axis 0
        if arr.ndim == 3 and arr.shape[0] not in (6,) and arr.shape[-1] == 6:
            arr = np.transpose(arr, (2, 0, 1))
        x = torch.from_numpy(arr).float()        # (6,H,W)
        y = torch.tensor(row.label, dtype=torch.float32)
        return x, y

# ---------------------- Model ----------------------

class GAFNet(nn.Module):
    def __init__(self, in_ch=6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64,128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        self.head = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv(x)               # (B,128,1,1)
        x = x.view(x.size(0), -1)      # (B,128)
        return self.head(x)            # (B,1)

# ---------------------- Training ----------------------

def train(labels_dir, data_dir, args):
    dfs = [pd.read_csv(f) for f in sorted(Path(labels_dir).glob("*_labels.csv"))]
    df  = pd.concat(dfs, ignore_index=True).sort_values('filename')
    n   = len(df)
    ti,vi = int(n*0.8), int(n*0.9)
    train_df, val_df = df[:ti], df[ti:vi]

    train_loader = DataLoader(KlineDataset(train_df,data_dir),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, persistent_workers=True)
    val_loader   = DataLoader(KlineDataset(val_df,  data_dir),
                              batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, persistent_workers=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = GAFNet(in_ch=6).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = OneCycleLR(optimizer,
                           max_lr=args.lr,
                           epochs=args.epochs,
                           steps_per_epoch=len(train_loader),
                           pct_start=0.1)
    criterion = nn.MSELoss()

    best_val = float('inf')
    for epoch in range(1, args.epochs+1):
        model.train()
        running_loss = total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} train")
        for x,y in pbar:
            x,y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x).squeeze(1)
            loss = criterion(out,y)
            loss.backward(); optimizer.step(); scheduler.step()

            bs = y.size(0)
            running_loss += loss.item()*bs; total += bs
            pbar.set_postfix({'mse':f"{running_loss/total:.6f}"})

        model.eval()
        val_loss=vt=0
        with torch.no_grad():
            for x,y in tqdm(val_loader, desc=" Valid"):
                x,y = x.to(device), y.to(device)
                preds = model(x).squeeze(1)
                l = criterion(preds,y)
                val_loss += l.item()*y.size(0); vt += y.size(0)
        val_mse = val_loss/vt
        print(f"Epoch {epoch}/{args.epochs} → Val MSE {val_mse:.6f}")
        if val_mse < best_val:
            best_val = val_mse
            torch.save(model.state_dict(), args.model_out)
            print("  Saved best model.")

    print("Training complete. Best Val MSE:", best_val)

# ---------------------- Entry Point ----------------------

if __name__=='__main__':
    args = parse_args()
    preprocess(args.input_dir, args.data_dir,
               args.freq, args.window, args.stride, None)
    label(    args.input_dir, args.data_dir, args.labels_dir,
               args.freq, args.window, args.stride,
               args.horizon, args.q_low, args.q_high)
    train(    args.labels_dir, args.data_dir, args)
