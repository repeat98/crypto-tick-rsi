#!/usr/bin/env python3
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pyts.image import GramianAngularField
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
from torchvision import models

def parse_args():
    p = argparse.ArgumentParser(
        description="Preprocess, label & train 1-channel GAF regression on tick CSVs"
    )
    p.add_argument('input_dir',   help='Directory of tick CSVs')
    p.add_argument('data_dir',    help='Where to save/load .npz tensors')
    p.add_argument('labels_dir',  help='Where to save label CSVs')
    p.add_argument('--freq',       default='1S',  help="Resample frequency (e.g. '1S', '100L')")
    p.add_argument('--window',     default='5T',  help="Window length (e.g. '5T')")
    p.add_argument('--stride',     default='1S',  help="Stride between windows (e.g. '1S')")
    p.add_argument('--horizon',    default='5T',  help="Future horizon for labeling (e.g. '5T')")
    p.add_argument('--q_low',   type=float, default=0.2, help="Lower quantile (unused)")
    p.add_argument('--q_high',  type=float, default=0.8, help="Upper quantile (unused)")
    p.add_argument('--image_size', type=int, default=None, help="Resize GAF to this size (None = no resize)")
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--epochs',     type=int, default=20)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--wd',         type=float, default=1e-5)
    p.add_argument('--workers',    type=int, default=os.cpu_count())
    p.add_argument('--model_out',   default='best_1c_regression.pth')
    return p.parse_args()

def normalize(x: np.ndarray):
    mn, mx = x.min(), x.max()
    if mx == mn:
        return np.zeros_like(x)
    return ((x - mn) / (mx - mn)) * 2 - 1

def load_csv(fp):
    """
    Load either K-line CSVs (with 'close') or trade CSVs (with 'price' & 'time').
    """
    df = pd.read_csv(fp)
    if 'price' in df.columns and 'time' in df.columns:
        df = df.rename(columns={'price': 'close', 'time': 'timestamp'})
    return df

def preprocess(input_dir, data_dir, freq, window, stride, image_size):
    """
    Build 1-channel GASF per sliding window over 'close' series.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    gasf = GramianAngularField(method='summation')
    w_bars = int(pd.Timedelta(window) / pd.Timedelta(freq))
    s_bars = max(int(pd.Timedelta(stride) / pd.Timedelta(freq)), 1)

    for fp in tqdm(sorted(Path(input_dir).glob("*.csv")), desc="Preprocess"):
        df = load_csv(fp)

        # timestamp handling
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
        else:
            tcol = df.columns[0]
            df[tcol] = pd.to_datetime(df[tcol], unit='ms')
            df = df.set_index(tcol)

        df = df.sort_index()
        if 'close' not in df.columns:
            raise KeyError(f"No 'close' (or 'price') column in {fp.name}")

        series = df['close'].resample(freq).last().ffill().to_numpy(np.float32)
        n = len(series)

        for i in range(0, n - w_bars + 1, s_bars):
            out = data_dir / f"{fp.stem}_win{i:05d}.npz"
            if out.exists():
                continue

            win = series[i : i + w_bars]
            a = normalize(win)
            g = gasf.fit_transform(a.reshape(1, -1))[0]

            if image_size and g.shape[0] != image_size:
                im = Image.fromarray(((g + 1) * 127.5).astype(np.uint8))
                im = im.resize((image_size, image_size), Image.BILINEAR)
                g = (np.array(im, np.float32) / 127.5) - 1.0

            np.savez_compressed(str(out), data=g.astype(np.float32))

def label(input_dir, data_dir, labels_dir, freq, window, stride, horizon, q_low, q_high):
    """
    Produce continuous-return labels for regression.
    """
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    w_bars = int(pd.Timedelta(window) / pd.Timedelta(freq))
    s_bars = max(int(pd.Timedelta(stride) / pd.Timedelta(freq)), 1)
    h_bars = int(pd.Timedelta(horizon) / pd.Timedelta(freq))

    for fp in tqdm(sorted(Path(input_dir).glob("*.csv")), desc="Label"):
        out = labels_dir / f"{fp.stem}_labels.csv"
        if out.exists():
            continue

        df = load_csv(fp)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
        else:
            tcol = df.columns[0]
            df[tcol] = pd.to_datetime(df[tcol], unit='ms')
            df = df.set_index(tcol)

        df = df.sort_index()
        series = df['close'].resample(freq).last().ffill().to_numpy(np.float32)

        idxs, rets = [], []
        for i in range(0, len(series) - w_bars + 1, s_bars):
            end = i + w_bars - 1
            if end + h_bars >= len(series):
                break
            p0, pf = series[end], series[end + h_bars]
            rets.append((pf - p0) / p0 if p0 else 0.0)
            idxs.append(i)

        rows = [
            {'filename': f"{fp.stem}_win{i:05d}.npz", 'label': r}
            for r, i in zip(rets, idxs)
        ]
        pd.DataFrame(rows).to_csv(out, index=False)

class KlineDataset(Dataset):
    def __init__(self, df, data_dir):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        arr = np.load(self.data_dir / row.filename)['data']
        x = torch.from_numpy(arr)[None].float()
        y = torch.tensor(row.label, dtype=torch.float32)
        return x, y

def train(labels_dir, data_dir, args):
    dfs = [pd.read_csv(f) for f in sorted(Path(labels_dir).glob("*_labels.csv"))]
    df  = pd.concat(dfs, ignore_index=True).sort_values('filename')
    n   = len(df)
    ti, vi = int(n * 0.8), int(n * 0.9)
    train_df, val_df = df[:ti], df[ti:vi]

    train_loader = DataLoader(
        KlineDataset(train_df, data_dir),
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, persistent_workers=True, pin_memory=True
    )
    val_loader = DataLoader(
        KlineDataset(val_df, data_dir),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, persistent_workers=True, pin_memory=True
    )

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    model = models.mobilenet_v3_small(
        weights=models.MobileNet_V3_Small_Weights.DEFAULT
    )
    orig = model.features[0][0]
    model.features[0][0] = nn.Conv2d(
        1, orig.out_channels,
        kernel_size=orig.kernel_size,
        stride=orig.stride,
        padding=orig.padding,
        bias=False
    )
    in_f = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_f, 1)

    for p in model.features.parameters():
        p.requires_grad = False

    model = model.to(device).to(memory_format=torch.channels_last)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=args.lr,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.3
    )
    criterion = nn.MSELoss()

    best_val = float('inf')
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = total_samples = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} train")
        for x, y in pbar:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            out = model(x).squeeze(1)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            bs = y.size(0)
            total_loss += loss.item() * bs
            total_samples += bs
            pbar.set_postfix(mse=total_loss / total_samples)

        model.eval()
        val_loss = val_samples = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=" Valid"):
                x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
                y = y.to(device, non_blocking=True)
                preds = model(x).squeeze(1)
                l = criterion(preds, y)
                val_loss += l.item() * y.size(0)
                val_samples += y.size(0)
        val_mse = val_loss / val_samples
        print(f"Epoch {epoch}/{args.epochs} → Val MSE {val_mse:.6f}")
        if val_mse < best_val:
            best_val = val_mse
            torch.save(model.state_dict(), args.model_out)
            print("  Saved best model.")

    print("Training complete. Best Val MSE:", best_val)

if __name__ == "__main__":
    args = parse_args()
    preprocess(
        args.input_dir, args.data_dir,
        args.freq, args.window, args.stride, args.image_size
    )
    label(
        args.input_dir, args.data_dir, args.labels_dir,
        args.freq, args.window, args.stride,
        args.horizon, args.q_low, args.q_high
    )
    train(args.labels_dir, args.data_dir, args)