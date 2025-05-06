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
from torchvision import models

def parse_args():
    p = argparse.ArgumentParser(
        description="Preprocess, label & train 1-channel GAF CNN on 1m K-line CSVs"
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
    p.add_argument('--image_size', type=int, default=None, help="Resize GAF to this size")
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--epochs',     type=int, default=20)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--wd',         type=float, default=1e-5)
    p.add_argument('--workers',    type=int, default=os.cpu_count())
    p.add_argument('--model_out',  default='best_1c_klines.pth')
    return p.parse_args()

def normalize(x: np.ndarray):
    mn, mx = x.min(), x.max()
    if mx == mn:
        return np.zeros_like(x)
    return ((x - mn) / (mx - mn)) * 2 - 1

def load_kline_csv(fp):
    with open(fp, 'r') as f:
        hdr = f.readline().strip().split(',')
    return pd.read_csv(fp, names=hdr, header=0)

def preprocess(input_dir, data_dir, freq, window, stride, image_size):
    """
    Build 1-channel GASF per sliding window.
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

        if 'close' not in df.columns:
            raise KeyError(f"No 'close' column in {fp.name}")
        series = df['close'].resample(freq).last().ffill().to_numpy(np.float32)
        n = len(series)

        for i in range(0, n - w_bars + 1, s_bars):
            out = data_dir / f"{fp.stem}_win{i:05d}.npz"
            if out.exists():
                continue

            win = series[i : i + w_bars]
            a = normalize(win)
            g = gasf.fit_transform(a.reshape(1,-1))[0]

            if image_size and g.shape[0] != image_size:
                from PIL import Image
                im = Image.fromarray(((g+1)/2*255).astype(np.uint8))
                im = im.resize((image_size,image_size), Image.BILINEAR)
                g = (np.array(im, np.float32)/127.5) - 1.0

            arr = g.astype(np.float32)  # shape (H,W)
            np.savez_compressed(str(out), data=arr)

def label(input_dir, data_dir, labels_dir, freq, window, stride, horizon, ql, qh):
    """
    Quantile-based 3-class labeling: -1 / 0 / +1
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

        df = load_kline_csv(fp)
        tcol = df.columns[0]
        df[tcol] = pd.to_datetime(df[tcol], unit="ms")
        df = df.set_index(tcol).sort_index()

        series = df['close'].resample(freq).last().ffill().to_numpy(np.float32)
        idxs, rets = [], []
        for i in range(0, len(series)-w_bars+1, s_bars):
            e = i + w_bars - 1
            if e + h_bars >= len(series):
                break
            p0, pf = series[e], series[e + h_bars]
            rets.append((pf - p0)/p0 if p0 else 0.0)
            idxs.append(i)

        if not rets:
            continue

        rets = np.array(rets)
        low, high = np.quantile(rets, ql), np.quantile(rets, qh)
        rows = []
        for r, i in zip(rets, idxs):
            lab = -1 if r <= low else 1 if r >= high else 0
            rows.append({
                'filename': f"{fp.stem}_win{i:05d}.npz",
                'label': lab
            })
        pd.DataFrame(rows).to_csv(out, index=False)

class KlineDataset(Dataset):
    def __init__(self, df, data_dir):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        arr = np.load(self.data_dir/row.filename)['data']
        x = torch.from_numpy(arr)[None].float()  # add channel dim → (1,H,W)
        y = int(row.label) + 1
        return x, y

def train(labels_dir, data_dir, args):
    dfs = [pd.read_csv(f) for f in sorted(Path(labels_dir).glob("*_labels.csv"))]
    df  = pd.concat(dfs, ignore_index=True).sort_values('filename')
    n   = len(df)
    ti, vi = int(n*0.8), int(n*0.9)
    train_df, val_df = df[:ti], df[ti:vi]

    train_loader = DataLoader(KlineDataset(train_df,data_dir),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, persistent_workers=True)
    val_loader   = DataLoader(KlineDataset(val_df,data_dir),
                              batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, persistent_workers=True)

    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    model  = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    # adapt to 1-channel input
    orig = model.features[0][0]
    model.features[0][0] = nn.Conv2d(1, orig.out_channels,
                                      kernel_size=orig.kernel_size,
                                      stride=orig.stride,
                                      padding=orig.padding,
                                      bias=False)
    in_f = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_f, 3)
    model = model.to(device).to(memory_format=torch.channels_last)
    for p in model.features.parameters():
        p.requires_grad = False

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = OneCycleLR(optimizer,
                           max_lr=args.lr,
                           epochs=args.epochs,
                           steps_per_epoch=len(train_loader),
                           pct_start=0.1)
    criterion = nn.CrossEntropyLoss()

    best_val = 0.0
    for epoch in range(1, args.epochs+1):
        model.train()
        rl=rc=rt=0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch} train")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            scheduler.step()

            preds = out.argmax(1)
            bs = y.size(0)
            rl += loss.item()*bs
            rc += (preds==y).sum().item()
            rt += bs
            pbar.set_postfix({
                'loss': f"{rl/rt:.4f}",
                'acc':  f"{rc/rt:.3f}"
            })

        model.eval()
        vc=vt=0
        for x, y in tqdm(val_loader, desc=" Valid"):
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            vc += (preds==y).sum().item()
            vt += y.size(0)
        vacc = vc/vt
        print(f"Epoch {epoch}/{args.epochs} → Val Acc {vacc:.4f}")
        if vacc > best_val:
            best_val = vacc
            torch.save(model.state_dict(), args.model_out)
            print("  Saved best model.")

    print("Training complete. Best Val Acc:", best_val)

if __name__ == "__main__":
    args = parse_args()
    preprocess(args.input_dir, args.data_dir,
               args.freq, args.window, args.stride, args.image_size)
    label(    args.input_dir, args.data_dir, args.labels_dir,
               args.freq, args.window, args.stride,
               args.horizon, args.q_low, args.q_high)
    train(    args.labels_dir, args.data_dir, args)