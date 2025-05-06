#!/usr/bin/env python3
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyts.image import GramianAngularField
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models


def parse_args():
    p = argparse.ArgumentParser(
        description="Step 1: 6-channel GASF+GADF with batch metrics"
    )
    p.add_argument('input_dir',  help='Parquet directory')
    p.add_argument('data_dir',   help='Where to save/load .npz tensors')
    p.add_argument('labels_dir', help='Where to save label CSVs')
    p.add_argument('--freq',      default='1S',  help="Resample freq")
    p.add_argument('--window',    default='5T',  help="Main window length")
    p.add_argument('--stride',    default='1T',  help="Stride between windows")
    p.add_argument('--horizon',   default='5T',  help="Return horizon")
    p.add_argument('--q_low',   type=float, default=0.2, help="Lower quantile")
    p.add_argument('--q_high',  type=float, default=0.8, help="Upper quantile")
    p.add_argument('--image_size', type=int, default=224, help="Resize size")
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--epochs',     type=int, default=20)
    p.add_argument('--lr',         type=float, default=1e-4)
    p.add_argument('--wd',         type=float, default=1e-5)
    p.add_argument('--workers',    type=int, default=os.cpu_count())
    p.add_argument('--model_out',  default='best_multichannel.pth')
    return p.parse_args()


def normalize(x: np.ndarray):
    mn, mx = x.min(), x.max()
    return np.zeros_like(x) if mx == mn else ((x - mn) / (mx - mn)) * 2 - 1


def preprocess(input_dir, data_dir, freq, window, stride, image_size):
    from PIL import Image

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    gasf = GramianAngularField(method='summation')
    gadf = GramianAngularField(method='difference')

    w_bars = int(pd.Timedelta(window) / pd.Timedelta(freq))
    s_bars = max(int(pd.Timedelta(stride) / pd.Timedelta(freq)), 1)
    short  = w_bars // 2
    long   = w_bars * 2

    files = sorted(Path(input_dir).glob('*.parquet'))
    for fp in tqdm(files, desc='Preprocessing'):
        df = pq.read_table(str(fp)).to_pandas()
        df['trade_time'] = pd.to_datetime(df['trade_time'])
        df = df.set_index('trade_time').sort_index()
        series = df['price'].resample(freq).last().ffill().to_numpy(np.float32)
        n = len(series)

        for i in range(0, n - w_bars + 1, s_bars):
            out_path = data_dir / f"{fp.stem}_win{i:05d}.npz"
            if out_path.exists():
                continue

            end = i + w_bars
            windows = [
                series[max(0, end - short):end],
                series[i:end],
                series[max(0, end - long):end],
            ]

            channels = []
            for win in windows:
                arr = normalize(win)
                g1 = gasf.fit_transform(arr.reshape(1, -1))[0]
                g2 = gadf.fit_transform(arr.reshape(1, -1))[0]
                for g in (g1, g2):
                    if image_size and g.shape[0] != image_size:
                        img = Image.fromarray(((g + 1) / 2 * 255).astype(np.uint8))
                        img = img.resize((image_size, image_size), Image.BILINEAR)
                        g = (np.array(img, np.float32) / 127.5) - 1.0
                    channels.append(g.astype(np.float32))

            tensor = np.stack(channels, axis=0)  # (6, H, W)
            np.savez_compressed(str(out_path), data=tensor)


def label(input_dir, data_dir, labels_dir, freq, window, stride, horizon, ql, qh):
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    w_bars = int(pd.Timedelta(window) / pd.Timedelta(freq))
    s_bars = max(int(pd.Timedelta(stride) / pd.Timedelta(freq)), 1)
    h_bars = int(pd.Timedelta(horizon) / pd.Timedelta(freq))

    files = sorted(Path(input_dir).glob('*.parquet'))
    for fp in tqdm(files, desc='Labeling'):
        out_csv = labels_dir / f"{fp.stem}_labels.csv"
        if out_csv.exists():
            continue

        df = pq.read_table(str(fp)).to_pandas()
        df['trade_time'] = pd.to_datetime(df['trade_time'])
        df = df.set_index('trade_time').sort_index()
        series = df['price'].resample(freq).last().ffill().to_numpy(np.float32)

        idxs, rets = [], []
        for i in range(0, len(series) - w_bars + 1, s_bars):
            end = i + w_bars - 1
            if end + h_bars >= len(series):
                break
            p0, pf = series[end], series[end + h_bars]
            rets.append((pf - p0) / p0 if p0 else 0.0)
            idxs.append(i)

        if not rets:
            continue

        rets = np.array(rets)
        low, high = np.quantile(rets, ql), np.quantile(rets, qh)
        rows = []
        for r, i in zip(rets, idxs):
            lab = -1 if r <= low else 1 if r >= high else 0
            rows.append({'filename': f"{fp.stem}_win{i:05d}.npz", 'label': lab})

        pd.DataFrame(rows).to_csv(out_csv, index=False)


class Fractal3Dataset(Dataset):
    def __init__(self, df, data_dir):
        self.df = df.reset_index(drop=True)
        self.data_dir = Path(data_dir)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        arr = np.load(self.data_dir / row.filename)['data']
        x = torch.from_numpy(arr).float()
        y = int(row.label) + 1
        return x, y


def train(labels_dir, data_dir, args):
    dfs = [pd.read_csv(f) for f in sorted(Path(labels_dir).glob('*_labels.csv'))]
    df = pd.concat(dfs, ignore_index=True).sort_values('filename')
    n = len(df)
    ti, vi = int(n * 0.8), int(n * 0.9)
    df_tr, df_v = df[:ti], df[ti:vi]

    dl_tr = DataLoader(Fractal3Dataset(df_tr, data_dir),
                       batch_size=args.batch_size, shuffle=True,
                       num_workers=args.workers, persistent_workers=True)
    dl_v = DataLoader(Fractal3Dataset(df_v, data_dir),
                      batch_size=args.batch_size, shuffle=False,
                      num_workers=args.workers, persistent_workers=True)

    device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    orig = model.features[0][0]
    model.features[0][0] = nn.Conv2d(6, orig.out_channels,
                                      kernel_size=orig.kernel_size,
                                      stride=orig.stride,
                                      padding=orig.padding,
                                      bias=False)
    in_f = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_f, 3)
    model = model.to(device).to(memory_format=torch.channels_last)
    for p in model.features.parameters():
        p.requires_grad = False

    opt   = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = OneCycleLR(opt, max_lr=args.lr, epochs=args.epochs,
                       steps_per_epoch=len(dl_tr), pct_start=0.3)
    crit  = nn.CrossEntropyLoss()

    best = 0.0
    for e in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(dl_tr, desc=f"Epoch {e} train")
        running_loss = running_correct = running_total = 0
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = crit(out, y)
            loss.backward()
            opt.step()
            sched.step()

            preds = out.argmax(1)
            batch_acc = (preds == y).float().mean().item()
            running_loss    += loss.item() * y.size(0)
            running_correct += (preds == y).sum().item()
            running_total   += y.size(0)
            pbar.set_postfix({
                'batch_loss': f"{loss.item():.4f}",
                'batch_acc':  f"{batch_acc:.3f}"
            })

        epoch_loss = running_loss / running_total
        epoch_acc  = running_correct / running_total

        model.eval()
        val_correct = val_total = 0
        for x, y in tqdm(dl_v, desc=" Validating"):
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            val_correct += (preds == y).sum().item()
            val_total   += y.size(0)
        val_acc = val_correct / val_total

        print(f"Epoch {e}/{args.epochs} → "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best:
            best = val_acc
            torch.save(model.state_dict(), args.model_out)
            print("  Saved best model.")

    print("Training complete. Best Val Acc:", best)


# -----------------------------------------------------------------------------
if __name__ == '__main__':
    args = parse_args()
    preprocess(args.input_dir, args.data_dir, args.freq, args.window,
               args.stride, args.image_size)
    label(args.input_dir, args.data_dir, args.labels_dir,
          args.freq, args.window, args.stride, args.horizon,
          args.q_low, args.q_high)
    train(args.labels_dir, args.data_dir, args)