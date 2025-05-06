#!/usr/bin/env python3
import warnings
import os
warnings.filterwarnings("ignore", category=UserWarning)   # pandas/dateutil warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


class BacktestDataset(Dataset):
    def __init__(self, npz_files, csv_dir, freq, window, stride, horizon):
        self.files   = sorted(npz_files)
        self.csv_dir = Path(csv_dir)
        self.freq    = freq
        self.window  = window
        self.stride  = stride
        self.horizon = horizon

        self.w = int(pd.Timedelta(window)  / pd.Timedelta(freq))
        self.s = max(int(pd.Timedelta(stride) / pd.Timedelta(freq)), 1)
        self.h = int(pd.Timedelta(horizon) / pd.Timedelta(freq))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        npz_path = self.files[idx]
        data = np.load(npz_path)['data']
        x = torch.from_numpy(data).float().unsqueeze(0)

        stem, win = npz_path.stem.rsplit('_win', 1)
        i = int(win)
        csv_path = self.csv_dir / f"{stem}.csv"
        df = pd.read_csv(csv_path, header=0)
        tcol = df.columns[0]
        df[tcol] = pd.to_datetime(df[tcol].astype(int), unit='ms')
        df = df.set_index(tcol).sort_index()
        series = df['close'].resample(self.freq).last().ffill().to_numpy(np.float32)

        end = i + self.w - 1
        if end + self.h < len(series):
            p0, pf = series[end], series[end + self.h]
            y = (pf - p0) / p0
        else:
            y = 0.0

        return x, torch.tensor(y, dtype=torch.float32), npz_path.name


def load_model(model_path, device):
    from torchvision import models
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    # adapt to 1-channel + regression
    orig = model.features[0][0]
    model.features[0][0] = torch.nn.Conv2d(1, orig.out_channels,
                                            kernel_size=orig.kernel_size,
                                            stride=orig.stride,
                                            padding=orig.padding,
                                            bias=False)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, 1)
    sd = torch.load(model_path, map_location='cpu')
    model.load_state_dict(sd)
    return model.to(device).eval()


def backtest(args):
    npz_files = list(Path(args.data_dir).glob("*.npz"))
    ds = BacktestDataset(npz_files, args.csv_dir,
                         args.freq, args.window,
                         args.stride, args.horizon)
    loader = DataLoader(ds, batch_size=args.batch_size,
                        num_workers=args.workers)

    # prefer CUDA, then MPS, then CPU
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    model  = load_model(args.model, device)

    preds, names = [], []
    with torch.no_grad():
        for x, _, fn in tqdm(loader, desc="Inferring"):
            out = model(x.to(device)).squeeze(1).cpu().numpy()
            preds.extend(out.tolist())
            names.extend(fn)

    cash, pos = args.capital, 0.0
    w, h, s = ds.w, ds.h, ds.s

    for pred, name in tqdm(zip(preds, names), desc="Simulating", total=len(preds)):
        stem, win = Path(name).stem.rsplit('_win', 1)
        i = int(win)
        csv_path = Path(args.csv_dir) / f"{stem}.csv"
        df = pd.read_csv(csv_path, header=0)
        tcol = df.columns[0]
        df[tcol] = pd.to_datetime(df[tcol].astype(int), unit='ms')
        series = df.set_index(tcol)['close'].resample(args.freq).last().ffill().to_numpy(np.float32)
        price = series[i + w - 1]

        if pred > args.thresh and cash > 0:
            qty = cash / price * (1 - args.spread)
            cash, pos = 0.0, qty
        elif pred <= args.thresh and pos > 0:
            exit_p = price * (1 - args.spread)
            cash = pos * exit_p
            pos = 0.0

    # close any remaining
    if pos > 0:
        exit_p = price * (1 - args.spread)
        cash = pos * exit_p

    print(f"\nFinal capital: {cash:.2f} (started with {args.capital:.2f})")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   required=True)
    p.add_argument("--csv_dir",    required=True)
    p.add_argument("--model",      required=True)
    p.add_argument("--freq",       default="1T")
    p.add_argument("--window",     default="5T")
    p.add_argument("--stride",     default="1T")
    p.add_argument("--horizon",    default="5T")
    p.add_argument("--thresh",   type=float, default=0.0015)
    p.add_argument("--spread",   type=float, default=0.0003)
    p.add_argument("--capital",  type=float, default=10000)
    p.add_argument("--batch_size",type=int,   default=64)
    p.add_argument("--workers",   type=int,   default=os.cpu_count())
    args = p.parse_args()

    backtest(args)