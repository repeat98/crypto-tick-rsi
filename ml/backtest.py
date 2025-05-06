#!/usr/bin/env python3
import os
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm.auto import tqdm


def parse_args():
    p = argparse.ArgumentParser(description="Backtest GAF regression model on tick data with realistic P&L metrics")
    p.add_argument('model_path', help='Path to trained model .pth')
    p.add_argument('labels_dir', help='Directory containing *_labels.csv files')
    p.add_argument('data_dir', help='Directory containing .npz input data')
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--threshold', type=float, default=0.0,
                   help='Decision threshold: predict > threshold ⇒ long')
    p.add_argument('--capital', type=float, default=1e4,
                   help='Initial capital for P&L simulation')
    p.add_argument('--cost_per_trade', type=float, default=0.0001,
                   help='Round-trip cost as fraction of capital per trade')
    return p.parse_args()


class BacktestDataset(Dataset):
    def __init__(self, labels_csv, data_dir):
        self.df = pd.read_csv(labels_csv)
        self.data_dir = Path(data_dir)
        # ensure sorted by filename so window order matches labels
        self.df = self.df.sort_values('filename').reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        arr = np.load(self.data_dir / row.filename)['data']
        x = torch.from_numpy(arr)[None].float()
        y = row.label
        return x, y, row.filename


def load_model(model_path, device):
    # load architecture directly to avoid torch.hub download issues
    model = models.mobilenet_v3_small(weights=None)
    # adapt first conv to single-channel input
    orig = model.features[0][0]
    model.features[0][0] = torch.nn.Conv2d(
        1, orig.out_channels,
        kernel_size=orig.kernel_size,
        stride=orig.stride,
        padding=orig.padding,
        bias=False
    )
    # adapt final classifier to single output
    in_f = model.classifier[3].in_features
    model.classifier[3] = torch.nn.Linear(in_f, 1)

    # load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device).eval()


def run_backtest(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.model_path, device)

    all_results = []

    labels_dir = Path(args.labels_dir)
    # iterate over label files with progress bar
    for csv_file in tqdm(sorted(labels_dir.glob('*_labels.csv')), desc="Backtesting files"):
        dataset = BacktestDataset(csv_file, args.data_dir)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=os.cpu_count(),
            pin_memory=True
        )

        preds, rets = [], []
        # iterate over batches with progress bar
        for X, y, _ in tqdm(loader, desc=f"  Batches [{csv_file.stem}]", leave=False):
            X = X.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            with torch.no_grad():
                out = model(X).squeeze(1).cpu().numpy()
            preds.extend(out.tolist())
            rets.extend(y.tolist())

        preds = np.array(preds)
        rets = np.array(rets)
        positions = (preds > args.threshold).astype(float)
        # compute dollar P&L per trade
        trade_pnl = args.capital * positions * rets
        # add transaction costs when position changes
        trades = np.abs(np.diff(positions, prepend=0))
        cost = args.capital * args.cost_per_trade * trades
        trade_pnl -= cost

        num_trades = int(trades.sum())
        win_trades = int((trade_pnl[positions==1] > 0).sum())
        win_rate = win_trades / num_trades if num_trades > 0 else float('nan')
        cum_return = float(np.sum(trade_pnl))

        # average win / loss
        wins = trade_pnl[trade_pnl>0]
        losses = trade_pnl[trade_pnl<0]
        avg_win = float(wins.mean()) if len(wins)>0 else 0.0
        avg_loss = float(losses.mean()) if len(losses)>0 else 0.0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss!=0 else float('nan')

        # compute daily P&L for Sharpe
        # here assume each window is same time spacing; use total P&L series
        daily_pnl = trade_pnl.reshape(-1, 1).sum(axis=1) if False else trade_pnl
        mu = np.mean(daily_pnl)
        sigma = np.std(daily_pnl) + 1e-9
        sharpe = (mu / sigma) * np.sqrt(252)

        # save per-file details
        per_df = pd.DataFrame({
            'filename': dataset.df.filename,
            'pred': preds,
            'ret': rets,
            'pos': positions,
            'trade_pnl': trade_pnl
        })
        per_out = labels_dir / f"backtest_{csv_file.stem}.csv"
        per_df.to_csv(per_out, index=False)

        all_results.append({
            'file': csv_file.name,
            'num_trades': num_trades,
            'win_trades': win_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'win_loss_ratio': win_loss_ratio,
            'cum_return': cum_return,
            'sharpe': sharpe
        })

    # summary
    summary = pd.DataFrame(all_results)
    print(summary)
    summary.to_csv(labels_dir / 'backtest_summary.csv', index=False)


if __name__ == '__main__':
    args = parse_args()
    run_backtest(args)
