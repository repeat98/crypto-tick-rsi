#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import csv
import numpy as np
import pandas as pd
import pyarrow.parquet as pq


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate multi-class labels (-1,0,1) for time-based GAF windows from SOLUSD parquet data"
    )
    parser.add_argument('parquet_dir', type=str,
                        help='Directory containing .parquet trade files')
    parser.add_argument('gaf_dir', type=str,
                        help='Directory where GAF .png files are stored')
    parser.add_argument('labels_dir', type=str,
                        help='Directory to save CSV label files')
    parser.add_argument('--freq', type=str, default='1S',
                        help="Resample frequency (pandas offset alias, e.g. '1S','10S','1T')")
    parser.add_argument('--window', type=str, default='5T',
                        help="Window length (alias, e.g. '5T' for 5 minutes)")
    parser.add_argument('--horizon', type=str, default='5T',
                        help="Prediction horizon (alias, e.g. '5T' for 5 minutes)")
    parser.add_argument('--stride', type=str, default=None,
                        help="Stride between label windows (alias, defaults to window)")
    parser.add_argument('--q_low', type=float, default=0.2,
                        help='Lower quantile for multi-class threshold (e.g., 0.2)')
    parser.add_argument('--q_high', type=float, default=0.8,
                        help='Upper quantile for multi-class threshold (e.g., 0.8)')
    return parser.parse_args()


def main():
    args = parse_args()
    parquet_dir = Path(args.parquet_dir)
    gaf_dir = Path(args.gaf_dir)
    labels_dir = Path(args.labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # compute bars per window/horizon/stride
    freq_td = pd.to_timedelta(args.freq)
    window_td = pd.to_timedelta(args.window)
    horizon_td = pd.to_timedelta(args.horizon)
    stride_td = pd.to_timedelta(args.stride) if args.stride else window_td

    window_bars = int(window_td / freq_td)
    horizon_bars = int(horizon_td / freq_td)
    stride_bars = int(stride_td / freq_td)

    if window_bars < 1 or horizon_bars < 1 or stride_bars < 1:
        raise ValueError("window, horizon, or stride is smaller than freq")

    parquet_files = sorted(parquet_dir.glob('*.parquet'))
    if not parquet_files:
        print(f"No parquet files found in {parquet_dir}")
        return

    for pq_file in parquet_files:
        print(f"Processing {pq_file.name}...")
        # read parquet
        table = pq.read_table(str(pq_file))
        df = table.to_pandas()
        if 'price' not in df.columns or 'trade_time' not in df.columns:
            print(f"Skipping {pq_file.name}, missing columns")
            continue
        df['trade_time'] = pd.to_datetime(df['trade_time'])
        df = df.set_index('trade_time').sort_index()

        # resample price series
        price_series = df['price'].resample(args.freq).last().ffill().to_numpy(dtype=np.float32)
        n = len(price_series)
        max_start = n - window_bars - horizon_bars + 1
        if max_start < 1:
            print(f"Not enough data in {pq_file.name}")
            continue

        # Precompute all log-returns for thresholding
        returns = []
        for start in range(0, max_start, stride_bars):
            end_idx = start + window_bars
            future_idx = end_idx + horizon_bars - 1
            p_now = price_series[end_idx - 1]
            p_future = price_series[future_idx]
            lr = np.log(p_future / p_now) if p_now > 0 else 0.0
            returns.append(lr)
        returns_arr = np.array(returns, dtype=np.float32)
        # Determine quantile thresholds
        low_th = np.quantile(returns_arr, args.q_low)
        high_th = np.quantile(returns_arr, args.q_high)

        # prepare CSV
        csv_path = labels_dir / f"{pq_file.stem}_labels.csv"
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'log_return', 'label'])

            idx = 0
            for start in range(0, max_start, stride_bars):
                lr = returns_arr[idx]
                # multi-class label: -1=down,0=neutral,1=up
                if lr <= low_th:
                    label = -1
                elif lr >= high_th:
                    label = 1
                else:
                    label = 0

                fname = f"{pq_file.stem}_win{start:05d}.png"
                if (gaf_dir / fname).exists():
                    writer.writerow([fname, f"{lr:.6f}", label])
                idx += 1

        print(f"Wrote multi-class labels for {pq_file.name} -> {csv_path.name}")

if __name__ == '__main__':
    main()
