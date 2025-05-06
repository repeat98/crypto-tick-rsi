#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyts.image import GramianAngularField
from tqdm.auto import tqdm
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert time series parquet files into 3-channel GAF images (short, main, long windows)"
    )
    parser.add_argument('input_dir', type=str, help='Directory with .parquet files')
    parser.add_argument('output_dir', type=str, help='Where to save GAF images')
    parser.add_argument('--freq', type=str, default='1S',
                        help="Resample frequency (pandas offset alias, e.g. '1S','1T')")
    parser.add_argument('--window', type=str, default='5T',
                        help="Main window length (e.g. '5T' for 5 minutes)")
    parser.add_argument('--stride', type=str, default='1T',
                        help="Stride between windows (e.g. '1T' for 1 minute)")
    parser.add_argument('--image_size', type=int, default=224,
                        help='Resize final GAF images to this size (pixels)')
    parser.add_argument('--method', type=str, choices=['summation','difference'],
                        default='summation', help='GAF method: summation or difference')
    parser.add_argument('--workers', type=int, default=os.cpu_count(),
                    help='Number of parallel workers')
    return parser.parse_args()


def normalize_series(series: np.ndarray) -> np.ndarray:
    """
    Normalize series to [-1,1] for GAF.
    """
    mn, mx = series.min(), series.max()
    if mx == mn:
        return np.zeros_like(series)
    scaled = (series - mn) / (mx - mn)
    return scaled * 2 - 1


def count_windows(series_length: int, freq: str, window: str, stride: str) -> int:
    freq_td   = pd.Timedelta(freq)
    main_td   = pd.Timedelta(window)
    stride_td = pd.Timedelta(stride)
    main_bars   = int(main_td / freq_td)
    stride_bars = max(int(stride_td / freq_td), 1)
    return max((series_length - main_bars) // stride_bars + 1, 0)


def main():
    args = parse_args()
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(input_dir.glob('*.parquet'))
    if not parquet_files:
        print(f"No parquet files found in {input_dir}")
        return

    # First pass: compute total number of windows across all files
    total_windows = 0
    file_windows = {}
    for fp in parquet_files:
        table = pq.read_table(str(fp))
        df = table.to_pandas()
        if 'price' not in df.columns or 'trade_time' not in df.columns:
            raise ValueError(f"Missing 'price' or 'trade_time' in {fp.name}")
        df['trade_time'] = pd.to_datetime(df['trade_time'])
        df = df.set_index('trade_time').sort_index()
        series = df['price'].resample(args.freq).last().ffill().to_numpy(dtype=np.float32)
        n = len(series)
        w = count_windows(n, args.freq, args.window, args.stride)
        if w > 0:
            file_windows[fp] = (series, n, w)
            total_windows += w

    if total_windows == 0:
        print("No windows to process.")
        return

    gazer = GramianAngularField(method=args.method)
    pbar = tqdm(total=total_windows, desc="Remaining GAFs")

    # Process each file in turn
    for fp, (series, n, total_w) in file_windows.items():
        # compute bar counts
        freq_td   = pd.Timedelta(args.freq)
        main_td   = pd.Timedelta(args.window)
        short_td  = main_td / 2
        long_td   = main_td * 2
        stride_td = pd.Timedelta(args.stride)

        main_bars   = int(main_td  / freq_td)
        short_bars  = max(int(short_td / freq_td), 1)
        long_bars   = max(int(long_td  / freq_td), short_bars + main_bars)
        stride_bars = max(int(stride_td/ freq_td), 1)

        # slide windows
        for i in range(0, n - main_bars + 1, stride_bars):
            out_name = f"{fp.stem}_win{i:05d}.png"
            out_path = output_dir / out_name
            if out_path.exists():
                # skip already-processed window
                pbar.update(1)
                continue

            end_idx = i + main_bars
            main_data  = series[i:end_idx]
            short_data = series[max(0, end_idx-short_bars):end_idx]
            long_start = max(0, end_idx-long_bars)
            long_data  = series[long_start:end_idx]

            m = normalize_series(main_data)
            s = normalize_series(short_data)
            l = normalize_series(long_data)

            gaf_m = gazer.fit_transform(m.reshape(1, -1))[0]
            gaf_s = gazer.fit_transform(s.reshape(1, -1))[0]
            gaf_l = gazer.fit_transform(l.reshape(1, -1))[0]

            im_m = ((gaf_m + 1)/2 * 255).astype(np.uint8)
            im_s = ((gaf_s + 1)/2 * 255).astype(np.uint8)
            im_l = ((gaf_l + 1)/2 * 255).astype(np.uint8)

            img_m = Image.fromarray(im_m)
            img_s = Image.fromarray(im_s)
            img_l = Image.fromarray(im_l)
            if args.image_size and args.image_size != main_bars:
                img_m = img_m.resize((args.image_size, args.image_size))
                img_s = img_s.resize((args.image_size, args.image_size))
                img_l = img_l.resize((args.image_size, args.image_size))

            stacked = np.stack([np.array(img_s), np.array(img_m), np.array(img_l)], axis=2)
            Image.fromarray(stacked).save(out_path)

            pbar.update(1)

    pbar.close()
    print("Done!")
    

if __name__ == '__main__':
    main()