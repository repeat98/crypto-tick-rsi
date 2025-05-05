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
from concurrent.futures import ProcessPoolExecutor, as_completed

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert time series parquet files into Gramian Angular Field images with fixed-time windows"
    )
    parser.add_argument('input_dir', type=str, help='Directory with .parquet files')
    parser.add_argument('output_dir', type=str, help='Where to save GAF images')
    parser.add_argument('--freq', type=str, default='1S',
                        help="Resample frequency (pandas offset alias, e.g. '1S','10S','1T')")
    parser.add_argument('--window', type=str, default='5T',
                        help="Window length (pandas offset alias, e.g. '5S','1T','5T')")
    parser.add_argument('--stride', type=str, default='1T',
                        help="Stride between windows (offset alias) e.g. '30S','15T'")
    parser.add_argument('--image_size', type=int, default=None,
                        help='Resize GAF images to this size (pixels)')
    parser.add_argument('--method', type=str, choices=['summation', 'difference'],
                        default='summation', help='GASF or GADF')
    parser.add_argument('--workers', type=int, default=os.cpu_count(),
                        help='Number of processes to spawn')
    return parser.parse_args()

def normalize_series(series: np.ndarray) -> np.ndarray:
    min_val, max_val = series.min(), series.max()
    if max_val == min_val:
        return np.zeros_like(series)
    scaled = (series - min_val) / (max_val - min_val)
    return scaled * 2 - 1

def process_file(args_tuple):
    fp, output_dir, freq, window, stride, image_size, method, position = args_tuple
    gazer = GramianAngularField(method=method)

    # Read parquet and set datetime index
    table = pq.read_table(str(fp))
    df = table.to_pandas()
    if 'price' not in df.columns or 'trade_time' not in df.columns:
        raise ValueError(f"Missing columns in {fp.name}")
    df['trade_time'] = pd.to_datetime(df['trade_time'])
    df = df.set_index('trade_time').sort_index()

    # Resample into fixed-time series
    series = df['price'].resample(freq).last().ffill().to_numpy(dtype=np.float32)
    n = len(series)

    # Compute bars per window and stride in number of samples
    window_bars = int(pd.Timedelta(window) / pd.Timedelta(freq))
    stride_bars = int(pd.Timedelta(stride) / pd.Timedelta(freq))
    if window_bars < 1 or stride_bars < 1:
        raise ValueError("Window or stride is smaller than resample frequency")

    # Slide and generate
    saved = 0
    total_windows = max((n - window_bars) // stride_bars + 1, 0)
    for i in tqdm(
        range(0, n - window_bars + 1, stride_bars),
        desc=f"{fp.name}", total=total_windows,
        position=position, leave=False
    ):
        window_data = series[i:i + window_bars]
        scaled = normalize_series(window_data)
        gaf = gazer.fit_transform(scaled.reshape(1, -1))[0]
        img_arr = ((gaf + 1) / 2 * 255).astype(np.uint8)
        img = Image.fromarray(img_arr)
        if image_size:
            img = img.resize((image_size, image_size))
        out_name = f"{fp.stem}_win{i:05d}.png"
        out_path = output_dir / out_name
        if not out_path.exists():
            img.save(out_path)
            saved += 1
    return saved

def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(input_dir.glob('*.parquet'))
    if not parquet_files:
        print(f"No parquet files found in {input_dir}")
        return

    # Prepare tasks
    task_args = []
    for pos, fp in enumerate(parquet_files):
        task_args.append((
            fp, output_dir,
            args.freq, args.window, args.stride,
            args.image_size, args.method,
            pos
        ))

    total_saved = 0
    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        futures = {exe.submit(process_file, ta): ta[0] for ta in task_args}
        for future in tqdm(as_completed(futures), total=len(futures),
                           desc='Files', position=len(task_args)):
            fp = futures[future]
            try:
                total_saved += future.result()
            except Exception as e:
                print(f"[ERROR] {fp.name}: {e}")

    print(f"Done! Generated {total_saved} GAF images.")

if __name__ == '__main__':
    main()
