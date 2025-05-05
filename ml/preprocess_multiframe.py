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
                        help='Number of processes to spawn')
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


def process_file(args_tuple):
    fp, output_dir, freq, window, stride, image_size, method, position = args_tuple
    gazer = GramianAngularField(method=method)

    # Read parquet and index by timestamp
    table = pq.read_table(str(fp))
    df = table.to_pandas()
    if 'price' not in df.columns or 'trade_time' not in df.columns:
        raise ValueError(f"Missing 'price' or 'trade_time' in {fp.name}")
    df['trade_time'] = pd.to_datetime(df['trade_time'])
    df = df.set_index('trade_time').sort_index()

    # Resample price series to fixed frequency
    series = df['price'].resample(freq).last().ffill().to_numpy(dtype=np.float32)
    n = len(series)

    # Compute bars per main/short/long windows and stride
    freq_td   = pd.Timedelta(freq)
    main_td   = pd.Timedelta(window)
    short_td  = main_td / 2
    long_td   = main_td * 2
    stride_td = pd.Timedelta(stride)

    main_bars   = int(main_td  / freq_td)
    short_bars  = max(int(short_td / freq_td), 1)
    long_bars   = max(int(long_td  / freq_td), short_bars + main_bars)
    stride_bars = max(int(stride_td/ freq_td), 1)

    if main_bars < 1:
        raise ValueError("window is smaller than frequency")

    total_windows = max((n - main_bars) // stride_bars + 1, 0)
    # Skip entire file if all expected windows are already present
    existing = len(list(output_dir.glob(f"{fp.stem}_win*.png")))
    if existing >= total_windows:
        return 0
    saved = 0

    # Slide windows
    for i in tqdm(range(0, n - main_bars + 1, stride_bars),
                  desc=fp.name, total=total_windows,
                  position=position, leave=False):
        out_name = f"{fp.stem}_win{i:05d}.png"
        out_path = output_dir / out_name
        if out_path.exists():
            continue

        end_idx = i + main_bars
        # slices for short, main, long
        main_data  = series[i:end_idx]
        short_data = series[max(0, end_idx-short_bars):end_idx]
        long_start = max(0, end_idx-long_bars)
        long_data  = series[long_start:end_idx]

        # normalize
        m = normalize_series(main_data)
        s = normalize_series(short_data)
        l = normalize_series(long_data)

        # compute GAFs
        gaf_m = gazer.fit_transform(m.reshape(1, -1))[0]
        gaf_s = gazer.fit_transform(s.reshape(1, -1))[0]
        gaf_l = gazer.fit_transform(l.reshape(1, -1))[0]

        # to uint8 images
        im_m = ((gaf_m + 1)/2 * 255).astype(np.uint8)
        im_s = ((gaf_s + 1)/2 * 255).astype(np.uint8)
        im_l = ((gaf_l + 1)/2 * 255).astype(np.uint8)

        # PIL images and resize
        img_m = Image.fromarray(im_m)
        img_s = Image.fromarray(im_s)
        img_l = Image.fromarray(im_l)
        if image_size and image_size != main_bars:
            img_m = img_m.resize((image_size, image_size))
            img_s = img_s.resize((image_size, image_size))
            img_l = img_l.resize((image_size, image_size))

        # stack into RGB-like channels (short, main, long)
        stacked = np.stack([np.array(img_s), np.array(img_m), np.array(img_l)], axis=2)
        out_img = Image.fromarray(stacked)

        # save
        out_img.save(out_path)
        saved += 1

    return saved


def main():
    args = parse_args()
    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_files = sorted(input_dir.glob('*.parquet'))
    if not parquet_files:
        print(f"No parquet files found in {input_dir}")
        return

    # Prepare tasks for parallel execution
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

    print(f"Done! Generated {total_saved} 3-channel GAF images.")

if __name__ == '__main__':
    main()
