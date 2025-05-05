#!/usr/bin/env python3
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from pyts.image import GramianAngularField
from tqdm.auto import tqdm
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert 1m-kline CSV files into 3-channel GAF images (short, main, long windows)"
    )
    parser.add_argument('input_dir', type=str, help='Directory with 1m-kline CSV files')
    parser.add_argument('output_dir', type=str, help='Where to save GAF images')
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
    mn, mx = series.min(), series.max()
    if mx == mn:
        return np.zeros_like(series)
    scaled = (series - mn) / (mx - mn)
    return scaled * 2 - 1


def process_file(args_tuple):
    csv_path, output_dir, window_bars, short_bars, long_bars, stride_bars, image_size, method, position = args_tuple
    gazer = GramianAngularField(method=method)

    # Load CSV
    df = pd.read_csv(csv_path)
    if 'close' not in df.columns:
        raise ValueError(f"Missing 'close' in {csv_path.name}")
    prices = df['close'].to_numpy(dtype=np.float32)
    n = len(prices)

    total_windows = max((n - window_bars) // stride_bars + 1, 0)
    # skip if already done
    existing = len(list(output_dir.glob(f"{csv_path.stem}_win*.png")))
    if existing >= total_windows:
        return 0

    saved = 0
    for i in tqdm(range(0, n - window_bars + 1, stride_bars),
                  desc=csv_path.name, total=total_windows,
                  position=position, leave=False):
        out_name = f"{csv_path.stem}_win{i:05d}.png"
        out_path = output_dir / out_name
        if out_path.exists():
            continue

        end = i + window_bars
        main_data  = prices[i:end]
        short_data = prices[max(0, end-short_bars):end]
        long_start = max(0, end-long_bars)
        long_data  = prices[long_start:end]

        m = normalize_series(main_data)
        s = normalize_series(short_data)
        l = normalize_series(long_data)

        gaf_m = gazer.fit_transform(m.reshape(1,-1))[0]
        gaf_s = gazer.fit_transform(s.reshape(1,-1))[0]
        gaf_l = gazer.fit_transform(l.reshape(1,-1))[0]

        im_m = Image.fromarray(((gaf_m+1)/2*255).astype(np.uint8))
        im_s = Image.fromarray(((gaf_s+1)/2*255).astype(np.uint8))
        im_l = Image.fromarray(((gaf_l+1)/2*255).astype(np.uint8))
        if image_size:
            im_m = im_m.resize((image_size,image_size))
            im_s = im_s.resize((image_size,image_size))
            im_l = im_l.resize((image_size,image_size))

        stacked = np.stack([np.array(im_s), np.array(im_m), np.array(im_l)], axis=2)
        out_img = Image.fromarray(stacked)
        out_img.save(out_path)
        saved += 1

    return saved


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # compute bar counts from window/stride strings
    freq_td = pd.Timedelta('1T')
    window_bars = int(pd.Timedelta(args.window) / freq_td)
    short_bars  = max(window_bars // 2, 1)
    long_bars   = window_bars * 2
    stride_bars = int(pd.Timedelta(args.stride) / freq_td)

    csv_files = sorted(input_dir.glob('*.csv'))
    task_args = []
    for pos, csvf in enumerate(csv_files):
        task_args.append((
            csvf, output_dir,
            window_bars, short_bars, long_bars, stride_bars,
            args.image_size, args.method, pos
        ))

    total = 0
    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        futures = {exe.submit(process_file, ta): ta[0] for ta in task_args}
        for future in tqdm(as_completed(futures), total=len(futures), desc='Files'):
            try:
                total += future.result()
            except Exception as e:
                print(f"[ERROR] {futures[future].name}: {e}")

    print(f"Done! Generated {total} 3-channel GAF images from CSV klines.")

if __name__ == '__main__':
    main()
