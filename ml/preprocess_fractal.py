#!/usr/bin/env python3
"""
Convert time-series Parquet files into fractal multi-scale GAF arrays using price only.
Each example produces a `depth`-channel N×N tensor, saved as compressed `.npz`.
Channels correspond to GAFs at scales 2^k × base window (k=0..depth-1).
Recommended: depth=4 for crypto (1,2,4,8 min scales).
"""
import os
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyts.image import GramianAngularField
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess price series into fractal multi-scale GAF .npz files"
    )
    parser.add_argument('input_dir', type=str, help='Directory with .parquet files')
    parser.add_argument('output_dir', type=str, help='Where to save .npz arrays')
    parser.add_argument('--freq', type=str, default='1S',
                        help="Resample frequency (e.g. '1S','1T')")
    parser.add_argument('--window', type=str, default='1T',
                        help="Base window length (e.g. '1T' for 1 minute)")
    parser.add_argument('--stride', type=str, default='30S',
                        help="Stride between windows (e.g. '30S')")
    parser.add_argument('--depth', type=int, default=4,
                        help='Number of fractal scales (powers of two)')
    parser.add_argument('--image_size', type=int, default=224,
                        help='Resize each GAF to this size (pixels)')
    parser.add_argument('--method', type=str, choices=['summation','difference'],
                        default='summation', help='GAF method')
    parser.add_argument('--workers', type=int, default=os.cpu_count(),
                        help='Number of parallel processes')
    return parser.parse_args()


def normalize_series(series: np.ndarray) -> np.ndarray:
    mn, mx = series.min(), series.max()
    if mx == mn:
        return np.zeros_like(series)
    scaled = (series - mn) / (mx - mn)
    return scaled * 2 - 1


def process_file(args_tuple):
    fp, out_dir, freq, base_bars, stride_bars, depth, img_size, method, position = args_tuple
    gazer = GramianAngularField(method=method)

    # Read parquet and set datetime index
    df = pq.read_table(str(fp)).to_pandas()
    if 'price' not in df.columns or 'trade_time' not in df.columns:
        raise ValueError(f"Missing required columns in {fp.name}")
    df['trade_time'] = pd.to_datetime(df['trade_time'])
    df = df.set_index('trade_time').sort_index()

    # Resample price series
    series = df['price'].resample(freq).last().ffill().to_numpy(dtype=np.float32)
    n = len(series)

    total_windows = max((n - base_bars) // stride_bars + 1, 0)
    saved = 0
    scales = [2**i for i in range(depth)]

    for i in tqdm(range(0, n - base_bars + 1, stride_bars),
                  desc=fp.name, total=total_windows,
                  position=position, leave=False):
        end = i + base_bars
        channels = []
        for s in scales:
            win_len = base_bars * s
            start = max(0, end - win_len)
            win = series[start:end]
            scaled = normalize_series(win)
            gaf = gazer.fit_transform(scaled.reshape(1, -1))[0]
            # resize if needed
            if img_size and gaf.shape[0] != img_size:
                from PIL import Image
                img = Image.fromarray(((gaf + 1)/2 * 255).astype(np.uint8))
                img = img.resize((img_size, img_size), Image.BILINEAR)
                arr = np.array(img, dtype=np.float32) / 127.5 - 1.0
            else:
                arr = gaf.astype(np.float32)
            channels.append(arr)
        # stack into (depth, H, W)
        tensor = np.stack(channels, axis=0)
        # save as .npz
        out_name = f"{fp.stem}_win{i:05d}.npz"
        out_path = out_dir / out_name
        if out_path.exists():
            continue
        np.savez_compressed(str(out_path), data=tensor)
        saved += 1

    return saved


def main():
    args = parse_args()
    inp = Path(args.input_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    freq = args.freq
    base_bars = int(pd.Timedelta(args.window) / pd.Timedelta(freq))
    stride_bars = max(int(pd.Timedelta(args.stride) / pd.Timedelta(freq)), 1)
    if base_bars < 1:
        raise ValueError("Base window smaller than freq")

    files = sorted(inp.glob('*.parquet'))
    tasks = [(f, out, freq, base_bars, stride_bars,
              args.depth, args.image_size, args.method, pos)
             for pos, f in enumerate(files)]

    total = 0
    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        futures = {exe.submit(process_file, t): t[0] for t in tasks}
        for fut in as_completed(futures):
            f = futures[fut]
            try:
                total += fut.result()
            except Exception as e:
                print(f"[ERROR] {f.name}: {e}")

    print(f"Done! Generated {total} fractal GAF .npz files.")

if __name__ == '__main__':
    main()
