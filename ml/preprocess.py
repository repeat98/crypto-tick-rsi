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
        description="Convert time series parquet files into Gramian Angular Field images"
    )
    parser.add_argument('input_dir', type=str, help='Directory with .parquet files')
    parser.add_argument('output_dir', type=str, help='Where to save GAF images')
    parser.add_argument('--seq_len', type=int, default=100, help='Sliding window length')
    parser.add_argument('--image_size', type=int, default=None,
                        help='Resize GAF images to this size (pixels)')
    parser.add_argument('--method', type=str, choices=['summation', 'difference'],
                        default='summation', help='GASF or GADF')
    parser.add_argument('--stride', type=int, default=512, help='Sliding window stride')
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
    """
    Worker for a single file.
    args_tuple = (
      file_path: Path,
      output_dir: Path,
      seq_len: int,
      image_size: int|None,
      method: str,
      stride: int,
      position: int
    )
    """
    file_path, output_dir, seq_len, image_size, method, stride, position = args_tuple
    gazer = GramianAngularField(method=method)
    # read parquet
    table = pq.read_table(str(file_path))
    df = table.to_pandas()
    if 'price' not in df.columns:
        raise ValueError(f"No 'price' column in {file_path}")
    series = df['price'].to_numpy(dtype=np.float32)
    n = len(series)
    if n <= seq_len:
        return 0

    # count windows
    total_windows = (n - seq_len) // stride + 1

    saved = 0
    # inner, per-window progress bar
    for i in tqdm(
        range(0, n - seq_len + 1, stride),
        desc=f"{file_path.name}",
        total=total_windows,
        position=position,
        leave=False
    ):
        out_name = f"{file_path.stem}_win{i:05d}.png"
        out_path = output_dir / out_name
        if out_path.exists():
            continue
        window = series[i : i + seq_len]
        scaled = normalize_series(window)
        gaf = gazer.fit_transform(scaled.reshape(1, -1))[0]
        img_arr = ((gaf + 1) / 2 * 255).astype(np.uint8)
        img = Image.fromarray(img_arr)
        if image_size and image_size != seq_len:
            img = img.resize((image_size, image_size))
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

    # prepare tasks, assigning each file a unique tqdm position
    task_args = []
    for pos, fp in enumerate(parquet_files):
        task_args.append((
            fp,
            output_dir,
            args.seq_len,
            args.image_size,
            args.method,
            args.stride,
            pos  # tqdm position
        ))

    total_saved = 0
    # outer bar to track file‐completion
    with ProcessPoolExecutor(max_workers=args.workers) as exe:
        futures = {exe.submit(process_file, ta): ta[0] for ta in task_args}
        for future in tqdm(as_completed(futures),
                           total=len(futures),
                           desc='Files',
                           position=len(task_args),
                           leave=True):
            fp = futures[future]
            try:
                saved = future.result()
                total_saved += saved
            except Exception as e:
                print(f"[ERROR] {fp.name}: {e}")

    print(f"Done! Generated {total_saved} new GAF images.")

if __name__ == '__main__':
    main()