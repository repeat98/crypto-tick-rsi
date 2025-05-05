#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd

def convert_csv_to_parquet(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir}")

    for csv_path in csv_files:
        # Peek at first line to detect header presence
        first_line = csv_path.open('r').readline().strip()
        # Expected header start
        has_header = first_line.lower().startswith('id,') or 'price' in first_line.lower()

        if has_header:
            df = pd.read_csv(csv_path)
        else:
            # No header: assign columns manually
            df = pd.read_csv(
                csv_path,
                header=None,
                names=['id', 'price', 'qty', 'base_qty', 'time', 'is_buyer_maker'],
            )

        if 'time' not in df.columns:
            raise ValueError(
                f"Expected a 'time' column in {csv_path.name}, but got: {df.columns.tolist()}"
            )

        # Convert ms-epoch integer to UTC datetime
        df['trade_time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
        df = df.drop(columns=['time'])

        # Rename id -> agg_id if present
        if 'id' in df.columns:
            df = df.rename(columns={'id': 'agg_id'})

        out_path = output_dir / (csv_path.stem + '.parquet')
        df.to_parquet(out_path, engine='pyarrow', index=False)
        print(f"Converted {csv_path.name} → {out_path.name}")

def main():
    p = argparse.ArgumentParser(
        description='Convert CSV tick files into Parquet for model training'
    )
    p.add_argument('input_dir', type=Path, help='Directory with .csv files')
    p.add_argument('output_dir', type=Path, help='Directory for .parquet output')
    args = p.parse_args()

    convert_csv_to_parquet(args.input_dir, args.output_dir)

if __name__ == '__main__':
    main()
