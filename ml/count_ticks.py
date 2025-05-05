import argparse
from pathlib import Path
import pyarrow.parquet as pq

def parse_args():
    parser = argparse.ArgumentParser(
        description="Count total ticks across all SOLUSD parquet files"
    )
    parser.add_argument(
        'input_dir', type=str,
        help='Path to directory containing .parquet files'
    )
    return parser.parse_args()

def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    total_ticks = 0

    parquet_files = sorted(input_dir.glob('*.parquet'))
    if not parquet_files:
        print(f"No parquet files found in {input_dir}")
        return

    for p in parquet_files:
        pf = pq.ParquetFile(str(p))
        total_ticks += pf.metadata.num_rows

    print(f"Total parquet files: {len(parquet_files)}")
    print(f"Total ticks (rows): {total_ticks:,}")

if __name__ == '__main__':
    main()