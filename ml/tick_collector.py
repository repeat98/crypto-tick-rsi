import asyncio
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import os
import json
import websockets
from datetime import datetime

BINANCE_WS_URL = "wss://stream.binance.com:9443/ws/{symbol}@trade"

class TickCollector:
    def __init__(self, symbol: str, out_dir: str, batch_size: int = 10000):
        self.symbol = symbol.lower()
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.batch_size = batch_size
        self.buffer = []
        self.file_index = 0

    async def connect(self):
        url = BINANCE_WS_URL.format(symbol=self.symbol)
        async with websockets.connect(url) as ws:
            print(f"Connected to {url}")
            await self._collect(ws)

    async def _collect(self, ws):
        async for message in ws:
            data = json.loads(message)
            # Standardize fields
            tick = {
                'time': datetime.utcfromtimestamp(data['E'] / 1000),
                'price': float(data['p']),
                'symbol': data['s'],
                'event_time': data['E'],
            }
            self.buffer.append(tick)
            if len(self.buffer) >= self.batch_size:
                self._flush()

    def _flush(self):
        df = pd.DataFrame(self.buffer)
        table = pa.Table.from_pandas(df)
        filename = os.path.join(
            self.out_dir,
            f"{self.symbol}_ticks_{self.file_index:06d}.parquet"
        )
        pq.write_table(table, filename)
        print(f"Wrote {len(self.buffer)} ticks to {filename}")
        self.buffer = []
        self.file_index += 1

    def close(self):
        if self.buffer:
            self._flush()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Collect Binance tick data to Parquet files")
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='Trading symbol, e.g. BTCUSDT')
    parser.add_argument('--out_dir', type=str, default='data', help='Directory to save Parquet files')
    parser.add_argument('--batch_size', type=int, default=10000, help='Ticks per file')
    args = parser.parse_args()

    collector = TickCollector(symbol=args.symbol, out_dir=args.out_dir, batch_size=args.batch_size)
    try:
        asyncio.run(collector.connect())
    except KeyboardInterrupt:
        print("Interrupted, flushing remaining data...")
        collector.close()
        print("Done.")
