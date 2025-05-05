import os
import time
import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime, timedelta

BINANCE_REST_URL = 'https://api.binance.com/api/v3/aggTrades'

class HistoricalTickCollector:
    def __init__(self, symbol: str, out_dir: str, start_ts: int, end_ts: int, interval_ms: int = 60000):
        """
        symbol: e.g. 'BTCUSDT'
        start_ts, end_ts: epoch ms boundaries
        interval_ms: fetch window size in ms (max ~1h or less to control volume)
        """
        self.symbol = symbol.upper()
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.start_ts = start_ts
        self.end_ts = end_ts
        self.interval_ms = interval_ms

    def fetch_window(self, start_time: int, end_time: int):
        params = {
            'symbol': self.symbol,
            'startTime': start_time,
            'endTime': end_time,
            'limit': 1000
        }
        resp = requests.get(BINANCE_REST_URL, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        # aggTrades fields: a, p, q, f, l, T, m, M
        ticks = []
        for trade in data:
            ticks.append({
                'agg_id': trade['a'],
                'price': float(trade['p']),
                'qty': float(trade['q']),
                'first_id': trade['f'],
                'last_id': trade['l'],
                'trade_time': datetime.utcfromtimestamp(trade['T']/1000),
                'is_buyer_maker': trade['m'],
                'ignore': trade['M']
            })
        return pd.DataFrame(ticks)

    def run(self):
        current_start = self.start_ts
        file_idx = 0
        while current_start < self.end_ts:
            current_end = min(current_start + self.interval_ms, self.end_ts)
            try:
                df = self.fetch_window(current_start, current_end)
            except Exception as e:
                print(f"Error fetching {current_start}-{current_end}: {e}, retrying in 5s")
                time.sleep(5)
                continue
            if df.empty:
                # no trades, advance
                current_start = current_end
                continue
            # save parquet
            table = pa.Table.from_pandas(df)
            filename = os.path.join(self.out_dir, f"{self.symbol}_hist_{file_idx:06d}.parquet")
            pq.write_table(table, filename)
            print(f"Wrote {len(df)} trades to {filename}")
            # advance start to last trade timestamp + 1ms
            last_ts = int(df['trade_time'].iloc[-1].timestamp() * 1000)
            current_start = last_ts + 1
            file_idx += 1
            # to respect API weight limits
            time.sleep(0.2)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser("Fetch historical aggTrades from Binance in windows")
    parser.add_argument('--symbol', type=str, required=True)
    parser.add_argument('--start', type=str, required=True, help='YYYY-MM-DD')
    parser.add_argument('--end', type=str, required=True, help='YYYY-MM-DD')
    parser.add_argument('--out_dir', type=str, default='historical_data')
    parser.add_argument('--interval_min', type=int, default=60, help='Window size in minutes per request')
    args = parser.parse_args()

    start_dt = datetime.fromisoformat(args.start)
    end_dt = datetime.fromisoformat(args.end)
    collector = HistoricalTickCollector(
        symbol=args.symbol,
        out_dir=args.out_dir,
        start_ts=int(start_dt.timestamp() * 1000),
        end_ts=int(end_dt.timestamp() * 1000),
        interval_ms=args.interval_min * 60 * 1000
    )
    collector.run()
