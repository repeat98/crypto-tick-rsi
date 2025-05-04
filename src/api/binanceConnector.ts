// src/api/binanceConnector.ts
import { Subject, timer } from 'rxjs';
import { retryWhen, delayWhen } from 'rxjs/operators';

export interface RawTick {
  e: string;        // event type
  E: number;        // event time in ms
  s: string;        // symbol
  p?: string;       // price (for trades)
  b?: string; a?: string; // bid/ask prices (for bookTicker)
}

export class BinanceConnector {
  private ws!: WebSocket;
  public ticks$ = new Subject<RawTick>();

  constructor(private symbol: string) {
    this.connect();
  }

  private connect() {
    const endpoint = `wss://stream.binance.com:9443/ws/${this.symbol.toLowerCase()}@trade`;
    this.ws = new WebSocket(endpoint);

    this.ws.onmessage = (msg) => {
      const data: RawTick = JSON.parse(msg.data);
      this.ticks$.next(data);
    };
    this.ws.onclose = () => this.reconnect();
    this.ws.onerror = () => this.ws.close();
  }

  private reconnect() {
    // exponential backoff
    timer(1000)
      .subscribe(() => this.connect());
  }

  public switchSymbol(newSymbol: string) {
    this.ws.close();
    this.symbol = newSymbol;
    this.connect();
  }

  public close() {
    this.ws.onclose = null;
    this.ws.onerror = null;
    this.ws.close();
    this.ticks$.complete();
  }

  /**
   * Fetch historical trades over the past duration and emit as RawTick[]
   */
  public async fetchHistorical(durationMs: number): Promise<RawTick[]> {
    const endTime = Date.now();
    const startTime = endTime - durationMs;
    // Binance trade endpoint returns up to 1000 recent trades
    const url = `https://api.binance.com/api/v3/trades?symbol=${this.symbol.toUpperCase()}&limit=1000`;
    const response = await fetch(url);
    const data: Array<{ id: number; price: string; qty: string; time: number }> = await response.json();
    // Filter by time and map to RawTick
    return data
      .filter(trade => trade.time >= startTime)
      .map(trade => ({
        e: 'historical',
        E: trade.time,
        s: this.symbol,
        p: trade.price,
      }));
  }
}