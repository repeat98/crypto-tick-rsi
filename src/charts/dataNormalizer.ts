// src/charts/dataNormalizer.ts
import { RawTick } from '../api/binanceConnector';

export interface ChartPoint {
  time: number;    // UNIX timestamp in seconds
  value: number;
}

export function normalizeTick(tick: RawTick): ChartPoint {
  // trade event
  if (tick.p) {
    return {
      time: Math.floor(tick.E / 1000),
      value: parseFloat(tick.p),
    };
  }
  // bookTicker event (mid-price)
  const mid = (parseFloat(tick.b!) + parseFloat(tick.a!)) / 2;
  return {
    time: Math.floor(tick.E / 1000),
    value: mid,
  };
}