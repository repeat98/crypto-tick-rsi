import { ChartPoint } from './dataNormalizer';

// EMA with smoothing factor α = 2/(n+1)
export class EMA {
  private prevEma?: number;
  private alpha: number;
  constructor(period: number) {
    this.alpha = 2 / (period + 1);
  }
  update(point: ChartPoint): ChartPoint {
    const price = point.value;
    const ema = this.prevEma === undefined
      ? price
      : price * this.alpha + this.prevEma * (1 - this.alpha);
    this.prevEma = ema;
    return { time: point.time, value: ema };
  }
}

// VWAP: cumulative (price * volume) / cumulative volume
export class VWAP {
  private cumPV = 0;
  private cumV = 0;
  update(point: ChartPoint, volume: number): ChartPoint {
    this.cumPV += point.value * volume;
    this.cumV += volume;
    const vwap = this.cumPV / this.cumV;
    return { time: point.time, value: vwap };
  }
}

// RSI with Wilder's smoothing
export class RSI {
  private period: number;
  private gains: number[] = [];
  private losses: number[] = [];
  private prevAvgGain?: number;
  private prevAvgLoss?: number;

  constructor(period: number) {
    this.period = period;
  }

  update(point: ChartPoint, prevPoint?: ChartPoint): ChartPoint {
    if (!prevPoint) {
      this.gains.push(0);
      this.losses.push(0);
      return { time: point.time, value: 50 };
    }
    const change = point.value - prevPoint.value;
    this.gains.push(Math.max(change, 0));
    this.losses.push(Math.max(-change, 0));

    if (this.gains.length > this.period) {
      this.gains.shift();
      this.losses.shift();
    }

    if (this.prevAvgGain === undefined) {
      // first RSI value: simple average
      this.prevAvgGain = this.gains.reduce((a, b) => a + b, 0) / this.period;
      this.prevAvgLoss = this.losses.reduce((a, b) => a + b, 0) / this.period;
    } else {
      // Wilder’s smoothing
      this.prevAvgGain = (this.prevAvgGain * (this.period - 1) + this.gains[this.gains.length - 1]) / this.period;
      this.prevAvgLoss = (this.prevAvgLoss * (this.period - 1) + this.losses[this.losses.length - 1]) / this.period;
    }

    const rs = this.prevAvgLoss === 0 ? 100 : this.prevAvgGain! / this.prevAvgLoss!;
    const rsi = 100 - 100 / (1 + rs);
    return { time: point.time, value: rsi };
  }
}