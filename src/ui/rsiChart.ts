import {
  createChart,
  IChartApi,
  ISeriesApi,
  LineStyle,
  LineData,
  AreaData,
  LineSeries,
  AreaSeries,
} from 'lightweight-charts';
import type { ChartPoint } from '../charts/dataNormalizer';

type SmoothingType = 'None' | 'SMA' | 'SMA+BB' | 'EMA' | 'SMMA' | 'WMA' | 'VWMA';

interface RSIChartOptions {
  period?: number;               // RSI lookback
  smoothingType?: SmoothingType; // smoothing for extra MA series
  smoothingLength?: number;      // for extra smoothing / BB
  bbMultiplier?: number;
  maTimeframeSeconds?: number;   // timeframe for the 1m MA in seconds
}

export class RSIChart {
  private container: HTMLElement;
  private chart: IChartApi;

  private rawSeries:      ISeriesApi<'Line'>  | null = null;
  private smoothSeries:   ISeriesApi<'Line'>  | null = null;
  private smoothingSeries:ISeriesApi<'Line'>  | null = null;
  private bbUpperSeries:  ISeriesApi<'Line'>  | null = null;
  private bbLowerSeries:  ISeriesApi<'Line'>  | null = null;
  private bbAreaSeries:   ISeriesApi<'Area'>  | null = null;
  private maSeries:       ISeriesApi<'Line'>  | null = null;  // 1m timeframe MA series

  private period: number;
  private prevPrice?: number;
  private gains: number[] = [];
  private losses: number[] = [];
  private avgGain?: number;
  private avgLoss?: number;

  private windowSize: number;
  private buffer: number[] = [];
  private maBuffer: { time: number; value: number }[] = []; // buffer for timeframe MA

  private smoothingType:   SmoothingType;
  private smoothingLength: number;
  private bbMultiplier:    number;
  private maTimeframe:     number; // in seconds

  constructor(container: HTMLElement, options: RSIChartOptions = {}) {
    this.container = container;
    container.style.width  = '100vw';
    container.style.height = '50vh';

    // User-specified or defaults:
    this.period           = options.period ?? 14;           // default RSI length 14
    this.windowSize       = options.smoothingLength ?? 40;
    this.smoothingType    = options.smoothingType   ?? 'SMA'; // default SMA smoothing
    this.smoothingLength  = options.smoothingLength ?? this.windowSize;
    this.bbMultiplier     = options.bbMultiplier    ?? 2;
    this.maTimeframe      = options.maTimeframeSeconds ?? 60; // 1 minute

    this.chart = createChart(container, this.currentOptions());
    new ResizeObserver(() => {
      this.chart.resize(container.clientWidth, container.clientHeight);
      this.chart.applyOptions(this.currentOptions());
    }).observe(container);
  }

  public update(point: ChartPoint) {
    const price = point.value;
    if (!isFinite(price)) return;

    // RSI core calculation (unchanged)...
    let rsiValue: number;
    if (this.prevPrice === undefined) {
      this.prevPrice = price;
      this.gains.push(0);
      this.losses.push(0);
      rsiValue = 50;
    } else {
      const change = price - this.prevPrice;
      this.prevPrice = price;
      this.gains.push(Math.max(0, change));
      this.losses.push(Math.max(0, -change));
      if (this.gains.length > this.period) { this.gains.shift(); this.losses.shift(); }
      if (this.avgGain === undefined && this.gains.length === this.period) {
        this.avgGain = this.gains.reduce((sum, v) => sum + v, 0) / this.period;
        this.avgLoss = this.losses.reduce((sum, v) => sum + v, 0) / this.period;
      }
      if (this.avgGain !== undefined) {
        this.avgGain = (this.avgGain * (this.period - 1) + this.gains[this.gains.length - 1]) / this.period;
        this.avgLoss = (this.avgLoss! * (this.period - 1) + this.losses[this.losses.length - 1]) / this.period;
        const rs = this.avgLoss === 0 ? Infinity : this.avgGain! / this.avgLoss!;
        rsiValue = 100 - 100 / (1 + rs);
      } else {
        rsiValue = 50;
      }
    }
    rsiValue = Math.max(0, Math.min(100, rsiValue));

    // Draw raw RSI...
    if (!this.rawSeries) {
      this.rawSeries = this.chart.addSeries(LineSeries);
      this.rawSeries.applyOptions({ color: '#7E57C2', lineWidth: 2, priceLineVisible: false });
      this.rawSeries.setData([{ time: point.time as any, value: rsiValue }]);
      [70, 50, 30].forEach(level =>
        this.rawSeries!.createPriceLine({ price: level, color: '#9E9E9E', lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: false, title: String(level) })
      );
    } else {
      this.rawSeries.update({ time: point.time as any, value: rsiValue });
    }

    // --- 1m timeframe MA of RSI ---
    const ts = point.time as number;
    this.maBuffer.push({ time: ts, value: rsiValue });
    const cutoff = ts - this.maTimeframe;
    this.maBuffer = this.maBuffer.filter(d => d.time >= cutoff);
    const maValue = this.maBuffer.reduce((s, d) => s + d.value, 0) / this.maBuffer.length;
    const maPoint: LineData = { time: point.time as any, value: maValue };
    if (!this.maSeries) {
      this.maSeries = this.chart.addSeries(LineSeries);
      this.maSeries.applyOptions({ color: '#29B6F6', lineWidth: 2, priceLineVisible: false });
      this.maSeries.setData([maPoint]);
    } else {
      this.maSeries.update(maPoint);
    }

    // Continue SMA buffer smoothing / BB logic...
    this.buffer.push(rsiValue);
    if (this.buffer.length > this.windowSize) this.buffer.shift();
    if (this.buffer.length < this.windowSize) return;

    const mean = this.buffer.reduce((a, v) => a + v, 0) / this.windowSize;
    if (!this.smoothSeries) {
      this.smoothSeries = this.chart.addSeries(LineSeries);
      this.smoothSeries.applyOptions({ color: '#F57F19', lineWidth: 2, priceLineVisible: false });
      this.smoothSeries.setData([{ time: point.time as any, value: mean }]);
    } else {
      this.smoothSeries.update({ time: point.time as any, value: mean });
    }

    if (this.smoothingType !== 'None') {
      const extra = this.computeMA(this.buffer, this.smoothingType);
      if (isFinite(extra)) {
        if (!this.smoothingSeries) {
          this.smoothingSeries = this.chart.addSeries(LineSeries);
          this.smoothingSeries.applyOptions({ color: '#FF9800', lineWidth: 1, priceLineVisible: false });
          this.smoothingSeries.setData([{ time: point.time as any, value: extra }]);
        } else {
          this.smoothingSeries.update({ time: point.time as any, value: extra });
        }
        if (this.smoothingType === 'SMA+BB') {
          const slice = this.buffer.slice(-this.smoothingLength);
          const variance = slice.reduce((sum, v) => sum + (v - extra) ** 2, 0) / this.smoothingLength;
          const stdev = Math.sqrt(variance);
          const upper = extra + stdev * this.bbMultiplier;
          const lower = extra - stdev * this.bbMultiplier;
          const up: LineData = { time: point.time as any, value: upper };
          const low: LineData = { time: point.time as any, value: lower };
          if (!this.bbUpperSeries) {
            this.bbUpperSeries = this.chart.addSeries(LineSeries);
            this.bbLowerSeries = this.chart.addSeries(LineSeries);
            this.bbAreaSeries  = this.chart.addSeries(AreaSeries);
            this.bbUpperSeries.applyOptions({ priceLineVisible: false });
            this.bbLowerSeries.applyOptions({ priceLineVisible: false });
            this.bbAreaSeries.applyOptions({ lineVisible: false, topColor: 'rgba(255,152,0,0.2)', bottomColor: 'rgba(255,152,0,0.05)' });
            this.bbUpperSeries.setData([up]);
            this.bbLowerSeries.setData([low]);
            this.bbAreaSeries.setData([up, low]);
          } else {
            this.bbUpperSeries.update(up);
            this.bbLowerSeries.update(low);
            this.bbAreaSeries.update(up);
            this.bbAreaSeries.update(low);
          }
        }
      }
    }
  }

  public clear() {
    [
      this.rawSeries,
      this.maSeries,
      this.smoothSeries,
      this.smoothingSeries,
      this.bbUpperSeries,
      this.bbLowerSeries,
      this.bbAreaSeries,
    ].forEach(s => s && this.chart.removeSeries(s));
    this.rawSeries = this.maSeries = this.smoothSeries = this.smoothingSeries = this.bbUpperSeries = this.bbLowerSeries = this.bbAreaSeries = null;
    this.prevPrice = undefined;
    this.gains = [];
    this.losses = [];
    this.avgGain = this.avgLoss = undefined;
    this.buffer = [];
    this.maBuffer = [];
  }

  public applyTheme() {
    this.chart.resize(this.container.clientWidth, this.container.clientHeight);
    this.chart.applyOptions(this.currentOptions());
  }

  public zoomToLast(seconds: number) {
    const now = Math.floor(Date.now() / 1000);
    this.chart.timeScale().setVisibleRange({ from: now - seconds, to: now });
  }

  private currentOptions() {
    const styles = getComputedStyle(document.body);
    return {
      width:  this.container.clientWidth,
      height: this.container.clientHeight,
      layout: {
        background: { color: styles.getPropertyValue('--bg-color').trim() },
        textColor:  styles.getPropertyValue('--text-color').trim(),
        padding:    { top: 0, bottom: 0 },
      },
      grid: {
        vertLines: { color: styles.getPropertyValue('--grid-line').trim() },
        horzLines: { color: styles.getPropertyValue('--grid-line').trim() },
      },
      priceScale: {
        position:      'right',
        autoScale:     false,
        visible:       true,
        borderVisible: false,
        scaleMargins:  { top: 0, bottom: 0 },
        minValue:      0,
        maxValue:      100,
      },
      timeScale: {
        borderVisible: false,
        timeVisible:   true,
        secondsVisible:true,
        shiftVisibleRangeOnNewBar: true,
      },
    } as any;
  }

  private computeMA(values: number[], type: SmoothingType): number {
    const len = this.smoothingLength;
    const data = values.slice(-len);
    if (data.length < len) return NaN;
    switch (type) {
      case 'SMA':
      case 'SMA+BB':
        return data.reduce((s, v) => s + v, 0) / len;
      case 'EMA': {
        const k = 2 / (len + 1);
        return data.reduce((p, v, i) => (i === 0 ? v : v * k + p * (1 - k)), data[0]);
      }
      case 'SMMA':
        return data.slice(1).reduce((p, v) => (p * (len - 1) + v) / len, data[0]);
      case 'WMA':
        return data.reduce((s, v, i) => s + v * (i + 1), 0) / ((len * (len + 1)) / 2);
      default:
        return NaN;
    }
  }
}
