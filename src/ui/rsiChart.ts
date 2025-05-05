// src/charts/rsiChart.ts
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
  smoothingType?: SmoothingType;
  smoothingLength?: number;      // for extra smoothing / BB
  bbMultiplier?: number;
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

  private period: number;
  private prevPrice?: number;
  private gains: number[] = [];
  private losses: number[] = [];
  private avgGain?: number;
  private avgLoss?: number;

  private windowSize: number;
  private buffer: number[] = [];

  private smoothingType:   SmoothingType;
  private smoothingLength: number;
  private bbMultiplier:    number;

  constructor(container: HTMLElement, options: RSIChartOptions = {}) {
    this.container = container;
    container.style.width  = '100vw';
    container.style.height = '50vh';

    this.period           = options.period ?? 14;
    this.windowSize       = options.smoothingLength ?? 40;
    this.smoothingType    = options.smoothingType   ?? 'None';
    this.smoothingLength  = options.smoothingLength ?? this.windowSize;
    this.bbMultiplier     = options.bbMultiplier    ?? 2;

    // Create chart with fixed 0–100 scale and current theme
    this.chart = createChart(container, this.currentOptions());
    new ResizeObserver(() => {
      this.chart.resize(container.clientWidth, container.clientHeight);
      this.chart.applyOptions(this.currentOptions());
    }).observe(container);
  }

  public update(point: ChartPoint) {
    const price = point.value;
    if (!isFinite(price)) return;

    // First tick: seed
    let rsiValue: number;
    if (this.prevPrice === undefined) {
      this.prevPrice = price;
      // push zero change to build arrays
      this.gains.push(0);
      this.losses.push(0);
      rsiValue = 50;  // neutral start
    } else {
      const change = price - this.prevPrice;
      this.prevPrice = price;
      this.gains.push(Math.max(0, change));
      this.losses.push(Math.max(0, -change));
      if (this.gains.length > this.period) {
        this.gains.shift();
        this.losses.shift();
      }

      if (this.avgGain === undefined && this.gains.length === this.period) {
        // first average
        this.avgGain = this.gains.reduce((sum, v) => sum + v, 0) / this.period;
        this.avgLoss = this.losses.reduce((sum, v) => sum + v, 0) / this.period;
      }

      if (this.avgGain !== undefined) {
        // Wilder smoothing
        this.avgGain = (this.avgGain * (this.period - 1) + this.gains[this.gains.length - 1]) / this.period;
        this.avgLoss = (this.avgLoss! * (this.period - 1) + this.losses[this.losses.length - 1]) / this.period;

        const rs = this.avgLoss === 0 ? Infinity : this.avgGain! / this.avgLoss!;
        rsiValue = 100 - 100 / (1 + rs);
      } else {
        // not enough data yet
        rsiValue = 50;
      }
    }

    // clamp to [0,100]
    rsiValue = Math.max(0, Math.min(100, rsiValue));

    // Raw RSI line
    if (!this.rawSeries) {
      this.rawSeries = this.chart.addSeries(LineSeries) as ISeriesApi<'Line'>;
      this.rawSeries.applyOptions({ color: '#7E57C2', lineWidth: 2, priceLineVisible: false });
      this.rawSeries.setData([{ time: point.time as any, value: rsiValue }]);
      [70, 50, 30].forEach(level =>
        this.rawSeries!.createPriceLine({
          price: level,
          color: '#9E9E9E',
          lineWidth: 1,
          lineStyle: LineStyle.Dashed,
          axisLabelVisible: false,
          title: String(level),
        })
      );
    } else {
      this.rawSeries.update({ time: point.time as any, value: rsiValue });
    }

    // Rolling buffer of RSI for SMA smoothing
    this.buffer.push(rsiValue);
    if (this.buffer.length > this.windowSize) this.buffer.shift();
    if (this.buffer.length < this.windowSize) return;

    // Simple SMA
    const mean = this.buffer.reduce((a, v) => a + v, 0) / this.windowSize;
    const smoothPoint = { time: point.time as any, value: mean };
    if (!this.smoothSeries) {
      this.smoothSeries = this.chart.addSeries(LineSeries) as ISeriesApi<'Line'>;
      this.smoothSeries.applyOptions({ color: '#F57F19', lineWidth: 2, priceLineVisible: false });
      this.smoothSeries.setData([smoothPoint as LineData]);
    } else {
      this.smoothSeries.update(smoothPoint as LineData);
    }

    // Optional extra smoothing + BB on the SMA series
    if (this.smoothingType !== 'None') {
      const maValue = this.computeMA(this.buffer, this.smoothingType);
      if (isFinite(maValue)) {
        const maPoint = { time: point.time as any, value: maValue };
        if (!this.smoothingSeries) {
          this.smoothingSeries = this.chart.addSeries(LineSeries) as ISeriesApi<'Line'>;
          this.smoothingSeries.applyOptions({ color: '#FF9800', lineWidth: 1, priceLineVisible: false });
          this.smoothingSeries.setData([maPoint as LineData]);
        } else {
          this.smoothingSeries.update(maPoint as LineData);
        }
        if (this.smoothingType === 'SMA+BB') {
          const slice = this.buffer.slice(-this.smoothingLength);
          const variance = slice.reduce((sum, v) => sum + (v - maValue) ** 2, 0) / this.smoothingLength;
          const stdev = Math.sqrt(variance);
          const upper = maValue + stdev * this.bbMultiplier;
          const lower = maValue - stdev * this.bbMultiplier;
          const upperData = { time: point.time as any, value: upper } as LineData;
          const lowerData = { time: point.time as any, value: lower } as LineData;

          if (!this.bbUpperSeries) {
            this.bbUpperSeries = this.chart.addSeries(LineSeries) as ISeriesApi<'Line'>;
            this.bbLowerSeries = this.chart.addSeries(LineSeries) as ISeriesApi<'Line'>;
            this.bbAreaSeries  = this.chart.addSeries(AreaSeries) as ISeriesApi<'Area'>;
            this.bbUpperSeries.applyOptions({ priceLineVisible: false });
            this.bbLowerSeries.applyOptions({ priceLineVisible: false });
            this.bbAreaSeries.applyOptions({
              lineVisible: false,
              topColor: 'rgba(255,152,0,0.2)',
              bottomColor: 'rgba(255,152,0,0.05)',
            });
            this.bbUpperSeries.setData([upperData]);
            this.bbLowerSeries.setData([lowerData]);
            (this.bbAreaSeries as ISeriesApi<'Area'>).setData([upperData, lowerData] as AreaData[]);
          } else {
            this.bbUpperSeries.update(upperData);
            this.bbLowerSeries.update(lowerData);
            (this.bbAreaSeries as ISeriesApi<'Area'>).update(upperData);
            (this.bbAreaSeries as ISeriesApi<'Area'>).update(lowerData);
          }
        }
      }
    }
  }

  public clear() {
    [
      this.rawSeries,
      this.smoothSeries,
      this.smoothingSeries,
      this.bbUpperSeries,
      this.bbLowerSeries,
      this.bbAreaSeries,
    ].forEach(s => s && this.chart.removeSeries(s));
    this.rawSeries = this.smoothSeries = this.smoothingSeries =
      this.bbUpperSeries = this.bbLowerSeries = this.bbAreaSeries = null;
    this.prevPrice = undefined;
    this.gains = [];
    this.losses = [];
    this.avgGain = this.avgLoss = undefined;
    this.buffer = [];
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