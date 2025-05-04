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
  smoothingType?: SmoothingType;
  smoothingLength?: number;
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

  private windowSize = 40;
  private buffer: number[] = [];

  private smoothingType:   SmoothingType;
  private smoothingLength: number;
  private bbMultiplier:    number;

  constructor(container: HTMLElement, options: RSIChartOptions = {}) {
    this.container = container;
    container.style.width = '100vw';
    container.style.height = '50vh';
    const { smoothingType = 'None', smoothingLength = this.windowSize, bbMultiplier = 2 } = options;
    this.smoothingType   = smoothingType;
    this.smoothingLength = smoothingLength;
    this.bbMultiplier    = bbMultiplier;

    // Create chart with fixed 0–100 scale and current theme
    this.chart = createChart(container, this.currentOptions());
    new ResizeObserver(() => {
      this.chart.resize(container.clientWidth, container.clientHeight);
      this.chart.applyOptions(this.currentOptions());
    }).observe(container);
  }

  public update(point: ChartPoint) {
    const rawVal = point.value;
    if (!isFinite(rawVal)) return;

    // Raw RSI line
    if (!this.rawSeries) {
      this.rawSeries = this.chart.addSeries(LineSeries) as ISeriesApi<'Line'>;
      this.rawSeries.applyOptions({ color: '#7E57C2', lineWidth: 2, priceLineVisible: false });
      this.rawSeries.setData([this.toLineData(point)]);
      this.rawSeries.createPriceLine({ price: 70, color: '#9E9E9E', lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: false, title: '70' });
      this.rawSeries.createPriceLine({ price: 50, color: '#9E9E9E', lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: false, title: '50' });
      this.rawSeries.createPriceLine({ price: 30, color: '#9E9E9E', lineWidth: 1, lineStyle: LineStyle.Dashed, axisLabelVisible: false, title: '30' });
    } else {
      this.rawSeries.update(this.toLineData(point));
    }

    // Rolling buffer
    this.buffer.push(rawVal);
    if (this.buffer.length > this.windowSize) this.buffer.shift();
    if (this.buffer.length < this.windowSize) return;

    // Simple SMA
    const mean = this.buffer.reduce((a, v) => a + v, 0) / this.windowSize;
    const smoothPoint = { time: point.time, value: mean };
    if (!this.smoothSeries) {
      this.smoothSeries = this.chart.addSeries(LineSeries) as ISeriesApi<'Line'>;
      this.smoothSeries.applyOptions({ color: '#F57F19', lineWidth: 2, priceLineVisible: false });
      this.smoothSeries.setData([this.toLineData(smoothPoint)]);
    } else {
      this.smoothSeries.update(this.toLineData(smoothPoint));
    }

    // Optional extra smoothing + BB
    if (this.smoothingType !== 'None') {
      const maValue = this.computeMA(this.buffer, this.smoothingType);
      if (isFinite(maValue)) {
        const maPoint = { time: point.time, value: maValue };
        if (!this.smoothingSeries) {
          this.smoothingSeries = this.chart.addSeries(LineSeries) as ISeriesApi<'Line'>;
          this.smoothingSeries.applyOptions({ color: '#FF9800', lineWidth: 1, priceLineVisible: false });
          this.smoothingSeries.setData([this.toLineData(maPoint)]);
        } else {
          this.smoothingSeries.update(this.toLineData(maPoint));
        }
        if (this.smoothingType === 'SMA+BB') {
          const slice = this.buffer.slice(-this.smoothingLength);
          const variance = slice.reduce((sum, v) => sum + (v - maValue) ** 2, 0) / this.smoothingLength;
          const stdev = Math.sqrt(variance);
          const upper = maValue + stdev * this.bbMultiplier;
          const lower = maValue - stdev * this.bbMultiplier;
          const upperData = this.toLineData({ time: point.time, value: upper });
          const lowerData = this.toLineData({ time: point.time, value: lower });

          if (!this.bbUpperSeries) {
            this.bbUpperSeries = this.chart.addSeries(LineSeries) as ISeriesApi<'Line'>;
            this.bbLowerSeries = this.chart.addSeries(LineSeries) as ISeriesApi<'Line'>;
            this.bbAreaSeries  = this.chart.addSeries(AreaSeries) as ISeriesApi<'Area'>;
            this.bbUpperSeries.applyOptions({ priceLineVisible: false });
            this.bbLowerSeries.applyOptions({ priceLineVisible: false });
            this.bbAreaSeries.applyOptions({ lineVisible: false, topColor: 'rgba(255,152,0,0.2)', bottomColor: 'rgba(255,152,0,0.05)' });
            this.bbUpperSeries.setData([upperData]);
            this.bbLowerSeries.setData([lowerData]);
            this.bbAreaSeries.setData([ upperData, lowerData ]);
          } else {
            this.bbUpperSeries.update(upperData);
            this.bbLowerSeries.update(lowerData);
            this.bbAreaSeries.update(upperData);
            this.bbAreaSeries.update(lowerData);
          }
        }
      }
    }
    // this.chart.timeScale().fitContent(); // Removed to prevent unwanted auto-fitting
  }

  public clear() {
    [ this.rawSeries,
      this.smoothSeries,
      this.smoothingSeries,
      this.bbUpperSeries,
      this.bbLowerSeries,
      this.bbAreaSeries,
    ].forEach(series => series && this.chart.removeSeries(series));

    this.rawSeries      =
    this.smoothSeries   =
    this.smoothingSeries=
    this.bbUpperSeries  =
    this.bbLowerSeries  =
    this.bbAreaSeries   = null;
    this.buffer = [];
  }

  public applyTheme() {
    const styles = getComputedStyle(document.body);
    this.chart.resize(this.container.clientWidth, this.container.clientHeight);
    this.chart.applyOptions(this.currentOptions());
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
      },
    } as any;
  }

  private toLineData(point: ChartPoint): LineData {
    const v = Math.max(0, Math.min(100, point.value));
    return { time: point.time as any, value: v };
  }

  private computeMA(values: number[], type: SmoothingType): number {
    const len = this.smoothingLength;
    const data = values.slice(-len);
    if (data.length < len) return NaN;
    switch (type) {
      case 'SMA': case 'SMA+BB':
        return data.reduce((s, v) => s + v, 0) / len;
      case 'EMA': {
        const k = 2 / (len + 1);
        return data.reduce((p, v, i) => i === 0 ? v : v * k + p * (1 - k), data[0]);
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