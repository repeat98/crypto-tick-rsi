// src/ui/mainChart.ts
import { createChart, LineSeries, CrosshairMode } from 'lightweight-charts';
import type { IChartApi, ISeriesApi } from 'lightweight-charts';

export interface ThemeableChartManager {
  applyTheme(): void;
}

export class ChartManager implements ThemeableChartManager {
  private container: HTMLElement;
  private chart: IChartApi;
  private priceSeries: ISeriesApi<'Line'> | null = null;

  constructor(priceContainer: HTMLElement) {
    this.container = priceContainer;
    this.container.style.width = '100vw';
    this.container.style.height = '50vh';

    this.chart = createChart(this.container, this.currentOptions());
    new ResizeObserver(() => {
      this.chart.resize(this.container.clientWidth, this.container.clientHeight);
      this.chart.applyOptions(this.currentOptions());
    }).observe(this.container);
  }

  public updatePrice(point: { time: string | number; value: number }) {
    const styles = getComputedStyle(document.body);
    if (!this.priceSeries) {
      this.priceSeries = this.chart.addSeries(LineSeries, {
        color: styles.getPropertyValue('--price-line-color').trim(),
      });
      this.priceSeries.setData([{ time: point.time, value: point.value }]);
    } else {
      this.priceSeries.update({ time: point.time, value: point.value });
    }
  }

  public clear() {
    if (this.priceSeries) {
      this.chart.removeSeries(this.priceSeries);
      this.priceSeries = null;
    }
  }

  public applyTheme() {
    const styles = getComputedStyle(document.body);
    this.chart.resize(this.container.clientWidth, this.container.clientHeight);
    this.chart.applyOptions(this.currentOptions());
    if (this.priceSeries) {
      this.priceSeries.applyOptions({
        color: styles.getPropertyValue('--price-line-color').trim(),
      });
    }
  }

  private currentOptions() {
    const styles = getComputedStyle(document.body);
    return {
      width:  this.container.clientWidth,
      height: this.container.clientHeight,
      layout: {
        background: { color: styles.getPropertyValue('--bg-color').trim() },
        textColor:  styles.getPropertyValue('--text-color').trim(),
      },
      grid: {
        vertLines: { color: styles.getPropertyValue('--grid-line').trim() },
        horzLines: { color: styles.getPropertyValue('--grid-line').trim() },
      },
      crosshair: {
        mode:     CrosshairMode.Normal,
        vertLine: { color: styles.getPropertyValue('--crosshair').trim(), width: 1, style: 0 },
        horzLine: { color: styles.getPropertyValue('--crosshair').trim(), width: 1, style: 0 },
      },
      rightPriceScale: { borderColor: styles.getPropertyValue('--grid-line').trim() },
      timeScale:       { borderColor: styles.getPropertyValue('--grid-line').trim(), timeVisible: true, secondsVisible: true },
      watermark: {
        visible: true,
        fontSize: 24,
        color:    styles.getPropertyValue('--watermark-color').trim(),
        text:     'TradingView Lite',
        horzAlign: 'center',
        vertAlign: 'center',
      },
    } as any;
  }
}

export function initMainChart(priceContainer: HTMLElement): { chartManager: ChartManager } {
  const chartManager = new ChartManager(priceContainer);
  return { chartManager };
}