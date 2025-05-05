import {
  createChart,
  IChartApi,
  ISeriesApi,
  HistogramSeries,
  LineSeries,
  HistogramData,
  LineData,
} from 'lightweight-charts';

export interface SMOChartOptions {
  bbLength?: number;
  bbMult?: number;
  kcLength?: number;
  kcMult?: number;
  useTrueRange?: boolean;
}

export class SqueezeMomentumChart {
  private container: HTMLElement;
  private chart: IChartApi;
  private histSeries: ISeriesApi<'Histogram'> | null = null;
  private zeroSeries: ISeriesApi<'Line'> | null = null;

  private bbLength: number;
  private bbMult: number;
  private kcLength: number;
  private kcMult: number;
  private useTrueRange: boolean;

  private closes: number[] = [];
  private highs: number[] = [];
  private lows: number[] = [];

  private prevVal: number = 0;

  constructor(container: HTMLElement, options: SMOChartOptions = {}) {
    this.container = container;
    container.style.width  = '100vw';
    container.style.height = '50vh';

    this.bbLength = options.bbLength ?? 20;
    this.bbMult = options.bbMult ?? 2.0;
    this.kcLength = options.kcLength ?? 20;
    this.kcMult = options.kcMult ?? 1.5;
    this.useTrueRange = options.useTrueRange ?? true;

    this.chart = createChart(container, this.currentOptions());
    new ResizeObserver(() => {
      this.chart.resize(container.clientWidth, container.clientHeight);
      this.chart.applyOptions(this.currentOptions());
    }).observe(container);
  }

  public update(point: { time: string | number; value: number }) {
    const price = point.value;
    if (!isFinite(price)) return;

    // Maintain OHLC buffers
    this.closes.push(price);
    this.highs.push(price);
    this.lows.push(price);
    if (this.closes.length > this.kcLength) {
      this.closes.shift();
      this.highs.shift();
      this.lows.shift();
    }
    if (this.closes.length < this.kcLength) return;

    // BB calculations
    const bbSource = this.closes.slice(-this.bbLength);
    const bbBasis = bbSource.reduce((a, b) => a + b, 0) / this.bbLength;
    const meanDiffs = bbSource.map(v => (v - bbBasis) ** 2);
    const bbDev = Math.sqrt(meanDiffs.reduce((a, b) => a + b, 0) / this.bbLength) * this.bbMult;
    const upperBB = bbBasis + bbDev;
    const lowerBB = bbBasis - bbDev;

    // KC calculations
    const ma = this.closes.slice(-this.kcLength).reduce((a, b) => a + b, 0) / this.kcLength;
    const trArray = this.useTrueRange
      ? this.closes.map((c, i) => {
          if (i === 0) return this.highs[i] - this.lows[i];
          return Math.max(
            this.highs[i] - this.lows[i],
            Math.abs(this.highs[i] - this.closes[i-1]),
            Math.abs(this.lows[i] - this.closes[i-1])
          );
        })
      : this.highs.map((h, i) => h - this.lows[i]);
    const kcRange = trArray.slice(-this.kcLength).reduce((a, b) => a + b, 0) / this.kcLength;
    const upperKC = ma + kcRange * this.kcMult;
    const lowerKC = ma - kcRange * this.kcMult;

    // Squeeze conditions
    const sqzOn = lowerBB > lowerKC && upperBB < upperKC;
    const sqzOff = lowerBB < lowerKC && upperBB > upperKC;
    const noSqz = !sqzOn && !sqzOff;

    // Momentum value (linear regression deviation)
    const highArr = this.highs.slice(-this.kcLength);
    const lowArr = this.lows.slice(-this.kcLength);
    const hh = Math.max(...highArr);
    const ll = Math.min(...lowArr);
    const avgHLLL = (hh + ll) / 2;
    const avgClose = this.closes.slice(-this.kcLength).reduce((a, b) => a + b, 0) / this.kcLength;
    const val = this.closes[this.closes.length-1] - (avgHLLL + avgClose) / 2;

    // Colors
    const up = val > 0;
    const rising = val > this.prevVal;
    const bcolor = up ? (rising ? 'lime' : 'green') : (!up && val < this.prevVal ? 'red' : 'maroon');
    const scolor = noSqz ? 'blue' : sqzOn ? 'black' : 'gray';
    this.prevVal = val;

    // Plot histogram
    const bar: HistogramData = { time: point.time as any, value: val, color: bcolor };
    if (!this.histSeries) {
      this.histSeries = this.chart.addSeries(HistogramSeries, {
        priceLineVisible: false,
        scaleMargins: { top: 0.7, bottom: 0.2 },
      }) as ISeriesApi<'Histogram'>;
      this.histSeries.setData([bar]);
      this.histSeries.priceScale().applyOptions({ autoScale: false, scaleMargins: { top: 0.7, bottom: 0.2 } });
    } else {
      this.histSeries.update(bar);
    }

    // Plot zero line
    const zeroPoint: LineData = { time: point.time as any, value: 0 };
    if (!this.zeroSeries) {
      this.zeroSeries = this.chart.addSeries(LineSeries, { lineWidth: 2, color: scolor, priceLineVisible: false });
      this.zeroSeries.setData([zeroPoint]);
    } else {
      this.zeroSeries.applyOptions({ color: scolor });
      this.zeroSeries.update(zeroPoint);
    }
  }

  public clear() {
    [this.histSeries, this.zeroSeries].forEach(s => {
      if (s) this.chart.removeSeries(s);
    });
    this.histSeries = this.zeroSeries = null;
    this.closes = [];
    this.highs = [];
    this.lows = [];
    this.prevVal = 0;
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
      timeScale: {
        borderVisible: false,
        timeVisible:   true,
        secondsVisible:true,
        shiftVisibleRangeOnNewBar: true,
      },
    } as any;
  }
}