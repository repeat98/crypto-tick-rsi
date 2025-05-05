// src/ui/main.ts
import { BinanceConnector, RawTick } from '../api/binanceConnector';
import { initMainChart } from './mainChart';
import { RSIChart } from './rsiChart';
import { throttleTime, map, pairwise } from 'rxjs/operators';
import { Subscription, animationFrameScheduler, fromEvent } from 'rxjs';

const symbolSelect = document.getElementById('symbol') as HTMLSelectElement;
const chartDiv      = document.getElementById('chart')!;
const rsiDiv        = document.getElementById('rsi-chart')!;
const themeToggle   = document.getElementById('theme-toggle') as HTMLButtonElement;

const { chartManager } = initMainChart(chartDiv);
const rsiChart = new RSIChart(rsiDiv, 14);  // 14-period RSI baked into the chart

let connector: BinanceConnector | null = null;
let liveSub:    Subscription | null = null;

// normalize a RawTick into { time: seconds, value: price }
function normalize(tick: RawTick) {
  const timeSec = Math.floor(tick.E / 1000);
  if (tick.p) {
    return { time: timeSec, value: parseFloat(tick.p) };
  }
  const mid = (parseFloat(tick.b!) + parseFloat(tick.a!)) / 2;
  return { time: timeSec, value: mid };
}

async function loadSymbol(symbol: string) {
  symbolSelect.disabled = true;
  try {
    chartManager.clear();
    rsiChart.clear();
    liveSub?.unsubscribe();
    connector?.close();

    connector = new BinanceConnector(symbol);

    // 1h history
    const rawHist = await connector.fetchHistorical(60 * 60 * 1000);
    const normed  = rawHist.map(normalize);
    normed.forEach(point => {
      chartManager.updatePrice(point);
      rsiChart.update(point);
    });

    // live ticks ~60FPS
    liveSub = connector.ticks$
      .pipe(
        throttleTime(16, animationFrameScheduler),
        map(normalize),
        pairwise()
      )
      .subscribe(([prev, curr]) => {
        chartManager.updatePrice(curr);
        rsiChart.update(curr);
      });
  } finally {
    symbolSelect.disabled = false;
  }
}

// theme toggling (unchanged)
function applyTheme(mode: 'light' | 'dark') {
  document.body.classList.toggle('dark', mode === 'dark');
  document.body.classList.toggle('light', mode === 'light');
  chartManager.applyTheme();
  rsiChart.applyTheme();
  themeToggle.textContent = mode === 'dark'
    ? 'Switch to Light Mode'
    : 'Switch to Dark Mode';
}

fromEvent(symbolSelect, 'change')
  .pipe(
    map((e: Event) => (e.target as HTMLSelectElement).value),
    throttleTime(300)
  )
  .subscribe(loadSymbol);

loadSymbol(symbolSelect.value);
applyTheme('dark');
themeToggle.addEventListener('click', () => {
  const newMode = document.body.classList.contains('dark') ? 'light' : 'dark';
  applyTheme(newMode);
});