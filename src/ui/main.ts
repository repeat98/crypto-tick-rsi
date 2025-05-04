// src/ui/main.ts
import { RSI } from '../charts/indicators';
import { RSIChart } from './rsiChart';
import { initMainChart } from './mainChart';
import { BinanceConnector } from '../api/binanceConnector';
import { normalizeTick } from '../charts/dataNormalizer';
import { throttleTime, map, pairwise } from 'rxjs/operators';
import { Subscription, animationFrameScheduler, fromEvent } from 'rxjs';

const symbolSelect = document.getElementById('symbol') as HTMLSelectElement;
const chartDiv      = document.getElementById('chart')!;
const rsiDiv        = document.getElementById('rsi-chart')!;
const themeToggle   = document.getElementById('theme-toggle') as HTMLButtonElement;

const { chartManager } = initMainChart(chartDiv);
const rsiChart = new RSIChart(rsiDiv);

let connector: BinanceConnector | null = null;
let liveSub:    Subscription | null = null;
let rsi:        RSI;

// loadSymbol now disables the selector until it's fully initialized
async function loadSymbol(symbol: string) {
  symbolSelect.disabled = true;
  try {
    // 1) clear charts and old socket
    chartManager.clear();
    rsiChart.clear();
    if (liveSub) {
      liveSub.unsubscribe();
      liveSub = null;
    }
    if (connector) {
      connector.close();  // new method to fully tear down WS
      connector = null;
    }

    // 2) reset RSI calculator
    rsi = new RSI(14);

    // 3) create new connector
    connector = new BinanceConnector(symbol);

    // 4) fetch & draw 1h history
    const rawHist = await connector.fetchHistorical(60 * 60 * 1000);
    const normed  = rawHist.map(normalizeTick);
    normed.forEach((curr, i, arr) => {
      if (i === 0) return;
      const prev = arr[i - 1];
      chartManager.updatePrice({ time: curr.time, value: curr.value });
      const r = rsi.update(curr, prev);
      rsiChart.update(r);
    });

    // 5) subscribe to live ticks at ~60 FPS
    liveSub = connector.ticks$
      .pipe(
        throttleTime(16, animationFrameScheduler),
        map(normalizeTick),
        pairwise()
      )
      .subscribe(([prev, curr]) => {
        chartManager.updatePrice({ time: curr.time, value: curr.value });
        const r = rsi.update(curr, prev);
        rsiChart.update(r);
      });
  } finally {
    symbolSelect.disabled = false;
  }
}

function applyTheme(mode: 'light' | 'dark') {
  document.body.classList.toggle('dark', mode === 'dark');
  document.body.classList.toggle('light', mode === 'light');
  chartManager.applyTheme();
  rsiChart.applyTheme();
  themeToggle.textContent = mode === 'dark' ? 'Switch to Light Mode' : 'Switch to Dark Mode';
}

// Debounce symbol changes by 300ms to prevent glitches
fromEvent(symbolSelect, 'change')
  .pipe(
    map((e: Event) => (e.target as HTMLSelectElement).value),
    throttleTime(300)
  )
  .subscribe(value => {
    loadSymbol(value);
  });

// initialize
loadSymbol(symbolSelect.value);

// theme toggle setup
themeToggle.addEventListener('click', () => {
  const newMode = document.body.classList.contains('dark') ? 'light' : 'dark';
  applyTheme(newMode);
});
// start in dark mode
applyTheme('dark');