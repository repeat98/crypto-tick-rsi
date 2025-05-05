// src/ui/main.ts
import { BinanceConnector, RawTick } from '../api/binanceConnector';
import { initMainChart } from './mainChart';
import { RSIChart } from './rsiChart';
import { throttleTime, map, pairwise } from 'rxjs/operators';
import { Subscription, animationFrameScheduler, fromEvent } from 'rxjs';

const symbolSelect = document.getElementById('symbol') as HTMLSelectElement;
const chartDiv      = document.getElementById('chart')!;
const rsiDiv        = document.getElementById('rsi-chart')!;
const pingDiv       = document.getElementById('ping')!;
const themeToggle   = document.getElementById('theme-toggle') as HTMLButtonElement;
const moonIcon = `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/></svg>`;
const sunIcon  = `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>`;

const { chartManager } = initMainChart(chartDiv);
const rsiChart = new RSIChart(rsiDiv, { period: 14 });

let connector: BinanceConnector | null = null;
let liveSub:    Subscription | null = null;
let pingSub:    Subscription | null = null;

function normalize(tick: RawTick) {
  const timeSec = Math.floor(tick.E / 1000);
  if (tick.p) return { time: timeSec, value: parseFloat(tick.p) };
  const mid = (parseFloat(tick.b!) + parseFloat(tick.a!)) / 2;
  return { time: timeSec, value: mid };
}

async function loadSymbol(symbol: string) {
  symbolSelect.disabled = true;
  try {
    chartManager.clear();
    rsiChart.clear();
    liveSub?.unsubscribe();
    pingSub?.unsubscribe();
    connector?.close();

    connector = new BinanceConnector(symbol);

    // Ping indicator on every raw tick
    pingSub = connector.ticks$.subscribe((tick: RawTick) => {
      const ms = Date.now() - tick.E;
      pingDiv.textContent = `Ping: ${ms.toFixed(0)} ms`;
    });

    // 1h history
    const rawHist = await connector.fetchHistorical(60 * 60 * 1000);
    const normed  = rawHist.map(normalize);
    normed.forEach(point => {
      chartManager.updatePrice(point);
      rsiChart.update(point);
    });

    // Live ticks ~60FPS for chart & RSI
    liveSub = connector.ticks$
      .pipe(
        throttleTime(16, animationFrameScheduler),
        map(normalize),
        pairwise()
      )
      .subscribe(([, curr]) => {
        chartManager.updatePrice(curr);
        rsiChart.update(curr);
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
  // update ping background too
  pingDiv.style.background  = getComputedStyle(document.body).getPropertyValue('--bg-color').trim();
  pingDiv.style.color       = getComputedStyle(document.body).getPropertyValue('--text-color').trim();
  pingDiv.style.borderColor = getComputedStyle(document.body).getPropertyValue('--grid-line').trim();
  // swap icon
  themeToggle.innerHTML = mode === 'dark' ? sunIcon : moonIcon;
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
  const newMode: 'light' | 'dark' = document.body.classList.contains('dark') ? 'light' : 'dark';
  applyTheme(newMode);
});