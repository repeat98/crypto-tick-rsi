import { BinanceConnector, RawTick } from '../api/binanceConnector';
import { initMainChart } from './mainChart';
import { RSIChart } from './rsiChart';
import { SqueezeMomentumChart } from './smoChart';
import { throttleTime, map, pairwise } from 'rxjs/operators';
import { Subscription, animationFrameScheduler, fromEvent } from 'rxjs';

const STORAGE_SYMBOL_KEY = 'selectedSymbol';
const STORAGE_MODE_KEY   = 'themeMode';

const symbolSelect = document.getElementById('symbol') as HTMLSelectElement;
const chartDiv      = document.getElementById('chart')!;
const rsiDiv        = document.getElementById('rsi-chart')!;
const smoDiv        = document.getElementById('smo-chart')!;
const pingDiv       = document.getElementById('ping')!;
const themeToggle   = document.getElementById('theme-toggle') as HTMLButtonElement;
const moonIcon = `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1111.21 3 7 7 0 0021 12.79z"/></svg>`;
const sunIcon  = `<svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>`;

const { chartManager } = initMainChart(chartDiv);
const rsiChart = new RSIChart(rsiDiv, { period: 14 });
const smoChart = new SqueezeMomentumChart(smoDiv, { period: 20 });

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
  localStorage.setItem(STORAGE_SYMBOL_KEY, symbol);
  try {
    chartManager.clear();
    rsiChart.clear();
    smoChart.clear();
    liveSub?.unsubscribe();
    pingSub?.unsubscribe();
    connector?.close();

    connector = new BinanceConnector(symbol);

    // Ping indicator
    pingSub = connector.ticks$.subscribe((tick: RawTick) => {
      const ms = Date.now() - tick.E;
      pingDiv.textContent = `Ping: ${ms.toFixed(0)} ms`;
    });

    // 1h history
    const rawHist = await connector.fetchHistorical(60 * 60 * 1000);
    const normed  = rawHist.map(normalize);
    // only keep the last 7 minutes of history (420 seconds)
    const nowSec = Math.floor(Date.now() / 1000);
    const sevenMinAgo = nowSec - 7 * 60;
    const recentNormed = normed.filter(point => point.time >= sevenMinAgo);
    recentNormed.forEach(point => {
      chartManager.updatePrice(point);
      rsiChart.update(point);
      smoChart.update(point);
    });

    // zoom all charts to last 7 minutes
    chartManager.zoomToLast(7 * 60);
    rsiChart.zoomToLast(7 * 60);
    smoChart.zoomToLast(7 * 60);

    // Live ticks ~60FPS
    liveSub = connector.ticks$
      .pipe(
        throttleTime(16, animationFrameScheduler),
        map(normalize),
        pairwise()
      )
      .subscribe(([, curr]) => {
        chartManager.updatePrice(curr);
        rsiChart.update(curr);
        smoChart.update(curr);
        // keep time scale pinned to real-time after each tick on all charts
        chartManager.zoomToLast(7 * 60);
        rsiChart.zoomToLast(7 * 60);
        smoChart.zoomToLast(7 * 60);
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
  smoChart.applyTheme();
  pingDiv.style.background  = getComputedStyle(document.body).getPropertyValue('--bg-color').trim();
  pingDiv.style.color       = getComputedStyle(document.body).getPropertyValue('--text-color').trim();
  pingDiv.style.borderColor = getComputedStyle(document.body).getPropertyValue('--grid-line').trim();
  themeToggle.innerHTML = mode === 'dark' ? sunIcon : moonIcon;
  localStorage.setItem(STORAGE_MODE_KEY, mode);
}

fromEvent(symbolSelect, 'change')
  .pipe(map((e: Event) => (e.target as HTMLSelectElement).value), throttleTime(300))
  .subscribe(loadSymbol);

const savedSymbol = localStorage.getItem(STORAGE_SYMBOL_KEY);
if (savedSymbol) symbolSelect.value = savedSymbol;
loadSymbol(symbolSelect.value);

const savedMode = localStorage.getItem(STORAGE_MODE_KEY) as 'light' | 'dark' | null;
applyTheme(savedMode === 'light' ? 'light' : 'dark');

themeToggle.addEventListener('click', () => {
  const newMode: 'light' | 'dark' = document.body.classList.contains('dark') ? 'light' : 'dark';
  applyTheme(newMode);
});