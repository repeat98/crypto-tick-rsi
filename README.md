# crypto-tick-rsi

A lightweight, real-time cryptocurrency price and RSI (Relative Strength Index) charting application built with TypeScript, Vite, RxJS, Binance WebSockets, and [lightweight-charts](https://github.com/tradingview/lightweight-charts).

## Features

* **Real-time price updates** via Binance WebSocket API
* **1-hour historical data** fetching to seed the chart on load
* **RSI calculation** with Wilder’s smoothing (14-period)
* **Interactive charts** powered by lightweight-charts
* **Theming**: dark and light modes
* **Symbol switching**: select from popular USDT pairs

## Getting Started

### Prerequisites

* Node.js (>=14.x)
* npm (>=6.x)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/repeat98/crypto-tick-rsi.git
   cd crypto-tick-rsi
   ```
2. Install dependencies:

   ```bash
   npm install
   ```

### Development

Run the development server:

```bash
npm run dev
```

Open your browser at `http://localhost:3000` (or the port indicated in the console).

### Production Build

To create an optimized production build:

```bash
npm run build
npm run preview
```

## Project Structure

```
.
├── docs/                 # Static site assets
├── src/                  # Source code
│   ├── api/              # Binance WebSocket connector and historical fetch
│   ├── charts/           # Data normalizers and indicators (EMA, VWAP, RSI)
│   └── ui/               # HTML & TypeScript UI entrypoints and chart initializers
├── README.md             # Project documentation
├── package.json          # npm scripts & dependencies
├── tsconfig.json         # TypeScript configuration
└── vite.config.ts        # Vite configuration
```

## Usage

* Select a trading pair from the dropdown to load 1-hour history and begin live updates.
* Click the **Switch to Light/Dark Mode** button to toggle themes.
* RSI chart appears below the price chart, with overbought (70) and oversold (30) levels marked.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Third-Party Licenses

* **lightweight-charts** (by TradingView, Inc.) is licensed under the Apache License, Version 2.0. A copy of the license is included in `licenses/lightweight-charts/LICENSE`.

  **Copyright 2023 TradingView, Inc.**

  Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with the License. You may obtain a copy of the License at

  ```
  http://www.apache.org/licenses/LICENSE-2.0
  ```

  Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND.
