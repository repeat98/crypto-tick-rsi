// vite.config.ts
import { defineConfig } from 'vite';
import tsconfigPaths from 'vite-plugin-tsconfig-paths';

export default defineConfig({
  // point Vite at your UI folder
  root: 'src/ui',
  base: '/crypto-tick-rsi/',

  // so imports like `import { ChartManager } from '@/charts/chartManager'` still work
  plugins: [tsconfigPaths()],

  // where to put the built files
  build: {
    outDir: '../../docs',
    emptyOutDir: true,
  },
});