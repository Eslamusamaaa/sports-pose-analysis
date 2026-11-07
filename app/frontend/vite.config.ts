import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// The backend is proxied so the browser only ever talks to the Vite origin,
// which keeps uploads and video streaming free of CORS preflight issues.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Native fs events drop consecutive edits on this Windows setup, so Vite
    // kept serving a stale module (e.g. JSX using a component whose import was
    // added in a later edit -> "X is not defined" and a blank page). Polling
    // is slower to notice a change but never misses one.
    watch: { usePolling: true, interval: 300 },
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
        // REQUIRED for the live view. Without ws:true Vite proxies ordinary
        // HTTP but silently refuses the WebSocket upgrade for /api/live/ws, so
        // the socket never opens, no snapshot ever arrives, and the UI sits on
        // "Not started" forever - even though the backend is happily running
        // and processing frames.
        ws: true,
      },
    },
  },
})
