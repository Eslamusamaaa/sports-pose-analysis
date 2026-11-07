import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './App'
import SwimApp from './SwimApp'
import './styles.css'

// Two separate pages without a router dependency: Vite's SPA fallback serves
// index.html for /swim, and the path picks which app to mount.
const isSwim = window.location.pathname.replace(/\/+$/, '') === '/swim'

createRoot(document.getElementById('root')!).render(
  <StrictMode>{isSwim ? <SwimApp /> : <App />}</StrictMode>,
)
