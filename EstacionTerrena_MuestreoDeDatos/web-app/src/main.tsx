import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './lib/consoleFilter'

if (localStorage.getItem('lb135-theme') === 'sol') document.documentElement.dataset.theme = 'sol'
import './index.css'
import App from './App.tsx'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
