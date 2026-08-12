import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // 개발 시 FastAPI(8000)로 프록시 — CORS 설정 불필요
    proxy: { '/api': 'http://localhost:8000' },
  },
})
