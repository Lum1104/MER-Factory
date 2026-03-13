import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { resolve } from 'path'
import { existsSync } from 'fs'

function servePublicDocs() {
  const base = '/MER-Factory/'
  return {
    name: 'serve-public-docs',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const docsPrefix = base + 'docs/'
        if (req.url?.startsWith(docsPrefix)) {
          const relative = req.url.slice(docsPrefix.length) || 'index.html'
          const filePath = resolve(__dirname, 'public/docs', relative)
          if (existsSync(filePath)) {
            req.url = base + 'docs/' + relative
          } else if (!relative.endsWith('/') && existsSync(filePath + '/index.html')) {
            req.url = base + 'docs/' + relative + '/index.html'
          }
        }
        next()
      })
    },
  }
}

export default defineConfig({
  plugins: [servePublicDocs(), react()],
  base: '/MER-Factory/',
  build: {
    outDir: 'dist',
  },
})
