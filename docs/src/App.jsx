import { Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import Hero from './components/Hero'
import PipelineAnimation from './components/PipelineAnimation'
import Features from './components/Features'
import QuickStart from './components/QuickStart'
import Footer from './components/Footer'
import DocsLayout from './components/docs/DocsLayout'
import './App.css'

const enModules = import.meta.glob('./content/en/*.mdx', { eager: true })
const zhModules = import.meta.glob('./content/zh/*.mdx', { eager: true })

function getDocPages(modules, prefix) {
  const pages = {}
  for (const path in modules) {
    const slug = path.split('/').pop().replace('.mdx', '')
    pages[slug] = modules[path]
  }
  return pages
}

const enPages = getDocPages(enModules, 'en')
const zhPages = getDocPages(zhModules, 'zh')

function Homepage() {
  return (
    <>
      <Hero />
      <PipelineAnimation />
      <Features />
      <QuickStart />
    </>
  )
}

function App() {
  return (
    <div className="app">
      <Navbar />
      <main>
        <Routes>
          <Route path="/" element={<Homepage />} />
          <Route path="/docs/*" element={<DocsLayout pages={enPages} lang="en" />} />
          <Route path="/zh/docs/*" element={<DocsLayout pages={zhPages} lang="zh" />} />
        </Routes>
      </main>
      <Footer />
    </div>
  )
}

export default App
