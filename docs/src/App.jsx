import Navbar from './components/Navbar'
import Hero from './components/Hero'
import PipelineAnimation from './components/PipelineAnimation'
import Features from './components/Features'
import QuickStart from './components/QuickStart'
import Footer from './components/Footer'
import './App.css'

function App() {
  return (
    <div className="app">
      <Navbar />
      <main>
        <Hero />
        <PipelineAnimation />
        <Features />
        <QuickStart />
      </main>
      <Footer />
    </div>
  )
}

export default App
