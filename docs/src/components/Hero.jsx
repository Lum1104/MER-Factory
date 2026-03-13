import { motion } from 'framer-motion';
import './Hero.css';

const containerVariants = {
  hidden: {},
  visible: {
    transition: {
      staggerChildren: 0.15,
    },
  },
};

const fadeInUp = {
  hidden: { opacity: 0, y: 30 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.6, ease: 'easeOut' },
  },
};

const meshBlobs = [
  { className: 'hero-mesh-blob hero-mesh-blob--coral', style: { top: '10%', left: '15%', width: '400px', height: '400px', background: 'var(--coral)' } },
  { className: 'hero-mesh-blob hero-mesh-blob--gold', style: { top: '60%', right: '10%', width: '350px', height: '350px', background: 'var(--gold)' } },
  { className: 'hero-mesh-blob hero-mesh-blob--teal', style: { bottom: '10%', left: '30%', width: '300px', height: '300px', background: 'var(--teal)' } },
  { className: 'hero-mesh-blob hero-mesh-blob--purple', style: { top: '30%', right: '25%', width: '380px', height: '380px', background: 'var(--purple)' } },
  { className: 'hero-mesh-blob hero-mesh-blob--mix', style: { top: '50%', left: '50%', width: '320px', height: '320px', background: 'linear-gradient(135deg, var(--coral), var(--purple))' } },
];

const floatingOrbs = [
  { style: { width: '100px', height: '100px', top: '15%', left: '10%', background: 'var(--coral)' }, delay: '0s' },
  { style: { width: '80px', height: '80px', top: '70%', right: '15%', background: 'var(--teal)' }, delay: '1s' },
  { style: { width: '120px', height: '120px', top: '40%', left: '75%', background: 'var(--purple)' }, delay: '2s' },
  { style: { width: '90px', height: '90px', bottom: '20%', left: '20%', background: 'var(--gold)' }, delay: '0.5s' },
  { style: { width: '150px', height: '150px', top: '25%', right: '5%', background: 'var(--coral)' }, delay: '1.5s' },
  { style: { width: '110px', height: '110px', bottom: '30%', right: '30%', background: 'var(--teal)' }, delay: '2.5s' },
  { style: { width: '85px', height: '85px', top: '5%', left: '55%', background: 'var(--gold)' }, delay: '3s' },
  { style: { width: '95px', height: '95px', bottom: '5%', right: '50%', background: 'var(--purple)' }, delay: '0.8s' },
];

export default function Hero() {
  return (
    <section className="hero">
      {/* Gradient mesh background */}
      {meshBlobs.map((blob, i) => (
        <div key={`blob-${i}`} className={blob.className} style={blob.style} />
      ))}

      {/* Floating orbs */}
      {floatingOrbs.map((orb, i) => (
        <div
          key={`orb-${i}`}
          className="hero-orb"
          style={{ ...orb.style, animationDelay: orb.delay }}
        />
      ))}

      {/* Content */}
      <motion.div
        className="hero-content"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        <motion.h1 className="hero-title" variants={fadeInUp}>
          MER-Factory
        </motion.h1>

        <motion.p className="hero-tagline" variants={fadeInUp}>
          Your Automated Factory for Multimodal Emotion Recognition Datasets
        </motion.p>

        <motion.div className="hero-buttons" variants={fadeInUp}>
          <a href="/MER-Factory/docs/" className="hero-btn-primary">
            Get Started
          </a>
          <a
            href="https://github.com/Lum1104/MER-Factory"
            target="_blank"
            rel="noopener noreferrer"
            className="hero-btn-secondary"
          >
            View on GitHub
          </a>
        </motion.div>

        <motion.div className="hero-image-wrapper" variants={fadeInUp}>
          <img
            src={`${import.meta.env.BASE_URL}mer-factory.jpeg`}
            alt="MER-Factory overview"
            className="hero-image"
          />
        </motion.div>
      </motion.div>
    </section>
  );
}
