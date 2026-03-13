import { motion } from 'framer-motion';
import './Features.css';

const pipelines = [
  {
    id: 'au',
    title: 'AU Pipeline',
    description:
      'Extract facial Action Units from images and generate emotion annotations via LLM reasoning.',
    color: 'coral',
    icon: (
      <svg width="40" height="40" viewBox="0 0 40 40" fill="none" stroke="var(--coral)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="20" cy="20" r="15" />
        <circle cx="14" cy="16" r="1.5" fill="var(--coral)" stroke="none" />
        <circle cx="26" cy="16" r="1.5" fill="var(--coral)" stroke="none" />
        <circle cx="14" cy="22" r="1" fill="var(--coral)" stroke="none" />
        <circle cx="20" cy="24" r="1" fill="var(--coral)" stroke="none" />
        <circle cx="26" cy="22" r="1" fill="var(--coral)" stroke="none" />
        <path d="M14 28 Q20 32 26 28" />
      </svg>
    ),
  },
  {
    id: 'audio',
    title: 'Audio Pipeline',
    description:
      'Process audio segments with emotion-aware speech analysis and LLM-based labeling.',
    color: 'gold',
    icon: (
      <svg width="40" height="40" viewBox="0 0 40 40" fill="none" stroke="var(--gold)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <line x1="4" y1="20" x2="4" y2="20" />
        <line x1="8" y1="14" x2="8" y2="26" />
        <line x1="12" y1="8" x2="12" y2="32" />
        <line x1="16" y1="12" x2="16" y2="28" />
        <line x1="20" y1="6" x2="20" y2="34" />
        <line x1="24" y1="10" x2="24" y2="30" />
        <line x1="28" y1="14" x2="28" y2="26" />
        <line x1="32" y1="8" x2="32" y2="32" />
        <line x1="36" y1="16" x2="36" y2="24" />
      </svg>
    ),
  },
  {
    id: 'video',
    title: 'Video Pipeline',
    description:
      'Analyze video clips combining visual and temporal cues for comprehensive emotion recognition.',
    color: 'teal',
    icon: (
      <svg width="40" height="40" viewBox="0 0 40 40" fill="none" stroke="var(--teal)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="4" y="8" width="24" height="24" rx="3" />
        <polygon points="28,14 36,10 36,30 28,26" />
        <polygon points="14,16 14,28 22,22" fill="var(--teal)" stroke="none" />
      </svg>
    ),
  },
  {
    id: 'image',
    title: 'Image Pipeline',
    description:
      'Process static images with facial expression and context analysis for emotion classification.',
    color: 'purple',
    icon: (
      <svg width="40" height="40" viewBox="0 0 40 40" fill="none" stroke="var(--purple)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="5" y="8" width="30" height="24" rx="3" />
        <circle cx="14" cy="17" r="3" />
        <polyline points="5,28 14,22 20,26 26,18 35,24" />
      </svg>
    ),
  },
  {
    id: 'mer',
    title: 'MER Pipeline',
    description:
      'Unified multimodal pipeline combining face, audio, and video for holistic emotion understanding.',
    color: 'mer',
    icon: (
      <svg width="40" height="40" viewBox="0 0 40 40" fill="none" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <defs>
          <linearGradient id="merGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="var(--coral)" />
            <stop offset="100%" stopColor="var(--purple)" />
          </linearGradient>
        </defs>
        <path d="M8 10 L20 20" stroke="url(#merGradient)" />
        <path d="M8 30 L20 20" stroke="url(#merGradient)" />
        <path d="M8 20 L20 20" stroke="url(#merGradient)" />
        <path d="M20 20 L32 20" stroke="url(#merGradient)" />
        <circle cx="8" cy="10" r="3" stroke="url(#merGradient)" />
        <circle cx="8" cy="20" r="3" stroke="url(#merGradient)" />
        <circle cx="8" cy="30" r="3" stroke="url(#merGradient)" />
        <circle cx="32" cy="20" r="4" stroke="url(#merGradient)" fill="none" />
        <circle cx="32" cy="20" r="1.5" fill="url(#merGradient)" stroke="none" />
      </svg>
    ),
  },
];

const values = [
  {
    title: 'Multiple LLM Backends',
    description:
      'Support for GPT-4, Claude, Gemini, and open-source models like LLaMA and Qwen.',
    icon: (
      <svg width="36" height="36" viewBox="0 0 36 36" fill="none" stroke="var(--teal)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="4" y="4" width="12" height="12" rx="2" />
        <rect x="20" y="4" width="12" height="12" rx="2" />
        <rect x="4" y="20" width="12" height="12" rx="2" />
        <rect x="20" y="20" width="12" height="12" rx="2" />
        <line x1="16" y1="10" x2="20" y2="10" />
        <line x1="10" y1="16" x2="10" y2="20" />
        <line x1="26" y1="16" x2="26" y2="20" />
        <line x1="16" y1="26" x2="20" y2="26" />
      </svg>
    ),
  },
  {
    title: 'Scientific Foundation',
    description:
      'Built on established emotion recognition research and validated annotation frameworks.',
    icon: (
      <svg width="36" height="36" viewBox="0 0 36 36" fill="none" stroke="var(--gold)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M14 4 V16 L8 28 H28 L22 16 V4" />
        <line x1="10" y1="4" x2="26" y2="4" />
        <circle cx="16" cy="22" r="1.5" fill="var(--gold)" stroke="none" />
        <circle cx="21" cy="24" r="1" fill="var(--gold)" stroke="none" />
        <circle cx="18" cy="18" r="1" fill="var(--gold)" stroke="none" />
      </svg>
    ),
  },
  {
    title: 'Quality Assurance',
    description:
      'Automated validation, consistency checks, and human-in-the-loop verification.',
    icon: (
      <svg width="36" height="36" viewBox="0 0 36 36" fill="none" stroke="var(--coral)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M18 4 L30 10 V18 C30 26 24 31 18 34 C12 31 6 26 6 18 V10 L18 4Z" />
        <polyline points="12,18 16,22 24,14" />
      </svg>
    ),
  },
  {
    title: 'Training Ready',
    description:
      'Export datasets in standard formats ready for model training and benchmarking.',
    icon: (
      <svg width="36" height="36" viewBox="0 0 36 36" fill="none" stroke="var(--purple)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="6" y="6" width="24" height="24" rx="3" />
        <line x1="6" y1="14" x2="30" y2="14" />
        <line x1="6" y1="22" x2="30" y2="22" />
        <line x1="14" y1="6" x2="14" y2="30" />
        <line x1="22" y1="6" x2="22" y2="30" />
        <polyline points="16,18 18,20 22,16" strokeWidth="1.5" />
      </svg>
    ),
  },
];

const containerVariants = {
  hidden: {},
  visible: {
    transition: {
      staggerChildren: 0.12,
    },
  },
};

const cardVariants = {
  hidden: { opacity: 0, y: 30 },
  visible: {
    opacity: 1,
    y: 0,
    transition: { duration: 0.5, ease: 'easeOut' },
  },
};

export default function Features() {
  return (
    <section id="features" className="features-section">
      <div className="features-container">
        <h2 className="features-title">Five Pipelines. One Factory.</h2>

        <motion.div
          className="features-grid"
          variants={containerVariants}
          initial="hidden"
          whileInView="visible"
          viewport={{ once: true, amount: 0.15 }}
        >
          {pipelines.map((pipeline) => (
            <motion.div
              key={pipeline.id}
              className={`feature-card feature-card--${pipeline.color}`}
              variants={cardVariants}
            >
              <div className="feature-card-icon">{pipeline.icon}</div>
              <h3 className="feature-card-title">{pipeline.title}</h3>
              <p className="feature-card-desc">{pipeline.description}</p>
            </motion.div>
          ))}
        </motion.div>

        <div className="values-section">
          <h3 className="values-title">Built for Researchers &amp; Engineers</h3>

          <motion.div
            className="values-grid"
            variants={containerVariants}
            initial="hidden"
            whileInView="visible"
            viewport={{ once: true, amount: 0.15 }}
          >
            {values.map((value) => (
              <motion.div
                key={value.title}
                className="value-card"
                variants={cardVariants}
              >
                <div className="value-card-icon">{value.icon}</div>
                <h4 className="value-card-title">{value.title}</h4>
                <p className="value-card-desc">{value.description}</p>
              </motion.div>
            ))}
          </motion.div>
        </div>
      </div>
    </section>
  );
}
