import { motion } from 'framer-motion';
import './PipelineAnimation.css';

export default function PipelineAnimation() {
  return (
    <motion.section
      id="pipeline"
      className="pipeline-section"
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.8, ease: 'easeOut' }}
      viewport={{ once: true, amount: 0.3 }}
    >
      <h2 className="pipeline-title">From Raw Data to Emotion Labels</h2>

      <div className="pipeline-container">
        <svg
          className="pipeline-svg"
          viewBox="0 0 900 310"
          width="100%"
          xmlns="http://www.w3.org/2000/svg"
        >
          {/* ── Defs: glow filter + gradients ── */}
          <defs>
            <filter id="glow">
              <feGaussianBlur stdDeviation="4" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>

            <linearGradient id="coralGrad" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#FF6B6B" />
              <stop offset="100%" stopColor="#ff8787" />
            </linearGradient>

            <linearGradient id="goldGrad" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#FFE66D" />
              <stop offset="100%" stopColor="#fff0a0" />
            </linearGradient>

            <linearGradient id="tealGrad" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#4ECDC4" />
              <stop offset="100%" stopColor="#6eddd6" />
            </linearGradient>

            <linearGradient id="purpleGrad" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#6C5CE7" />
              <stop offset="100%" stopColor="#8a7eef" />
            </linearGradient>

            <linearGradient id="outputBorderGrad" x1="0" y1="0" x2="1" y2="1">
              <stop offset="0%" stopColor="#FF6B6B" />
              <stop offset="100%" stopColor="#4ECDC4" />
            </linearGradient>
          </defs>

          {/* ── Connecting paths (static, low opacity) ── */}
          <path
            id="pathVideoLLM"
            d="M 140,75 C 280,75 310,170 385,170"
            stroke="#FF6B6B"
            strokeWidth="2"
            fill="none"
            opacity="0.3"
          />
          <path
            id="pathAudioLLM"
            d="M 140,165 C 250,165 300,170 385,170"
            stroke="#FFE66D"
            strokeWidth="2"
            fill="none"
            opacity="0.3"
          />
          <path
            id="pathFaceLLM"
            d="M 140,255 C 280,255 310,170 385,170"
            stroke="#4ECDC4"
            strokeWidth="2"
            fill="none"
            opacity="0.3"
          />
          <path
            id="pathLLMOutput"
            d="M 515,170 C 600,170 650,170 720,170"
            stroke="#6C5CE7"
            strokeWidth="2"
            fill="none"
            opacity="0.3"
          />

          {/* ── Animated flow dots ── */}
          {/* Video → LLM (coral) */}
          <circle r="4" fill="#FF6B6B" filter="url(#glow)">
            <animateMotion dur="2.5s" begin="0s" repeatCount="indefinite" fill="freeze">
              <mpath href="#pathVideoLLM" />
            </animateMotion>
          </circle>
          <circle r="4" fill="#FF6B6B" filter="url(#glow)">
            <animateMotion dur="2.5s" begin="1.25s" repeatCount="indefinite" fill="freeze">
              <mpath href="#pathVideoLLM" />
            </animateMotion>
          </circle>

          {/* Audio → LLM (gold) */}
          <circle r="4" fill="#FFE66D" filter="url(#glow)">
            <animateMotion dur="3s" begin="0s" repeatCount="indefinite" fill="freeze">
              <mpath href="#pathAudioLLM" />
            </animateMotion>
          </circle>
          <circle r="4" fill="#FFE66D" filter="url(#glow)">
            <animateMotion dur="3s" begin="1.5s" repeatCount="indefinite" fill="freeze">
              <mpath href="#pathAudioLLM" />
            </animateMotion>
          </circle>

          {/* Face → LLM (teal) */}
          <circle r="4" fill="#4ECDC4" filter="url(#glow)">
            <animateMotion dur="3.5s" begin="0s" repeatCount="indefinite" fill="freeze">
              <mpath href="#pathFaceLLM" />
            </animateMotion>
          </circle>
          <circle r="4" fill="#4ECDC4" filter="url(#glow)">
            <animateMotion dur="3.5s" begin="1.75s" repeatCount="indefinite" fill="freeze">
              <mpath href="#pathFaceLLM" />
            </animateMotion>
          </circle>

          {/* LLM → Output (purple) */}
          <circle r="4" fill="#6C5CE7" filter="url(#glow)">
            <animateMotion dur="2.8s" begin="0s" repeatCount="indefinite" fill="freeze">
              <mpath href="#pathLLMOutput" />
            </animateMotion>
          </circle>
          <circle r="4" fill="#6C5CE7" filter="url(#glow)">
            <animateMotion dur="2.8s" begin="1.4s" repeatCount="indefinite" fill="freeze">
              <mpath href="#pathLLMOutput" />
            </animateMotion>
          </circle>

          {/* ── Input nodes ── */}

          {/* Video node */}
          <rect x="20" y="50" width="120" height="50" rx="12" fill="url(#coralGrad)" />
          <polygon points="68,67 68,87 85,77" fill="#fff" />
          <text
            x="80"
            y="120"
            textAnchor="middle"
            fontFamily="'Outfit', sans-serif"
            fontSize="13"
            fill="#9a8fb5"
          >
            Video
          </text>

          {/* Audio node */}
          <rect x="20" y="140" width="120" height="50" rx="12" fill="url(#goldGrad)" />
          <path
            d="M 50,165 Q 57,155 64,165 Q 71,175 78,165 Q 85,155 92,165 Q 99,175 106,165"
            stroke="#fff"
            strokeWidth="2.5"
            fill="none"
            strokeLinecap="round"
          />
          <text
            x="80"
            y="210"
            textAnchor="middle"
            fontFamily="'Outfit', sans-serif"
            fontSize="13"
            fill="#9a8fb5"
          >
            Audio
          </text>

          {/* Face node */}
          <rect x="20" y="230" width="120" height="50" rx="12" fill="url(#tealGrad)" />
          <circle cx="80" cy="251" r="12" stroke="#fff" strokeWidth="2" fill="none" />
          <circle cx="75" cy="248" r="1.5" fill="#fff" />
          <circle cx="85" cy="248" r="1.5" fill="#fff" />
          <path
            d="M 74,256 Q 80,261 86,256"
            stroke="#fff"
            strokeWidth="1.5"
            fill="none"
            strokeLinecap="round"
          />
          <text
            x="80"
            y="300"
            textAnchor="middle"
            fontFamily="'Outfit', sans-serif"
            fontSize="13"
            fill="#9a8fb5"
          >
            Face
          </text>

          {/* ── Central LLM node ── */}
          <rect x="385" y="140" width="130" height="60" rx="16" fill="url(#purpleGrad)" />
          <text
            x="450"
            y="176"
            textAnchor="middle"
            fontFamily="'Outfit', sans-serif"
            fontSize="18"
            fontWeight="700"
            fill="#fff"
          >
            LLM
          </text>

          {/* ── Output node ── */}
          <rect
            x="720"
            y="145"
            width="140"
            height="50"
            rx="14"
            fill="none"
            stroke="url(#outputBorderGrad)"
            strokeWidth="2"
          />
          <text
            x="790"
            y="175"
            textAnchor="middle"
            fontFamily="'Outfit', sans-serif"
            fontSize="14"
            fontWeight="600"
            fill="#e8e4f0"
          >
            MERR Dataset
          </text>
        </svg>
      </div>
    </motion.section>
  );
}
