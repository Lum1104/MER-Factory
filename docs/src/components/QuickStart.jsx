import { useState } from 'react';
import { motion } from 'framer-motion';
import { Link } from 'react-router-dom';
import './QuickStart.css';

const codeLines = [
  { type: 'comment', text: '# Install prerequisites' },
  {
    tokens: [
      { type: 'command', text: 'brew' },
      { type: 'plain', text: ' install ' },
      { type: 'string', text: 'ffmpeg' },
    ],
  },
  { type: 'empty' },
  { type: 'comment', text: '# Clone and set up MER-Factory' },
  {
    tokens: [
      { type: 'command', text: 'git' },
      { type: 'plain', text: ' clone ' },
      { type: 'string', text: 'https://github.com/Lum1104/MER-Factory.git' },
    ],
  },
  {
    tokens: [
      { type: 'command', text: 'cd' },
      { type: 'plain', text: ' MER-Factory' },
    ],
  },
  {
    tokens: [
      { type: 'command', text: 'conda' },
      { type: 'plain', text: ' create ' },
      { type: 'flag', text: '-n' },
      { type: 'plain', text: ' mer-factory python=3.12' },
    ],
  },
  {
    tokens: [
      { type: 'command', text: 'conda' },
      { type: 'plain', text: ' activate ' },
      { type: 'string', text: 'mer-factory' },
    ],
  },
  {
    tokens: [
      { type: 'command', text: 'pip' },
      { type: 'plain', text: ' install ' },
      { type: 'flag', text: '-r' },
      { type: 'plain', text: ' requirements.txt' },
    ],
  },
  { type: 'empty' },
  { type: 'comment', text: '# Configure environment' },
  {
    tokens: [
      { type: 'command', text: 'cp' },
      { type: 'plain', text: ' .env.example ' },
      { type: 'string', text: '.env' },
    ],
  },
  { type: 'comment', text: '# Edit .env: set OPENFACE_EXECUTABLE, API keys' },
  { type: 'empty' },
  { type: 'comment', text: '# Run the pipeline' },
  {
    tokens: [
      { type: 'command', text: 'python' },
      { type: 'plain', text: ' main.py ' },
      { type: 'string', text: 'input/' },
      { type: 'plain', text: ' ' },
      { type: 'string', text: 'output/' },
      { type: 'plain', text: ' ' },
      { type: 'flag', text: '--type' },
      { type: 'plain', text: ' MER ' },
      { type: 'flag', text: '--silent' },
    ],
  },
];

const codeText = `# Install prerequisites
brew install ffmpeg

# Clone and set up MER-Factory
git clone https://github.com/Lum1104/MER-Factory.git
cd MER-Factory
conda create -n mer-factory python=3.12
conda activate mer-factory
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env: set OPENFACE_EXECUTABLE, API keys

# Run the pipeline
python main.py input/ output/ --type MER --silent`;

function renderLine(line, index) {
  if (line.type === 'empty') {
    return <span key={index}>{'\n'}</span>;
  }
  if (line.type === 'comment') {
    return (
      <span key={index}>
        <span className="token-comment">{line.text}</span>
        {'\n'}
      </span>
    );
  }
  return (
    <span key={index}>
      {line.tokens.map((token, i) => {
        const classMap = {
          command: 'token-command',
          flag: 'token-flag',
          string: 'token-string',
          plain: '',
        };
        const className = classMap[token.type] || '';
        return className ? (
          <span key={i} className={className}>
            {token.text}
          </span>
        ) : (
          <span key={i}>{token.text}</span>
        );
      })}
      {'\n'}
    </span>
  );
}

export default function QuickStart() {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(codeText);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard API not available
    }
  };

  return (
    <motion.section
      id="quickstart"
      className="quickstart-section"
      initial={{ opacity: 0, y: 40 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-100px' }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
    >
      <div className="quickstart-container">
        <h2 className="quickstart-title">Get Started in Seconds</h2>

        <div className="code-block">
          <div className="code-header">
            <div className="code-dots">
              <span className="code-dot code-dot--red" />
              <span className="code-dot code-dot--yellow" />
              <span className="code-dot code-dot--green" />
            </div>
            <span className="code-title">Terminal</span>
            <button
              className={`copy-btn${copied ? ' copy-btn--copied' : ''}`}
              onClick={handleCopy}
            >
              {copied ? (
                <>
                  <svg
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                  Copied
                </>
              ) : (
                <>
                  <svg
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                  </svg>
                  Copy
                </>
              )}
            </button>
          </div>

          <div className="code-content">
            <pre>
              <code>{codeLines.map(renderLine)}</code>
            </pre>
          </div>
        </div>

        <Link className="docs-link" to="/docs/">
          Read the full documentation &rarr;
        </Link>
      </div>
    </motion.section>
  );
}
