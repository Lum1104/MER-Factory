import { useState } from 'react';
import { motion } from 'framer-motion';
import './QuickStart.css';

const codeLines = [
  { type: 'comment', text: '# Clone the repository' },
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
  { type: 'empty' },
  { type: 'comment', text: '# Install dependencies' },
  {
    tokens: [
      { type: 'command', text: 'pip' },
      { type: 'plain', text: ' install ' },
      { type: 'flag', text: '-r' },
      { type: 'plain', text: ' ' },
      { type: 'string', text: 'requirements.txt' },
    ],
  },
  { type: 'empty' },
  { type: 'comment', text: '# Run the pipeline' },
  {
    tokens: [
      { type: 'command', text: 'python' },
      { type: 'plain', text: ' ' },
      { type: 'string', text: 'main.py' },
      { type: 'plain', text: ' ' },
      { type: 'flag', text: '--pipeline' },
      { type: 'plain', text: ' mer ' },
      { type: 'flag', text: '--input' },
      { type: 'plain', text: ' ' },
      { type: 'string', text: './data/sample.mp4' },
    ],
  },
];

const codeText = `git clone https://github.com/Lum1104/MER-Factory.git
cd MER-Factory
pip install -r requirements.txt
python main.py --pipeline mer --input ./data/sample.mp4`;

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

        <a className="docs-link" href="/MER-Factory/docs/">
          Read the full documentation &rarr;
        </a>
      </div>
    </motion.section>
  );
}
