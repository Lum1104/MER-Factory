import { Link, useLocation } from 'react-router-dom';
import './Footer.css';

function Footer() {
  const { pathname } = useLocation();
  const inDocs = pathname.startsWith('/docs') || pathname.startsWith('/zh/docs');

  return (
    <footer className={`footer${inDocs ? ' footer--docs' : ''}`}>
      <div className="footer-container">
        <div className="footer-grid">
          <div className="footer-brand">
            <img
              src={`${import.meta.env.BASE_URL}logo.svg`}
              alt="MER-Factory"
            />
            <p>
              Automated Multimodal Emotion Recognition Dataset Creation
            </p>
            <p className="footer-author">Created by MER-Factory Team</p>
          </div>

          <div className="footer-column">
            <h4>Resources</h4>
            <ul>
              <li><Link to="/docs/">Documentation</Link></li>
              <li><Link to="/docs/getting-started">Getting Started</Link></li>
              <li><Link to="/docs/api-reference">API Reference</Link></li>
              <li><Link to="/docs/examples">Examples</Link></li>
            </ul>
          </div>

          <div className="footer-column">
            <h4>Community</h4>
            <ul>
              <li>
                <a
                  href="https://github.com/Lum1104/MER-Factory"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  GitHub
                </a>
              </li>
              <li>
                <a
                  href="https://github.com/Lum1104/MER-Factory/issues"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  Issues
                </a>
              </li>
              <li>
                <a
                  href="https://github.com/Lum1104/MER-Factory/discussions"
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  Discussions
                </a>
              </li>
            </ul>
          </div>
        </div>

        <div className="footer-citation">
          <details>
            <summary>Cite MER-Factory</summary>
            <pre><code>{`@article{lian2025mer,
  title={MER-Factory: Towards Building An Automated MER Dataset Factory},
  author={Lian, Zheng and others},
  journal={arXiv preprint arXiv:2503.03275},
  year={2025}
}`}</code></pre>
          </details>
        </div>

        <div className="footer-bottom">
          <p>MIT License. 2025 MER-Factory.</p>
          <p>Built with React + Vite</p>
        </div>
      </div>
    </footer>
  );
}

export default Footer;
