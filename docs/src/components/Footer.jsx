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
            <pre><code>{`@software{Lin_MER-Factory_2025,
  author = {Lin, Yuxiang and Zheng, Shunchao},
  doi = {10.5281/zenodo.15847351},
  license = {MIT},
  month = {7},
  title = {{MER-Factory}},
  url = {https://github.com/Lum1104/MER-Factory},
  version = {0.1.0},
  year = {2025}
}

@inproceedings{NEURIPS2024_c7f43ada,
  author = {Cheng, Zebang and Cheng, Zhi-Qi and He, Jun-Yan and Wang, Kai and Lin, Yuxiang and Lian, Zheng and Peng, Xiaojiang and Hauptmann, Alexander},
  booktitle = {Advances in Neural Information Processing Systems},
  editor = {A. Globerson and L. Mackey and D. Belgrave and A. Fan and U. Paquet and J. Tomczak and C. Zhang},
  pages = {110805--110853},
  publisher = {Curran Associates, Inc.},
  title = {Emotion-LLaMA: Multimodal Emotion Recognition and Reasoning with Instruction Tuning},
  url = {https://proceedings.neurips.cc/paper_files/paper/2024/file/c7f43ada17acc234f568dc66da527418-Paper-Conference.pdf},
  volume = {37},
  year = {2024}
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
