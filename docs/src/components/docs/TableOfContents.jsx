import { useState, useEffect } from 'react';
import './TableOfContents.css';

export default function TableOfContents() {
  const [headings, setHeadings] = useState([]);
  const [activeId, setActiveId] = useState('');

  useEffect(() => {
    const timer = setTimeout(() => {
      const els = document.querySelectorAll('.docs-content h2, .docs-content h3');
      const items = Array.from(els)
        .filter((el) => el.id)
        .map((el) => ({
          id: el.id,
          text: el.textContent.replace(/ #$/, ''),
          level: el.tagName === 'H2' ? 2 : 3,
        }));
      setHeadings(items);
    }, 100);
    return () => clearTimeout(timer);
  }, []);

  useEffect(() => {
    if (headings.length === 0) return;

    const observer = new IntersectionObserver(
      (entries) => {
        const visible = entries.filter((e) => e.isIntersecting);
        if (visible.length > 0) {
          setActiveId(visible[0].target.id);
        }
      },
      { rootMargin: '-80px 0px -60% 0px', threshold: 0 }
    );

    headings.forEach(({ id }) => {
      const el = document.getElementById(id);
      if (el) observer.observe(el);
    });

    return () => observer.disconnect();
  }, [headings]);

  if (headings.length === 0) return null;

  return (
    <nav className="toc" aria-label="Table of contents">
      <h4 className="toc-title">On this page</h4>
      <ul className="toc-list">
        {headings.map((h) => (
          <li key={h.id} className={`toc-item toc-item--${h.level}`}>
            <a
              href={`#${h.id}`}
              className={`toc-link${activeId === h.id ? ' toc-link--active' : ''}`}
              onClick={(e) => {
                e.preventDefault();
                document.getElementById(h.id)?.scrollIntoView({ behavior: 'smooth' });
              }}
            >
              {h.text}
            </a>
          </li>
        ))}
      </ul>
    </nav>
  );
}
