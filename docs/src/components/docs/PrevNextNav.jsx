import { Link, useLocation } from 'react-router-dom';
import { enNav, zhNav } from '../../data/docsNav';
import './PrevNextNav.css';

export default function PrevNextNav({ lang }) {
  const nav = lang === 'zh' ? zhNav : enNav;
  const basePath = lang === 'zh' ? '/zh/docs' : '/docs';
  const { pathname } = useLocation();

  const currentSlug = pathname === basePath || pathname === basePath + '/'
    ? 'index'
    : pathname.replace(basePath + '/', '').replace(/\/$/, '');

  const currentIndex = nav.findIndex((item) => item.slug === currentSlug);
  const prev = currentIndex > 0 ? nav[currentIndex - 1] : null;
  const next = currentIndex < nav.length - 1 ? nav[currentIndex + 1] : null;

  const getTo = (slug) => (slug === 'index' ? basePath : `${basePath}/${slug}`);

  if (!prev && !next) return null;

  return (
    <nav className="prev-next" aria-label="Page navigation">
      {prev ? (
        <Link to={getTo(prev.slug)} className="prev-next-card prev-next-card--prev">
          <span className="prev-next-label">Previous</span>
          <span className="prev-next-title">{prev.icon} {prev.title}</span>
        </Link>
      ) : <div />}
      {next ? (
        <Link to={getTo(next.slug)} className="prev-next-card prev-next-card--next">
          <span className="prev-next-label">Next</span>
          <span className="prev-next-title">{next.title} {next.icon}</span>
        </Link>
      ) : <div />}
    </nav>
  );
}
