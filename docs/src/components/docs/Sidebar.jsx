import { useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { enNav, zhNav } from '../../data/docsNav';
import LanguageSwitcher from './LanguageSwitcher';
import './Sidebar.css';

export default function Sidebar({ lang }) {
  const [mobileOpen, setMobileOpen] = useState(false);
  const nav = lang === 'zh' ? zhNav : enNav;
  const basePath = lang === 'zh' ? '/zh/docs' : '/docs';
  const location = useLocation();

  const getTo = (slug) => slug === 'index' ? basePath : `${basePath}/${slug}`;

  const isActive = (slug) => {
    const path = location.pathname;
    if (slug === 'index') {
      return path === basePath || path === basePath + '/';
    }
    return path === `${basePath}/${slug}` || path === `${basePath}/${slug}/`;
  };

  return (
    <>
      <button
        className={`sidebar-toggle${mobileOpen ? ' sidebar-toggle--open' : ''}`}
        onClick={() => setMobileOpen(!mobileOpen)}
        aria-label="Toggle sidebar"
      >
        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          {mobileOpen ? (
            <><line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" /></>
          ) : (
            <><line x1="3" y1="12" x2="21" y2="12" /><line x1="3" y1="6" x2="21" y2="6" /><line x1="3" y1="18" x2="21" y2="18" /></>
          )}
        </svg>
      </button>

      {mobileOpen && <div className="sidebar-overlay" onClick={() => setMobileOpen(false)} />}

      <aside className={`sidebar${mobileOpen ? ' sidebar--open' : ''}`}>
        <div className="sidebar-header">
          <LanguageSwitcher />
        </div>
        <nav className="sidebar-nav">
          {nav.map((item) => (
            <NavLink
              key={item.slug}
              to={getTo(item.slug)}
              end={item.slug === 'index'}
              className={() => `sidebar-link${isActive(item.slug) ? ' sidebar-link--active' : ''}`}
              onClick={() => setMobileOpen(false)}
            >
              <span className="sidebar-icon">{item.icon}</span>
              <span>{item.title}</span>
            </NavLink>
          ))}
        </nav>
      </aside>
    </>
  );
}
