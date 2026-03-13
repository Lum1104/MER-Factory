import { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import useScrollPosition from '../hooks/useScrollPosition';
import './Navbar.css';

export default function Navbar() {
  const scrollY = useScrollPosition();
  const [menuOpen, setMenuOpen] = useState(false);

  const scrolled = scrollY > 50;

  const toggleMenu = () => {
    setMenuOpen((prev) => !prev);
  };

  const closeMenu = () => {
    setMenuOpen(false);
  };

  const navLinks = (
    <>
      <li>
        <Link to="/docs/" onClick={closeMenu}>
          Docs
        </Link>
      </li>
      <li>
        <a
          href="https://github.com/Lum1104/MER-Factory"
          target="_blank"
          rel="noopener noreferrer"
          onClick={closeMenu}
        >
          <svg
            className="navbar-github-icon"
            viewBox="0 0 16 16"
            aria-hidden="true"
          >
            <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z" />
          </svg>
          GitHub
        </a>
      </li>
    </>
  );

  return (
    <nav className={`navbar${scrolled ? ' navbar--scrolled' : ''}`}>
      <Link to="/" className="navbar-logo" onClick={closeMenu}>
        <img
          src={`${import.meta.env.BASE_URL}logo.svg`}
          alt="MER-Factory"
        />
      </Link>

      <ul className="navbar-links">{navLinks}</ul>

      <button
        className={`hamburger${menuOpen ? ' active' : ''}`}
        onClick={toggleMenu}
        aria-label="Toggle navigation menu"
        aria-expanded={menuOpen}
      >
        <span />
        <span />
        <span />
      </button>

      <div className={`mobile-menu${menuOpen ? ' mobile-menu--open' : ''}`}>
        <ul>{navLinks}</ul>
      </div>
    </nav>
  );
}
