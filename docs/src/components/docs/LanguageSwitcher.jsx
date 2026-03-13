import { useNavigate, useLocation } from 'react-router-dom';
import './LanguageSwitcher.css';

export default function LanguageSwitcher() {
  const navigate = useNavigate();
  const { pathname } = useLocation();

  const isZh = pathname.startsWith('/zh/');
  const currentLang = isZh ? 'zh' : 'en';

  const handleSwitch = () => {
    if (isZh) {
      navigate(pathname.replace(/^\/zh\//, '/'));
    } else {
      navigate('/zh' + pathname);
    }
  };

  return (
    <button className="lang-switcher" onClick={handleSwitch} title={isZh ? 'Switch to English' : '切换到中文'}>
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="12" cy="12" r="10" />
        <line x1="2" y1="12" x2="22" y2="12" />
        <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
      </svg>
      <span>{currentLang === 'en' ? '中文' : 'EN'}</span>
    </button>
  );
}
