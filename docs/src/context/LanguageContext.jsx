import { createContext, useContext } from 'react';
import { useLocation } from 'react-router-dom';

const LanguageContext = createContext({ lang: 'en' });

export function LanguageProvider({ children }) {
  const { pathname } = useLocation();
  const lang = pathname.startsWith('/zh/') || pathname.startsWith('/MER-Factory/zh/') ? 'zh' : 'en';

  return (
    <LanguageContext.Provider value={{ lang }}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage() {
  return useContext(LanguageContext);
}
