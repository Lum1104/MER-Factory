import { useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import { MDXProvider } from '@mdx-js/react';
import Sidebar from './Sidebar';
import TableOfContents from './TableOfContents';
import PrevNextNav from './PrevNextNav';
import { mdxComponents } from './MDXComponents';
import './DocsLayout.css';

export default function DocsLayout({ pages, lang }) {
  const { pathname } = useLocation();
  const basePath = lang === 'zh' ? '/zh/docs' : '/docs';

  const slug =
    pathname === basePath || pathname === basePath + '/'
      ? 'index'
      : pathname.replace(basePath + '/', '').replace(/\/$/, '');

  const page = pages[slug];
  const Content = page?.default;
  const frontmatter = page?.frontmatter || {};

  useEffect(() => {
    window.scrollTo(0, 0);
  }, [pathname]);

  useEffect(() => {
    if (frontmatter.title) {
      document.title = `${frontmatter.title} — MER-Factory Docs`;
    }
  }, [frontmatter.title]);

  if (!Content) {
    return (
      <div className="docs-layout">
        <Sidebar lang={lang} />
        <div className="docs-main">
          <div className="docs-content">
            <h1 className="mdx-heading">Page not found</h1>
            <p>The requested documentation page does not exist.</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="docs-layout">
      <Sidebar lang={lang} />
      <div className="docs-main">
        <article className="docs-content" key={pathname}>
          <MDXProvider components={mdxComponents}>
            <Content />
          </MDXProvider>
          <PrevNextNav lang={lang} />
        </article>
        <div className="docs-toc-column">
          <TableOfContents key={pathname} />
        </div>
      </div>
    </div>
  );
}
