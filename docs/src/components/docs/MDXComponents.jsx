import { Link } from 'react-router-dom';
import './MDXComponents.css';

function MdxLink({ href, children, ...props }) {
  if (href && (href.startsWith('/') || href.startsWith('./'))) {
    return <Link to={href} {...props}>{children}</Link>;
  }
  return <a href={href} target="_blank" rel="noopener noreferrer" {...props}>{children}</a>;
}

function MdxPre({ children, ...props }) {
  return <div className="mdx-code-wrapper"><pre {...props}>{children}</pre></div>;
}

function MdxCode({ children, className, ...props }) {
  if (className) {
    return <code className={className} {...props}>{children}</code>;
  }
  return <code className="mdx-inline-code" {...props}>{children}</code>;
}

function MdxTable({ children }) {
  return (
    <div className="mdx-table-wrapper">
      <table>{children}</table>
    </div>
  );
}

function MdxBlockquote({ children }) {
  return <blockquote className="mdx-blockquote">{children}</blockquote>;
}

function MdxH1({ children, ...props }) {
  return <h1 className="mdx-heading" {...props}>{children}</h1>;
}

function MdxH2({ children, ...props }) {
  return <h2 className="mdx-heading" {...props}>{children}</h2>;
}

function MdxH3({ children, ...props }) {
  return <h3 className="mdx-heading" {...props}>{children}</h3>;
}

function MdxH4({ children, ...props }) {
  return <h4 className="mdx-heading" {...props}>{children}</h4>;
}

function MdxImg({ src, alt, ...props }) {
  const resolvedSrc = src && src.startsWith('/') ? src : src;
  return (
    <span className="mdx-img-wrapper">
      <img src={resolvedSrc} alt={alt || ''} loading="lazy" {...props} />
    </span>
  );
}

export const mdxComponents = {
  a: MdxLink,
  pre: MdxPre,
  code: MdxCode,
  table: MdxTable,
  blockquote: MdxBlockquote,
  h1: MdxH1,
  h2: MdxH2,
  h3: MdxH3,
  h4: MdxH4,
  img: MdxImg,
};
