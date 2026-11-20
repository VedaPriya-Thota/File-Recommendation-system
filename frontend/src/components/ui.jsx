import React, { useEffect } from 'react';
import { File, FileText, FileSpreadsheet, FileCode, FileJson, Presentation, FileType2, X, Loader2 } from 'lucide-react';

export const fmtBytes = (n) => {
  if (n == null) return '—';
  if (n < 1024) return `${n} B`;
  const u = ['KB', 'MB', 'GB', 'TB']; let i = -1;
  do { n /= 1024; i++; } while (n >= 1024 && i < u.length - 1);
  return `${n.toFixed(n >= 100 ? 0 : 1)} ${u[i]}`;
};
export const fmtNum = (n) => (n ?? 0).toLocaleString();
export const timeAgo = (s) => {
  if (!s) return '—';
  const d = new Date(s); if (isNaN(d)) return '—';
  const sec = (Date.now() - d.getTime()) / 1000;
  if (sec < 60) return 'just now'; if (sec < 3600) return `${Math.floor(sec / 60)} min ago`;
  if (sec < 86400) return `${Math.floor(sec / 3600)} h ago`; if (sec < 86400 * 45) return `${Math.floor(sec / 86400)} d ago`;
  return d.toLocaleDateString();
};
export const fmtDate = (s) => { const d = new Date(s); return isNaN(d) ? '—' : d.toLocaleString(); };

const CODE = new Set(['py', 'js', 'jsx', 'ts', 'tsx', 'java', 'c', 'cpp', 'h', 'go', 'rs', 'rb', 'php', 'cs', 'sql', 'sh', 'html', 'css', 'kt', 'swift', 'yaml', 'yml', 'toml', 'xml']);
export function FileIcon({ type, size = 17 }) {
  let I = File;
  if (['pdf', 'docx', 'txt', 'md', 'rst'].includes(type)) I = FileText;
  else if (['csv', 'tsv', 'xlsx', 'xlsm'].includes(type)) I = FileSpreadsheet;
  else if (['json', 'jsonl'].includes(type)) I = FileJson;
  else if (type === 'pptx') I = Presentation;
  else if (CODE.has(type)) I = FileCode;
  return <div className="fileicon"><I size={size} /></div>;
}
export const TypeBadge = ({ t }) => <span className="badge">.{t}</span>;

export function StateBadge({ state, error }) {
  const m = { content_indexed: ['ok', 'Indexed'], unsupported: ['', 'Metadata only'], failed: ['err', 'Failed'], metadata_indexed: ['info', 'Pending'], content_extracted: ['info', 'Pending'], discovered: ['info', 'Pending'] };
  const [c, l] = m[state] || ['', state];
  return <span className={`badge ${c}`} title={error || ''}>{l}</span>;
}
export const Spinner = ({ size = 16 }) => <Loader2 size={size} className="spin" />;
export const Skeletons = ({ n = 4 }) => <div className="col">{Array.from({ length: n }).map((_, i) => <div key={i} className="skeleton" style={{ height: 18, width: `${90 - i * 12}%` }} />)}</div>;

export function Empty({ icon: I, title, text, children }) {
  return <div className="empty"><div className="ico">{I && <I size={26} />}</div><h3 style={{ color: 'var(--text)' }}>{title}</h3><div style={{ maxWidth: 420 }}>{text}</div>{children}</div>;
}
export function Modal({ title, onClose, children, foot, width }) {
  useEffect(() => { const h = (e) => e.key === 'Escape' && onClose(); window.addEventListener('keydown', h); return () => window.removeEventListener('keydown', h); }, [onClose]);
  return (
    <div className="overlay" onMouseDown={(e) => e.target === e.currentTarget && onClose()}>
      <div className="modal" style={width ? { width } : null} role="dialog" aria-modal="true">
        <div className="modal-head"><h2>{title}</h2><button className="btn icon ghost" onClick={onClose} aria-label="Close"><X size={18} /></button></div>
        <div className="modal-body">{children}</div>
        {foot && <div className="modal-foot">{foot}</div>}
      </div>
    </div>
  );
}
export const Switch = ({ checked, onChange, disabled }) => (
  <label className="switch"><input type="checkbox" checked={!!checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)} /><span /></label>
);

export function Highlight({ text, terms = [] }) {
  const ts = (terms || []).filter((t) => t && t.length > 1).map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
  if (!ts.length) return <>{text}</>;
  const rx = new RegExp(`(${ts.map((t) => `${t}\\w*`).join('|')})`, 'gi');
  return <>{String(text).split(rx).map((p, i) => (i % 2 ? <mark key={i}>{p}</mark> : <React.Fragment key={i}>{p}</React.Fragment>))}</>;
}

/** Minimal, safe Markdown subset (bold, italic, code, bullets, [F#] citations). Never uses innerHTML. */
export function Markdown({ text, onCite }) {
  const inline = (s, key) => {
    const parts = [];
    const rx = /(\*\*[^*]+\*\*|`[^`]+`|\*[^*\n]+\*|\[F?\d+(?: · [^\]]+)?\])/g;
    let last = 0, m, i = 0;
    while ((m = rx.exec(s))) {
      if (m.index > last) parts.push(s.slice(last, m.index));
      const tok = m[0];
      if (tok.startsWith('**')) parts.push(<strong key={`${key}-${i++}`}>{tok.slice(2, -2)}</strong>);
      else if (tok.startsWith('`')) parts.push(<code key={`${key}-${i++}`}>{tok.slice(1, -1)}</code>);
      else if (/^\[F?\d/.test(tok)) { const inner = tok.slice(1, -1); parts.push(<span key={`${key}-${i++}`} className="cite" title={inner} onClick={() => onCite && onCite(inner.split(' · ')[0])}>{inner}</span>); }
      else parts.push(<em key={`${key}-${i++}`}>{tok.slice(1, -1)}</em>);
      last = m.index + tok.length;
    }
    if (last < s.length) parts.push(s.slice(last));
    return parts;
  };
  const blocks = String(text || '').split(/\n{2,}/);
  return (
    <div className="answer">
      {blocks.map((b, bi) => {
        const lines = b.split('\n').filter((l) => l.trim());
        if (lines.length && lines.every((l) => /^\s*[-*•]\s+/.test(l))) {
          return <ul key={bi}>{lines.map((l, li) => <li key={li}>{inline(l.replace(/^\s*[-*•]\s+/, ''), `${bi}-${li}`)}</li>)}</ul>;
        }
        return <p key={bi}>{lines.map((l, li) => <React.Fragment key={li}>{li > 0 && <br />}{inline(l, `${bi}-${li}`)}</React.Fragment>)}</p>;
      })}
    </div>
  );
}
