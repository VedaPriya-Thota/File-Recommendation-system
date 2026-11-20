import React, { useEffect, useState } from 'react';
import { X, ExternalLink, FolderOpen, Copy, Download, Sparkles } from 'lucide-react';
import { api, errMsg } from '../api/client';
import { FileIcon, StateBadge, Spinner, fmtBytes, fmtDate, Highlight } from './ui.jsx';

export default function FileDrawer({ fileId, query, onClose, toast, onAsk }) {
  const [d, setD] = useState(null);
  const [err, setErr] = useState('');
  useEffect(() => {
    setD(null); setErr('');
    api.preview(fileId, query).then(setD).catch((e) => setErr(errMsg(e)));
  }, [fileId, query]);
  useEffect(() => { const h = (e) => e.key === 'Escape' && onClose(); window.addEventListener('keydown', h); return () => window.removeEventListener('keydown', h); }, [onClose]);

  const f = d?.file;
  const act = async (fn, ok) => { try { await fn(); toast(ok); } catch (e) { toast(errMsg(e), true); } };
  const terms = query ? query.split(/\s+/) : [];
  return (
    <aside className="drawer" role="complementary" aria-label="File preview">
      <div className="modal-head">
        <div className="row grow" style={{ minWidth: 0 }}>{f && <FileIcon type={f.file_type} />}<h2 className="trunc">{f?.filename || 'Loading…'}</h2></div>
        <button className="btn icon ghost" onClick={onClose} aria-label="Close preview"><X size={18} /></button>
      </div>
      <div className="drawer-body">
        {err && <div className="alert err">{err}</div>}
        {!d && !err && <Spinner />}
        {f && (
          <>
            <div className="row wrap">
              <button className="btn sm primary" onClick={() => act(() => api.openFile(f.file_id), 'Opened with your default app')}><ExternalLink size={14} /> Open</button>
              <button className="btn sm" onClick={() => act(() => api.reveal(f.file_id), 'Opened containing folder')}><FolderOpen size={14} /> Show in folder</button>
              <button className="btn sm" onClick={() => act(() => navigator.clipboard.writeText(f.abs_path), 'Path copied')}><Copy size={14} /> Copy path</button>
              <a className="btn sm" href={api.rawUrl(f.file_id)}><Download size={14} /> Download</a>
              {onAsk && <button className="btn sm" onClick={() => onAsk(f)}><Sparkles size={14} /> Ask AI</button>}
            </div>
            <dl className="kv">
              <dt>Location</dt><dd className="mono">{f.abs_path}</dd>
              <dt>Folder</dt><dd>{f.folder_label}</dd>
              <dt>Size</dt><dd>{fmtBytes(f.size_bytes)}</dd>
              <dt>Modified</dt><dd>{fmtDate(f.modified)}</dd>
              <dt>Index status</dt><dd><StateBadge state={f.state} error={f.error_code} /> {f.error_code && <span className="muted small">({f.error_code})</span>} {f.chunks > 0 && <span className="muted small"> {f.chunks} chunk(s)</span>}</dd>
              {f.duplicate_of && <><dt>Duplicate</dt><dd>Identical content to file #{f.duplicate_of} (embeddings were reused)</dd></>}
              {Object.entries(f.meta || {}).filter(([k]) => !['keys', 'columns', 'sheets'].includes(k)).map(([k, v]) => <React.Fragment key={k}><dt>{k.replace(/_/g, ' ')}</dt><dd>{String(v)}</dd></React.Fragment>)}
              {f.meta?.columns && <><dt>columns</dt><dd>{f.meta.columns.join(', ')}</dd></>}
              {f.meta?.sheets && <><dt>sheets</dt><dd>{f.meta.sheets.map((s) => `${s.name} (${s.rows} rows)`).join(', ')}</dd></>}
            </dl>
            {d.excerpts.length > 0 && (
              <div><h3>Matches for “{query}”</h3>{d.excerpts.map((e, i) => <div key={i} className="excerpt"><Highlight text={e.text} terms={e.matched_terms} /></div>)}</div>
            )}
            {d.table ? d.table.map((t) => (
              <div key={t.sheet}><h3>{t.sheet} <span className="muted small">— {t.total_rows.toLocaleString()} rows</span></h3>
                <div className="table-wrap" style={{ maxHeight: 340, overflow: 'auto', border: '1px solid var(--border)', borderRadius: 10 }}>
                  <table className="t"><thead><tr>{t.rows[0]?.map((c, i) => <th key={i}>{c}</th>)}</tr></thead>
                    <tbody>{t.rows.slice(1).map((r, i) => <tr key={i} style={{ cursor: 'default' }}>{r.map((c, j) => <td key={j}>{c}</td>)}</tr>)}</tbody></table></div></div>
            )) : d.text ? (
              <div><h3>Content preview {d.truncated && <span className="muted small">(first {d.text.length.toLocaleString()} of {d.total_chars.toLocaleString()} characters)</span>}</h3><pre className="preview">{d.text}</pre></div>
            ) : <div className="alert info">{f.state === 'unsupported' ? 'This file type is not supported for content extraction, so only its metadata is indexed.' : 'No extractable text.'}</div>}
          </>
        )}
      </div>
    </aside>
  );
}
