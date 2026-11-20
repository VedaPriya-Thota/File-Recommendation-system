import React, { useEffect, useState, useCallback } from 'react';
import { Search, ArrowUp, ArrowDown, ExternalLink, FolderOpen, Copy, Eye, Files as FilesIcon, FolderPlus, ChevronLeft, ChevronRight } from 'lucide-react';
import { api, errMsg } from '../api/client';
import { Empty, FileIcon, Skeletons, StateBadge, fmtBytes, fmtNum, timeAgo } from '../components/ui.jsx';

const PAGE = 50;
export default function FilesPage({ go, toast, openPreview }) {
  const [f, setF] = useState({ q: '', file_type: '', group: '', folder_id: '', state: '', modified_within_days: '', duplicates_only: false, sort: 'modified', direction: 'desc' });
  const [debQ, setDebQ] = useState('');
  const [offset, setOffset] = useState(0);
  const [data, setData] = useState(null);
  const [facets, setFacets] = useState(null);
  const [folders, setFolders] = useState([]);
  const [err, setErr] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => { api.facets().then(setFacets).catch(() => {}); api.folders().then(setFolders).catch(() => {}); }, []);
  useEffect(() => { const t = setTimeout(() => { setDebQ(f.q); setOffset(0); }, 250); return () => clearTimeout(t); }, [f.q]);
  const load = useCallback(() => {
    setLoading(true);
    const p = { ...f, q: debQ, limit: PAGE, offset };
    Object.keys(p).forEach((k) => (p[k] === '' || p[k] === false) && delete p[k]);
    api.files(p).then((d) => { setData(d); setErr(''); }).catch((e) => setErr(errMsg(e))).finally(() => setLoading(false));
  }, [f.file_type, f.group, f.folder_id, f.state, f.modified_within_days, f.duplicates_only, f.sort, f.direction, debQ, offset]);
  useEffect(() => { load(); }, [load]);
  useEffect(() => { const t = setInterval(load, 6000); return () => clearInterval(t); }, [load]);

  const set = (k, v) => { setF((x) => ({ ...x, [k]: v })); setOffset(0); };
  const act = async (e, fn, ok) => { e.stopPropagation(); try { await fn(); toast(ok); } catch (x) { toast(errMsg(x), true); } };
  const sortBy = (k) => setF((x) => ({ ...x, sort: k, direction: x.sort === k && x.direction === 'desc' ? 'asc' : 'desc' }));
  const Th = ({ k, children }) => <th onClick={() => sortBy(k)} style={{ cursor: 'pointer' }}>{children}{f.sort === k && (f.direction === 'asc' ? <ArrowUp size={11} /> : <ArrowDown size={11} />)}</th>;
  const filtered = f.q || f.file_type || f.group || f.folder_id || f.state || f.modified_within_days || f.duplicates_only;
  const total = data?.total ?? 0;

  return (
    <div className="page">
      <div className="page-head"><div><h1>All Files</h1><p>Every file discovered in your connected folders{data ? ` — ${fmtNum(total)} ${filtered ? 'match' : 'total'}` : ''}.</p></div></div>
      <div className="card">
        <div className="toolbar">
          <div className="search"><Search size={16} /><input className="input" placeholder="Search file names and paths…" value={f.q} onChange={(e) => setF({ ...f, q: e.target.value })} aria-label="Search files" /></div>
          <select className="select" value={f.group} onChange={(e) => set('group', e.target.value)} aria-label="Category"><option value="">All categories</option><option value="document">Documents</option><option value="data">Data</option><option value="code">Code</option></select>
          <select className="select" value={f.file_type} onChange={(e) => set('file_type', e.target.value)} aria-label="Type"><option value="">All types</option>{facets?.types.map((t) => <option key={t.t} value={t.t}>.{t.t} ({t.n})</option>)}</select>
          <select className="select" value={f.folder_id} onChange={(e) => set('folder_id', e.target.value)} aria-label="Folder"><option value="">All folders</option>{folders.map((x) => <option key={x.folder_id} value={x.folder_id}>{x.label}</option>)}</select>
          <select className="select" value={f.state} onChange={(e) => set('state', e.target.value)} aria-label="Status"><option value="">Any status</option><option value="content_indexed">Indexed</option><option value="unsupported">Metadata only</option><option value="failed">Failed</option></select>
          <select className="select" value={f.modified_within_days} onChange={(e) => set('modified_within_days', e.target.value)} aria-label="Modified"><option value="">Any time</option><option value="1">Last 24 h</option><option value="7">Last 7 days</option><option value="30">Last 30 days</option><option value="365">Last year</option></select>
          <label className="row small"><input type="checkbox" checked={f.duplicates_only} onChange={(e) => set('duplicates_only', e.target.checked)} /> Duplicates</label>
          {filtered && <button className="btn sm" onClick={() => { setF({ ...f, q: '', file_type: '', group: '', folder_id: '', state: '', modified_within_days: '', duplicates_only: false }); setOffset(0); }}>Clear</button>}
        </div>
      </div>
      <div className="card" style={{ padding: 0 }}>
        {err && <div className="alert err" style={{ margin: 14 }}>{err}</div>}
        {!data ? <div style={{ padding: 18 }}><Skeletons n={6} /></div> : data.items.length === 0 ? (
          filtered ? <Empty icon={Search} title="No files match these filters" text="Try removing a filter or changing the search text." />
            : <Empty icon={FilesIcon} title="No files yet" text="Connect a folder to see its files here."><button className="btn primary" onClick={() => go('folders')}><FolderPlus size={16} /> Connect a folder</button></Empty>
        ) : (
          <div className="table-wrap" style={{ opacity: loading ? .7 : 1 }}>
            <table className="t"><thead><tr><Th k="name">Name</Th><Th k="type">Type</Th><th>Folder</th><Th k="size">Size</Th><Th k="modified">Modified</Th><th>Status</th><th /></tr></thead>
              <tbody>{data.items.map((x) => (
                <tr key={x.file_id} onClick={() => openPreview(x.file_id, debQ)}>
                  <td><div className="row"><FileIcon type={x.file_type} /><div style={{ minWidth: 0 }}><div style={{ fontWeight: 600 }} className="trunc">{x.filename}</div><div className="muted small trunc" style={{ maxWidth: 420 }} title={x.abs_path}>{x.rel_path}</div></div></div></td>
                  <td><span className="badge">.{x.file_type}</span></td><td className="muted">{x.folder_label}</td><td className="muted">{fmtBytes(x.size_bytes)}</td><td className="muted" title={x.modified}>{timeAgo(x.modified)}</td>
                  <td><StateBadge state={x.state} error={x.error_code} />{x.duplicate_of && <span className="badge info" style={{ marginLeft: 4 }}>duplicate</span>}</td>
                  <td><div className="row" style={{ gap: 2, justifyContent: 'flex-end' }}>
                    <button className="btn icon ghost" title="Preview" onClick={(e) => { e.stopPropagation(); openPreview(x.file_id, debQ); }}><Eye size={15} /></button>
                    <button className="btn icon ghost" title="Open" onClick={(e) => act(e, () => api.openFile(x.file_id), 'Opened with your default app')}><ExternalLink size={15} /></button>
                    <button className="btn icon ghost" title="Show in folder" onClick={(e) => act(e, () => api.reveal(x.file_id), 'Opened containing folder')}><FolderOpen size={15} /></button>
                    <button className="btn icon ghost" title="Copy path" onClick={(e) => act(e, () => navigator.clipboard.writeText(x.abs_path), 'Path copied')}><Copy size={15} /></button></div></td>
                </tr>))}</tbody></table>
          </div>
        )}
        {data && total > PAGE && (
          <div className="row" style={{ padding: 12, justifyContent: 'space-between', borderTop: '1px solid var(--border)' }}>
            <span className="muted small">{fmtNum(offset + 1)}–{fmtNum(Math.min(offset + PAGE, total))} of {fmtNum(total)}</span>
            <div className="row"><button className="btn sm" disabled={offset === 0} onClick={() => setOffset(Math.max(0, offset - PAGE))}><ChevronLeft size={14} /> Prev</button>
              <button className="btn sm" disabled={offset + PAGE >= total} onClick={() => setOffset(offset + PAGE)}>Next <ChevronRight size={14} /></button></div>
          </div>
        )}
      </div>
    </div>
  );
}
