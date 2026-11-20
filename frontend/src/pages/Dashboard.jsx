import React, { useEffect, useState } from 'react';
import { FolderTree, Files, Database, HardDrive, AlertTriangle, Layers, Sparkles, FolderPlus, Cpu, ShieldCheck } from 'lucide-react';
import { api, errMsg } from '../api/client';
import { Empty, FileIcon, Skeletons, fmtBytes, fmtNum, timeAgo, StateBadge } from '../components/ui.jsx';
import FolderPicker from '../components/FolderPicker.jsx';

export default function Dashboard({ go, toast, openPreview }) {
  const [s, setS] = useState(null);
  const [err, setErr] = useState('');
  const [emb, setEmb] = useState(null);
  const [adding, setAdding] = useState(false);
  const load = () => api.stats().then(setS).catch((e) => setErr(errMsg(e)));
  useEffect(() => { load(); api.embedding().then(setEmb).catch(() => {}); const t = setInterval(load, 2500); return () => clearInterval(t); }, []);

  if (err && !s) return <div className="page"><div className="alert err">Could not reach the backend: {err}</div></div>;
  if (!s) return <div className="page"><h1>Overview</h1><div className="card"><Skeletons /></div></div>;
  const maxType = Math.max(1, ...s.types.map((t) => t.n));

  return (
    <div className="page">
      <div className="page-head">
        <div><h1>Overview</h1><p>What the assistant can currently see and search.</p></div>
        <div className="row"><button className="btn" onClick={() => setAdding(true)}><FolderPlus size={16} /> Connect folder</button><button className="btn primary" onClick={() => go('assistant')} disabled={!s.indexed}><Sparkles size={16} /> Ask the assistant</button></div>
      </div>
      {s.folders === 0 ? (
        <div className="card"><Empty icon={FolderTree} title="Connect your first folder" text="Choose a directory on this computer. Its files are indexed locally (nothing is uploaded) so the assistant can search and analyze them.">
          <button className="btn primary" onClick={() => setAdding(true)}><FolderPlus size={16} /> Connect a folder</button></Empty></div>
      ) : (
        <>
          {s.active_scans.length > 0 && (
            <div className="alert info"><Layers size={18} /><div className="grow">Indexing in progress: {s.active_scans.map((j) => `${j.phase === 'queued' ? 'queued' : j.phase.replace('_', ' ')} (${fmtNum(j.processed)}/${fmtNum(j.total)})`).join(' · ')}</div><button className="btn sm" onClick={() => go('folders')}>Details</button></div>
          )}
          <div className="grid stats">
            {[[FolderTree, 'Connected folders', s.folders], [Files, 'Files discovered', s.files], [Database, 'Searchable (indexed)', s.indexed], [Layers, 'Text chunks / vectors', s.chunks],
              [HardDrive, 'Total size', fmtBytes(s.bytes)], [AlertTriangle, 'Failed / metadata-only', `${s.failed} / ${s.unsupported}`]].map(([I, l, v]) => (
              <div className="card stat" key={l}><div className="l"><I size={14} />{l}</div><div className="v">{typeof v === 'number' ? fmtNum(v) : v}</div></div>
            ))}
          </div>
          <div className="grid two">
            <div className="card">
              <div className="card-head"><h2>File types</h2><button className="btn sm" onClick={() => go('files')}>Browse all</button></div>
              {s.types.length === 0 ? <div className="muted">Nothing indexed yet.</div> : s.types.map((t) => (
                <div className="bar-row" key={t.t}><span className="mono">.{t.t}</span><div className="bar"><div style={{ width: `${(t.n / maxType) * 100}%` }} /></div><span className="muted">{fmtNum(t.n)}</span></div>
              ))}
            </div>
            <div className="card col">
              <h2>System status</h2>
              <div className="row"><Cpu size={16} className="muted" /><div className="grow"><div>Embeddings</div><div className="muted small">{emb ? emb.name : 'loading…'}</div></div>{emb && <span className={`badge ${emb.semantic ? 'ok' : 'warn'}`}>{emb.semantic ? 'semantic' : 'keyword-like'}</span>}</div>
              {emb && !emb.semantic && <div className="alert warn small">Semantic search needs a sentence-embedding model. Install <span className="mono">sentence-transformers</span> (see README) for meaning-based matching; keyword search works regardless.</div>}
              <div className="row"><Sparkles size={16} className="muted" /><div className="grow"><div>Language model</div><div className="muted small">{s.llm.active ? `${s.llm.provider} · ${s.llm.model}` : s.llm.reason}</div></div><span className={`badge ${s.llm.active ? 'ok' : ''}`}>{s.llm.active ? 'on' : 'off'}</span></div>
              <div className="row"><ShieldCheck size={16} className="muted" /><div className="grow"><div>Privacy</div><div className="muted small">Read-only access · local index · {s.duplicates} duplicate file(s) deduplicated</div></div></div>
              <div className="row"><Sparkles size={16} className="muted" /><div className="grow"><div>Assistant requests</div></div><span className="badge">{s.assistant_runs}</span></div>
            </div>
          </div>
          <div className="card">
            <div className="card-head"><h2>Recently modified</h2></div>
            {s.recent.length === 0 ? <div className="muted">No files yet.</div> : (
              <div className="table-wrap"><table className="t"><tbody>{s.recent.map((f) => (
                <tr key={f.file_id} onClick={() => openPreview(f.file_id)}><td style={{ width: 44 }}><FileIcon type={f.file_type} /></td><td><div style={{ fontWeight: 600 }}>{f.filename}</div><div className="muted small trunc" style={{ maxWidth: 520 }}>{f.folder_label} › {f.rel_path}</div></td>
                  <td><StateBadge state={f.state} /></td><td className="muted">{fmtBytes(f.size_bytes)}</td><td className="muted">{timeAgo(f.modified)}</td></tr>))}</tbody></table></div>
            )}
          </div>
        </>
      )}
      {adding && <FolderPicker onClose={() => setAdding(false)} onAdded={() => { load(); go('folders'); }} toast={toast} />}
    </div>
  );
}
