import React, { useEffect, useState, useCallback } from 'react';
import { FolderPlus, RefreshCw, Trash2, Square, Pencil, Eye, AlertTriangle, CheckCircle2, FolderTree, RotateCcw } from 'lucide-react';
import { api, errMsg } from '../api/client';
import { Empty, Modal, Skeletons, Spinner, Switch, fmtBytes, fmtNum, timeAgo } from '../components/ui.jsx';
import FolderPicker from '../components/FolderPicker.jsx';

const BUSY = ['queued', 'discovering', 'loading_model', 'indexing', 'removing'];
const PHASE = { queued: 'Queued', discovering: 'Discovering files…', loading_model: 'Loading embedding model…', indexing: 'Indexing', removing: 'Removing deleted files…' };

function FolderCard({ f, reload, toast }) {
  const [edit, setEdit] = useState(false);
  const [confirm, setConfirm] = useState(false);
  const [showErr, setShowErr] = useState(false);
  const job = f.scan, busy = job && BUSY.includes(job.phase);
  const pct = job && job.total ? Math.min(100, Math.round((job.processed / job.total) * 100)) : 0;
  const sum = f.last_scan_summary || {}, errs = sum.errors || [];
  const run = async (fn, ok) => { try { await fn(); if (ok) toast(ok); reload(); } catch (e) { toast(errMsg(e), true); } };
  const status = !f.exists ? ['err', 'Unavailable'] : busy ? ['info', PHASE[job.phase]] : f.status === 'error' || f.status === 'unavailable' ? ['err', 'Error'] : ['ok', 'Up to date'];
  return (
    <div className="card col">
      <div className="row wrap">
        <FolderTree size={20} color="var(--primary)" /><div className="grow" style={{ minWidth: 0 }}><h2 className="trunc">{f.label}</h2><div className="mono muted trunc" title={f.path}>{f.path}</div></div>
        <span className={`badge ${status[0]}`}>{busy && <Spinner size={11} />} {status[1]}</span>
      </div>
      {(f.last_error || (job && job.error)) && <div className="alert err"><AlertTriangle size={16} />{job?.error || f.last_error}</div>}
      {busy && (
        <div className="col" style={{ gap: 6 }}>
          <div className={`progress ${job.total ? '' : 'indet'}`}><div style={{ width: `${pct}%` }} /></div>
          <div className="small muted row"><span>{job.phase === 'indexing' ? `${fmtNum(job.processed)} / ${fmtNum(job.total)} files (${pct}%)` : PHASE[job.phase]}</span>{job.current && <span className="mono trunc grow">{job.current}</span>}</div>
        </div>
      )}
      <div className="grid stats" style={{ gridTemplateColumns: 'repeat(auto-fit,minmax(110px,1fr))' }}>
        {[['Files', f.counts.files], ['Indexed', f.counts.indexed], ['Metadata only', f.counts.unsupported], ['Failed', f.counts.failed], ['Chunks', f.counts.chunks]].map(([l, v]) => (
          <div key={l}><div className="small muted">{l}</div><div style={{ fontSize: 20, fontWeight: 700 }}>{fmtNum(v)}</div></div>))}
        <div><div className="small muted">Size</div><div style={{ fontSize: 20, fontWeight: 700 }}>{fmtBytes(f.counts.bytes)}</div></div>
      </div>
      <div className="small muted row wrap" style={{ gap: 14 }}>
        <span>Last scan: {f.last_scan_finished ? timeAgo(f.last_scan_finished) : 'never'}</span>
        {sum.discovered != null && <span>+{sum.new} new · {sum.updated} updated · {sum.moved} moved · {sum.deleted} removed · {sum.duplicates} duplicate(s) · {sum.excluded} excluded</span>}
        <span>Auto-sync {f.watch ? 'on' : 'off'}</span>
      </div>
      {errs.length > 0 && (
        <div><button className="btn sm ghost" onClick={() => setShowErr(!showErr)}><AlertTriangle size={13} /> {errs.length} file(s) could not be read {showErr ? '▴' : '▾'}</button>
          {showErr && <div className="card flat" style={{ marginTop: 6, maxHeight: 180, overflow: 'auto' }}>{errs.map((e, i) => <div key={i} className="small"><span className="mono">{e.rel_path}</span> — <span className="badge err">{e.error_code}</span></div>)}</div>}</div>
      )}
      <div className="row wrap">
        {busy ? <button className="btn sm" onClick={() => run(() => api.cancelScan(f.folder_id), 'Stopping scan…')}><Square size={13} /> Stop</button>
          : <button className="btn sm" onClick={() => run(() => api.scan(f.folder_id), 'Scan started')}><RefreshCw size={13} /> Rescan</button>}
        <button className="btn sm" disabled={busy} onClick={() => run(() => api.rebuild(f.folder_id), 'Rebuilding index…')}><RotateCcw size={13} /> Rebuild</button>
        <button className="btn sm" onClick={() => setEdit(true)}><Pencil size={13} /> Exclusions</button>
        <label className="row small" style={{ marginLeft: 6 }}><Switch checked={f.watch} onChange={(v) => run(() => api.patchFolder(f.folder_id, { watch: v }))} /> Auto-sync</label>
        <div className="grow" /><button className="btn sm danger" onClick={() => setConfirm(true)}><Trash2 size={13} /> Remove</button>
      </div>
      {edit && <ExclusionModal f={f} onClose={() => setEdit(false)} onSaved={() => { setEdit(false); reload(); toast('Exclusions saved — rescanning'); }} />}
      {confirm && <Modal title="Remove this folder?" onClose={() => setConfirm(false)} foot={<><button className="btn" onClick={() => setConfirm(false)}>Cancel</button>
        <button className="btn danger" onClick={() => { setConfirm(false); run(() => api.removeFolder(f.folder_id), 'Folder removed from the index'); }}>Remove from index</button></>}>
        <div>This deletes the search index for <strong>{f.label}</strong> ({fmtNum(f.counts.files)} files). <strong>Your files on disk are not touched.</strong></div></Modal>}
    </div>
  );
}

function ExclusionModal({ f, onClose, onSaved }) {
  const [text, setText] = useState(f.exclusions.join('\n'));
  const [busy, setBusy] = useState(false); const [err, setErr] = useState('');
  const save = async () => { setBusy(true); try { await api.patchFolder(f.folder_id, { exclusions: text.split('\n') }); onSaved(); } catch (e) { setErr(errMsg(e)); setBusy(false); } };
  return (
    <Modal title={`Exclusions for ${f.label}`} onClose={onClose} foot={<><button className="btn" onClick={onClose}>Cancel</button><button className="btn primary" onClick={save} disabled={busy}>{busy && <Spinner />} Save &amp; rescan</button></>}>
      {err && <div className="alert err">{err}</div>}
      <div className="muted small">One pattern per line, applied on top of the global exclusions in Settings. Examples: <span className="mono">archive/</span> (a directory), <span className="mono">*.bak</span>, <span className="mono">drafts/*</span>.</div>
      <textarea className="textarea" value={text} onChange={(e) => setText(e.target.value)} rows={8} placeholder="private/&#10;*.bak" />
    </Modal>
  );
}

export default function Folders({ toast }) {
  const [list, setList] = useState(null);
  const [adding, setAdding] = useState(false);
  const [err, setErr] = useState('');
  const load = useCallback(() => api.folders().then((d) => { setList(d); setErr(''); }).catch((e) => setErr(errMsg(e))), []);
  useEffect(() => { load(); }, [load]);
  const anyBusy = list?.some((f) => f.scan && BUSY.includes(f.scan.phase));
  useEffect(() => { const t = setInterval(load, anyBusy ? 800 : 4000); return () => clearInterval(t); }, [load, anyBusy]);
  return (
    <div className="page">
      <div className="page-head"><div><h1>Connected Folders</h1><p>Directories you have authorized. They are scanned recursively, read-only, and kept in sync automatically.</p></div>
        <button className="btn primary" onClick={() => setAdding(true)}><FolderPlus size={16} /> Connect folder</button></div>
      {err && <div className="alert err">{err}</div>}
      {!list ? <div className="card"><Skeletons /></div> : list.length === 0 ? (
        <div className="card"><Empty icon={FolderTree} title="No folders connected" text="Pick a directory on this computer. Only what you authorize is ever read."><button className="btn primary" onClick={() => setAdding(true)}><FolderPlus size={16} /> Connect a folder</button></Empty></div>
      ) : list.map((f) => <FolderCard key={f.folder_id} f={f} reload={load} toast={toast} />)}
      {adding && <FolderPicker onClose={() => setAdding(false)} onAdded={load} toast={toast} />}
    </div>
  );
}
