import React, { useEffect, useState } from 'react';
import { Folder, FolderOpen, ArrowUp, Monitor, ShieldCheck } from 'lucide-react';
import { api, errMsg } from '../api/client';
import { Modal, Spinner } from './ui.jsx';

export default function FolderPicker({ onClose, onAdded, toast }) {
  const [path, setPath] = useState('');
  const [listing, setListing] = useState(null);
  const [typed, setTyped] = useState('');
  const [label, setLabel] = useState('');
  const [loading, setLoading] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [native, setNative] = useState(false);

  const load = async (p) => {
    setLoading(true); setError('');
    try { const d = await api.browse(p); setListing(d); setPath(d.path); setTyped(d.path); } catch (e) { setError(errMsg(e)); } finally { setLoading(false); }
  };
  useEffect(() => { load(''); }, []);

  const connect = async () => {
    setBusy(true); setError('');
    try { const f = await api.addFolder({ path: typed || path, label: label || undefined }); toast(`Connected “${f.label}” — scanning started`); onAdded(f); onClose(); }
    catch (e) { setError(errMsg(e)); } finally { setBusy(false); }
  };
  const pickNative = async () => {
    setNative(true); setError('');
    try { const d = await api.pick(); if (d.path) { setTyped(d.path); await load(d.path).catch(() => {}); } } catch (e) { setError(errMsg(e)); } finally { setNative(false); }
  };

  return (
    <Modal title="Connect a folder" onClose={onClose} width={680}
      foot={<><button className="btn" onClick={onClose}>Cancel</button><button className="btn primary" disabled={busy || !(typed || path)} onClick={connect}>{busy ? <Spinner /> : <ShieldCheck size={16} />} Authorize &amp; index this folder</button></>}>
      <div className="alert info"><ShieldCheck size={18} style={{ flex: 'none' }} /><div>Only the folder you choose (and its sub-folders) will be read. Files are never modified, moved or uploaded — indexing happens on this computer.</div></div>
      {error && <div className="alert err">{error}</div>}
      <div className="row">
        <input className="input mono" value={typed} onChange={(e) => setTyped(e.target.value)} placeholder="Absolute path, e.g. /home/me/Documents" onKeyDown={(e) => e.key === 'Enter' && load(typed)} />
        <button className="btn" onClick={() => load(typed)}>Go</button>
        <button className="btn" onClick={pickNative} disabled={native} title="Open the operating-system folder dialog">{native ? <Spinner /> : <Monitor size={16} />} System dialog</button>
      </div>
      <div className="dirlist">
        {loading ? <div style={{ padding: 16 }}><Spinner /></div> : (
          <>
            {listing?.parent && <button onClick={() => load(listing.parent)}><ArrowUp size={16} /> <span className="muted">.. (up)</span></button>}
            {listing?.entries.map((e) => <button key={e.path} onClick={() => load(e.path)}><Folder size={16} color="var(--primary)" /><span className="grow trunc">{listing.shortcuts ? e.path : e.name}</span></button>)}
            {listing && !listing.entries.length && <div className="muted" style={{ padding: 14 }}>No sub-folders here.</div>}
          </>
        )}
      </div>
      <div className="row"><FolderOpen size={16} className="muted" /><span className="small muted">Selected:</span><span className="mono trunc grow">{typed || '—'}</span></div>
      <input className="input" value={label} onChange={(e) => setLabel(e.target.value)} placeholder="Display name (optional)" />
    </Modal>
  );
}
