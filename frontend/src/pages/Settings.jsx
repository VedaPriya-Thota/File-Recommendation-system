import React, { useEffect, useState } from 'react';
import { Save, RotateCcw, RefreshCw, ShieldCheck, Cpu, Database, SlidersHorizontal, Trash2 } from 'lucide-react';
import { api, errMsg } from '../api/client';
import { Skeletons, Spinner, Switch, Modal } from '../components/ui.jsx';

export default function SettingsPage({ toast }) {
  const [data, setData] = useState(null);
  const [s, setS] = useState(null);
  const [dirty, setDirty] = useState(false);
  const [saving, setSaving] = useState(false);
  const [err, setErr] = useState('');
  const [key, setKey] = useState('');
  const [emb, setEmb] = useState(null);
  const [confirmReset, setConfirmReset] = useState(false);

  const load = () => api.settings().then((d) => { setData(d); setS(d.settings); setDirty(false); }).catch((e) => setErr(errMsg(e)));
  useEffect(() => { load(); api.embedding().then(setEmb).catch(() => {}); }, []);
  if (err && !s) return <div className="page"><div className="alert err">{err}</div></div>;
  if (!s) return <div className="page"><h1>Settings</h1><div className="card"><Skeletons /></div></div>;

  const set = (sec, k, v) => { setS((x) => ({ ...x, [sec]: { ...x[sec], [k]: v } })); setDirty(true); };
  const save = async () => {
    setSaving(true);
    try { const r = await api.saveSettings(s); setS(r.settings); setData((d) => ({ ...d, llm: r.llm })); setDirty(false); toast('Settings saved'); }
    catch (e) { toast(errMsg(e), true); } finally { setSaving(false); }
  };
  const saveKey = async () => {
    try { const r = await api.setKey(s.model.provider, key); setData((d) => ({ ...d, llm: r.llm })); setKey(''); toast(key ? 'API key stored in memory for this session' : 'API key cleared'); }
    catch (e) { toast(errMsg(e), true); }
  };
  const llm = data.llm;
  const fmts = data.supported_formats;

  return (
    <div className="page">
      <div className="page-head"><div><h1>Settings</h1><p>Indexing, privacy and model configuration. Stored locally in {data.data_dir} ({data.database}).</p></div>
        <div className="row"><button className="btn" onClick={() => setConfirmReset(true)}><RotateCcw size={15} /> Reset defaults</button><button className="btn primary" onClick={save} disabled={!dirty || saving}>{saving ? <Spinner /> : <Save size={15} />} Save changes</button></div></div>

      <div className="card">
        <div className="card-head"><h2><SlidersHorizontal size={16} style={{ verticalAlign: -3 }} /> Indexing</h2></div>
        <div className="setting"><div><strong>Automatic sync</strong><div className="muted small">Watch connected folders and index added, changed, moved and deleted files.</div></div>
          <div className="row"><input className="input" type="number" style={{ width: 90 }} min={10} max={3600} value={s.indexing.watch_interval_s} onChange={(e) => set('indexing', 'watch_interval_s', e.target.value)} disabled={!s.indexing.watch_enabled} aria-label="Sync interval seconds" /><span className="small muted">sec</span><Switch checked={s.indexing.watch_enabled} onChange={(v) => set('indexing', 'watch_enabled', v)} /></div></div>
        <div className="setting"><div><strong>Semantic indexing</strong><div className="muted small">Create embeddings for meaning-based search. Turn off for faster, keyword-only indexing.</div></div><Switch checked={s.indexing.semantic_enabled} onChange={(v) => set('indexing', 'semantic_enabled', v)} /></div>
        <div className="setting"><div><strong>Include hidden files</strong><div className="muted small">Files and folders starting with “.”</div></div><Switch checked={s.indexing.include_hidden} onChange={(v) => set('indexing', 'include_hidden', v)} /></div>
        <div className="setting"><div><strong>Maximum file size</strong><div className="muted small">Larger files are listed but not read.</div></div><div className="row"><input className="input" type="number" style={{ width: 90 }} min={1} max={500} value={s.indexing.max_file_mb} onChange={(e) => set('indexing', 'max_file_mb', e.target.value)} /><span className="small muted">MB</span></div></div>
        <div style={{ padding: '12px 0' }}><strong>Supported formats</strong><div className="muted small" style={{ marginBottom: 8 }}>Unchecked formats are still listed (metadata only) but never read.</div>
          <div className="fmt">{fmts.map((x) => <label key={x}><input type="checkbox" checked={s.indexing.enabled_formats.includes(x)} onChange={(e) => set('indexing', 'enabled_formats', e.target.checked ? [...s.indexing.enabled_formats, x] : s.indexing.enabled_formats.filter((y) => y !== x))} /><span>.{x}</span></label>)}</div></div>
        <div style={{ padding: '12px 0 4px' }}><strong>Global exclusions</strong><div className="muted small" style={{ marginBottom: 8 }}>One pattern per line. A trailing “/” matches directories only. Applies to every folder.</div>
          <textarea className="textarea" rows={7} value={s.indexing.exclusions.join('\n')} onChange={(e) => set('indexing', 'exclusions', e.target.value.split('\n'))} />
          <div className="row" style={{ marginTop: 8 }}><button className="btn sm" onClick={() => set('indexing', 'exclusions', data.default_exclusions)}>Restore default exclusions</button>
            <button className="btn sm" onClick={async () => { try { if (dirty) await save(); const r = await api.rescanAll(); toast(`Rescanning ${r.queued} folder(s)`); } catch (e) { toast(errMsg(e), true); } }}><RefreshCw size={13} /> Apply &amp; rescan all</button></div></div>
      </div>

      <div className="card">
        <div className="card-head"><h2><ShieldCheck size={16} style={{ verticalAlign: -3 }} /> Privacy</h2></div>
        <div className="alert ok" style={{ marginBottom: 6 }}><ShieldCheck size={16} />Indexing, search and analysis run on this computer. The app only reads folders you connected and never modifies your files.</div>
        <div className="setting"><div><strong>Allow sending excerpts to an external model</strong><div className="muted small">Required before any provider outside this machine (e.g. Anthropic, OpenAI) receives short excerpts of your files. Local servers such as Ollama need no consent.</div></div><Switch checked={s.privacy.allow_external_llm} onChange={(v) => set('privacy', 'allow_external_llm', v)} /></div>
        <div className="setting"><div><strong>Keep assistant history</strong><div className="muted small">Stores your queries and results in the local database (Recommendations → History).</div></div>
          <div className="row"><Switch checked={s.privacy.log_queries} onChange={(v) => set('privacy', 'log_queries', v)} /><button className="btn sm danger" onClick={async () => { await api.clearHistory(); toast('History cleared'); }}><Trash2 size={13} /> Clear now</button></div></div>
      </div>

      <div className="card">
        <div className="card-head"><h2><Cpu size={16} style={{ verticalAlign: -3 }} /> Model</h2><span className={`badge ${llm.active ? 'ok' : ''}`}>{llm.active ? `Active: ${llm.provider}` : 'Not active'}</span></div>
        {!llm.active && <div className="alert info" style={{ marginBottom: 8 }}>{llm.reason} The assistant still works without a model (search, résumé checks, statistics and comparisons are computed locally).</div>}
        <div className="setting"><div><strong>Provider</strong><div className="muted small">Ollama / LM Studio use “OpenAI-compatible” with a localhost URL.</div></div>
          <select className="select" value={s.model.provider} onChange={(e) => set('model', 'provider', e.target.value)}><option value="none">None (deterministic)</option><option value="openai_compatible">OpenAI-compatible (local or hosted)</option><option value="anthropic">Anthropic</option></select></div>
        {s.model.provider !== 'none' && (
          <>
            <div className="setting"><div><strong>Model name</strong></div><input className="input" style={{ width: 280 }} value={s.model.model} onChange={(e) => set('model', 'model', e.target.value)} placeholder={s.model.provider === 'anthropic' ? 'claude-haiku-4-5-20251001' : 'llama3.1'} /></div>
            {s.model.provider === 'openai_compatible' && <div className="setting"><div><strong>Base URL</strong></div><input className="input mono" style={{ width: 280 }} value={s.model.base_url} onChange={(e) => set('model', 'base_url', e.target.value)} placeholder="http://localhost:11434/v1" /></div>}
            <div className="setting"><div><strong>API key</strong><div className="muted small">Kept in memory for this session only (never saved). Prefer the {s.model.provider === 'anthropic' ? 'ANTHROPIC_API_KEY' : 'DISCOVERY_LLM_API_KEY'} environment variable. {llm.has_key ? '✓ A key is available.' : 'No key detected.'}</div></div>
              <div className="row"><input className="input" type="password" style={{ width: 200 }} value={key} onChange={(e) => setKey(e.target.value)} placeholder="paste key" autoComplete="off" /><button className="btn sm" onClick={saveKey}>{key ? 'Use key' : 'Clear key'}</button></div></div>
            <div className="setting"><div><strong>Timeout / max tokens</strong></div><div className="row"><input className="input" type="number" style={{ width: 80 }} value={s.model.timeout_s} onChange={(e) => set('model', 'timeout_s', e.target.value)} /><input className="input" type="number" style={{ width: 90 }} value={s.model.max_tokens} onChange={(e) => set('model', 'max_tokens', e.target.value)} /></div></div>
            {llm.external && !s.privacy.allow_external_llm && <div className="alert warn" style={{ marginTop: 8 }}>This provider is outside your computer. Enable “Allow sending excerpts to an external model” in Privacy to use it.</div>}
          </>
        )}
        <div className="setting"><div><strong>Files analyzed per request</strong><div className="muted small">Excerpts per file / characters per excerpt shown as evidence.</div></div>
          <div className="row"><input className="input" type="number" style={{ width: 70 }} min={1} max={10} value={s.assistant.max_files_analyzed} onChange={(e) => set('assistant', 'max_files_analyzed', e.target.value)} />
            <input className="input" type="number" style={{ width: 70 }} min={1} max={6} value={s.assistant.excerpts_per_file} onChange={(e) => set('assistant', 'excerpts_per_file', e.target.value)} />
            <input className="input" type="number" style={{ width: 80 }} min={150} max={1500} value={s.assistant.excerpt_chars} onChange={(e) => set('assistant', 'excerpt_chars', e.target.value)} /></div></div>
      </div>

      <div className="card">
        <div className="card-head"><h2><Database size={16} style={{ verticalAlign: -3 }} /> Search engine</h2></div>
        <dl className="kv"><dt>Database</dt><dd>{data.database}</dd><dt>Embeddings</dt><dd>{emb ? `${emb.name} (${emb.dimension}-d)` : 'loading…'} {emb && <span className={`badge ${emb.semantic ? 'ok' : 'warn'}`}>{emb.semantic ? 'semantic' : 'not semantic'}</span>}</dd></dl>
        {emb && !emb.semantic && <div className="alert warn" style={{ marginTop: 8 }}>No sentence-embedding model is installed, so only keyword/metadata search is trusted. Run <span className="mono">pip install sentence-transformers</span> and restart for semantic search (changing the embedding model requires “Rebuild” on each folder).</div>}
      </div>
      {confirmReset && <Modal title="Reset settings?" onClose={() => setConfirmReset(false)} foot={<><button className="btn" onClick={() => setConfirmReset(false)}>Cancel</button><button className="btn danger" onClick={async () => { setConfirmReset(false); const r = await api.resetSettings(); setS(r.settings); setDirty(false); toast('Settings reset'); }}>Reset</button></>}>All settings return to their defaults. Connected folders and indexes are kept.</Modal>}
    </div>
  );
}
