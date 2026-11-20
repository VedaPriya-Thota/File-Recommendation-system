import React, { useEffect, useState } from 'react';
import { ThumbsUp, Search, History, Trash2 } from 'lucide-react';
import { api, errMsg } from '../api/client';
import { Empty, Spinner, timeAgo, Skeletons } from '../components/ui.jsx';
import { RefCard, ActivityLog } from '../components/AnswerView.jsx';
import { useAsk } from './Assistant.jsx';

export default function Recommendations({ toast, openPreview, go }) {
  const [text, setText] = useState('');
  const [runs, ask] = useAsk('find');
  const [hist, setHist] = useState(null);
  const [stats, setStats] = useState(null);
  const [loaded, setLoaded] = useState(null);
  const loadHist = () => api.history(30).then(setHist).catch(() => setHist([]));
  useEffect(() => { loadHist(); api.stats().then(setStats).catch(() => {}); }, []);
  useEffect(() => { if (runs[0] && !runs[0].running) loadHist(); }, [runs[0]?.running]);

  const open = async (id) => { try { const d = await api.poll(id, 0); setLoaded(d.result); } catch (e) { toast(errMsg(e), true); } };
  const submit = () => { const q = text.trim(); if (!q) return; setLoaded(null); setText(''); ask(q); };
  const current = loaded || runs[0]?.result;
  const running = !loaded && runs[0]?.running;
  const ideas = stats ? stats.recent.slice(0, 3).map((f) => f.filename.replace(/\.[^.]+$/, '').replace(/[_-]+/g, ' ')) : [];
  const clear = async () => { await api.clearHistory(); setHist([]); toast('History cleared'); };

  return (
    <div className="page">
      <div className="page-head"><div><h1>Recommendations</h1><p>Describe what you need — you get the most relevant files, why they match, and the passages that prove it.</p></div></div>
      <div className="card">
        <div className="row"><div className="search"><Search size={16} /><input className="input" value={text} onChange={(e) => setText(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && submit()} placeholder="What are you working on? e.g. “quarterly budget”, “onboarding docs”…" aria-label="Recommendation query" /></div>
          <button className="btn primary" onClick={submit} disabled={!text.trim() || (stats && stats.indexed === 0)}><ThumbsUp size={16} /> Recommend</button></div>
        {ideas.length > 0 && <div className="row wrap" style={{ marginTop: 10 }}><span className="small muted">Based on your recent files:</span>{ideas.map((i) => <button key={i} className="chip" onClick={() => { setLoaded(null); ask(i); }}>{i}</button>)}</div>}
      </div>
      <div className="grid two">
        <div className="col">
          {running && <div className="card"><h3 style={{ marginBottom: 6 }}>Working…</h3><ActivityLog events={runs[0].events} running /></div>}
          {!running && runs[0]?.error && !loaded && <div className="alert err">{runs[0].error}</div>}
          {current && (
            <>
              <div className="card"><div className="small muted">Results for</div><h2>“{current.query}”</h2>
                <div className="small muted" style={{ marginTop: 4 }}>{current.references.length} file(s) · {current.duration_s}s · {current.verification?.excerpts_checked ?? 0} excerpts verified</div>
                {current.warnings?.map((w, i) => <div key={i} className="alert warn" style={{ marginTop: 8 }}>{w}</div>)}
                {current.status === 'no_results' && <div className="alert info" style={{ marginTop: 8 }}>No indexed file matched. Try other keywords.</div>}</div>
              {current.references.map((r) => <RefCard key={r.ref} r={r} onPreview={(id) => openPreview(id, current.retrieval_query)} toast={toast} />)}
            </>
          )}
          {!current && !running && <div className="card"><Empty icon={ThumbsUp} title="No recommendations yet" text={stats && stats.indexed === 0 ? 'Connect and index a folder first.' : 'Describe a topic above, or pick a previous search from the history.'}>
            {stats && stats.indexed === 0 && <button className="btn primary" onClick={() => go('folders')}>Connect a folder</button>}</Empty></div>}
        </div>
        <div className="card" style={{ alignSelf: 'start' }}>
          <div className="card-head"><h2><History size={15} style={{ verticalAlign: -2 }} /> History</h2>{hist?.length > 0 && <button className="btn sm ghost danger" onClick={clear}><Trash2 size={13} /> Clear</button>}</div>
          {hist === null ? <Skeletons n={3} /> : hist.length === 0 ? <div className="muted small">Your past searches and recommendations appear here (stored only on this computer; disable in Settings → Privacy).</div> : (
            <div className="col" style={{ gap: 6 }}>{hist.map((h) => (
              <button key={h.run_id} className="nav-item" style={{ display: 'block' }} onClick={() => open(h.run_id)}>
                <div className="trunc" style={{ color: 'var(--text)', fontWeight: 600 }}>{h.query}</div>
                <div className="small muted trunc">{timeAgo(h.created_at)} · {h.n_refs} result(s){h.top.length ? ` · ${h.top[0]}` : ''}</div></button>))}</div>
          )}
        </div>
      </div>
    </div>
  );
}
