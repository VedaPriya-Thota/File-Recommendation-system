import React, { useEffect, useRef, useState } from 'react';
import { Send, Sparkles, FolderPlus } from 'lucide-react';
import { api, errMsg } from '../api/client';
import AnswerView, { AgentActivity } from '../components/AnswerView.jsx';
import { Empty, Spinner } from '../components/ui.jsx';

export function useAsk(mode) {
  const [runs, setRuns] = useState([]);   // newest first: {id, query, events, result, error, running}
  const [convId, setConvId] = useState(null);
  const convRef = useRef(null);
  const timers = useRef({});
  const reset = () => { convRef.current = null; setConvId(null); setRuns([]); };
  const ask = async (query) => {
    const local = { key: Math.random(), query, events: [], result: null, error: '', running: true };
    setRuns((r) => [local, ...r]);
    const patch = (p) => setRuns((r) => r.map((x) => (x.key === local.key ? { ...x, ...p } : x)));
    try {
      const { run_id } = await api.ask(query, mode === 'find' ? null : convRef.current, mode);
      let since = 0, events = [];
      const poll = async () => {
        try {
          const d = await api.poll(run_id, since);
          since = d.next; events = [...events, ...d.events];
          patch({ events });
          if (d.status === 'running') { timers.current[local.key] = setTimeout(poll, 450); return; }
          if (d.result?.conversation_id) { convRef.current = d.result.conversation_id; setConvId(d.result.conversation_id); }
          patch({ running: false, result: d.result, error: d.error || '' });
        } catch (e) { patch({ running: false, error: errMsg(e) }); }
      };
      poll();
    } catch (e) { patch({ running: false, error: errMsg(e) }); }
  };
  useEffect(() => () => Object.values(timers.current).forEach(clearTimeout), []);
  return [runs, ask, { convId, reset }];
}

export default function Assistant({ go, toast, openPreview, pendingAsk }) {
  const [text, setText] = useState('');
  const [runs, ask, conv] = useAsk();
  const [stats, setStats] = useState(null);
  const handled = useRef(null);
  useEffect(() => { api.stats().then(setStats).catch(() => {}); }, []);
  useEffect(() => { if (pendingAsk && handled.current !== pendingAsk.n) { handled.current = pendingAsk.n; ask(pendingAsk.text); } }, [pendingAsk]);

  const submit = () => { const q = text.trim(); if (!q) return; setText(''); ask(q); };
  const suggestions = [];
  if (stats) {
    const types = new Set(stats.types.map((t) => t.t));
    stats.recent.slice(0, 1).forEach((f) => suggestions.push(`Summarize ${f.filename}`));
    if (['docx', 'pdf'].some((t) => types.has(t))) suggestions.push('Find my resume and suggest improvements');
    if (['csv', 'xlsx'].some((t) => types.has(t))) suggestions.push('Analyze my spreadsheets and identify trends');
    if (['py', 'js', 'ts', 'java', 'go'].some((t) => types.has(t))) suggestions.push('Find source code that implements authentication');
    if (types.has('md') || types.has('py')) suggestions.push('What does my project do?');
  }
  const noIndex = stats && stats.indexed === 0;

  return (
    <div className="page">
      <div className="page-head"><div><h1>AI Assistant</h1><p>Ask about your files. Answers cite the files and passages they come from, and follow-up questions keep their context.</p></div>
        {runs.length > 0 && <button className="btn" onClick={conv.reset}><Sparkles size={15} /> New conversation</button>}</div>
      {noIndex && (
        <div className="card"><Empty icon={FolderPlus} title={stats.folders ? 'Still indexing…' : 'No folders connected'} text={stats.folders ? 'Nothing is searchable yet. This page will work as soon as the first files are indexed.' : 'Connect a folder first so there is something to search.'}>
          <button className="btn primary" onClick={() => go('folders')}>Open Connected Folders</button></Empty></div>
      )}
      <div className="card">
        <div className="ask">
          <textarea className="textarea" value={text} onChange={(e) => setText(e.target.value)} placeholder={runs.length ? 'Ask a follow-up, e.g. “What are its main limitations?”' : 'e.g. “What does the contract say about termination?”, “Summarize the annual report”, “What is the total revenue by region?”'} rows={2}
            onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submit(); } }} aria-label="Ask the assistant" />
          <button className="btn primary" onClick={submit} disabled={!text.trim() || noIndex}><Send size={16} /> Ask</button>
        </div>
        {suggestions.length > 0 && <div className="row wrap" style={{ marginTop: 12 }}>{suggestions.map((s) => <button key={s} className="chip" onClick={() => ask(s)} disabled={noIndex}>{s}</button>)}</div>}
      </div>
      {runs.length === 0 && !noIndex && <div className="card"><Empty icon={Sparkles} title="Ask anything about your files" text="The assistant searches your indexed folders, reads the matching passages, analyzes data files, and verifies its evidence before answering." /></div>}
      {runs.map((r) => (
        <div key={r.key} className="col">
          <div className="row"><strong style={{ fontSize: 15 }}>“{r.query}”</strong>{r.running && <Spinner />}</div>
          {r.running && <AgentActivity events={r.events} running defaultOpen />}
          {r.error && <div className="alert err">{r.error}</div>}
          {r.result && <AnswerView result={r.result} onPreview={(id) => openPreview(id, r.result.retrieval_query)} toast={toast} onAsk={ask} />}
        </div>
      ))}
    </div>
  );
}
