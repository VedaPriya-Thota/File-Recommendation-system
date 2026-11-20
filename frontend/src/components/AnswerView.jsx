import React, { useState } from 'react';
import { ChevronDown, ChevronUp, Info, Layers, RefreshCw } from 'lucide-react';
import { CheckCircle2, AlertTriangle, Loader2, Search, Brain, FileSearch, BarChart3, ShieldCheck, Cpu, Wrench, Eye, ExternalLink, FolderOpen, Copy, Bot, XCircle } from 'lucide-react';
import { api, errMsg } from '../api/client';
import { FileIcon, Markdown, Highlight, TypeBadge, fmtBytes } from './ui.jsx';

const KIND = { understand: Brain, agent: Bot, tool: Wrench, retrieve: FileSearch, analysis: BarChart3, verify: ShieldCheck, llm: Cpu };

const AGENT_STYLE = {
  Orchestrator: ['#7c3aed', 'Plans & coordinates'], 'File Discovery Agent': ['#0ea5e9', 'Finds the right files'], 'Retrieval Agent': ['#14b8a6', 'Pulls out the specifics'],
  'Analysis Agent': ['#f59e0b', 'Analyses & calculates'], 'Verification Agent': ['#ef4444', 'Checks the evidence'], 'Response Agent': ['#22c55e', 'Writes the answer'],
};
const ICON = { plan: Brain, handoff: Layers, retry: RefreshCw, agent_start: Bot, agent_done: CheckCircle2, tool: Wrench, llm: Cpu, verify: ShieldCheck, retrieve: FileSearch, analysis: BarChart3, agent: Bot, understand: Brain };

function Chip({ agent, small }) {
  const [c] = AGENT_STYLE[agent] || ['#64748b'];
  return <span className="agent-pill" style={{ background: `${c}22`, color: c, borderColor: `${c}55`, fontSize: small ? 11 : 12 }}>{agent.replace(' Agent', '')}</span>;
}

export function metricsOf(events, metrics) {
  if (metrics) return metrics;
  const agents = [...new Set(events.map((e) => e.agent).filter(Boolean))];
  return { agents, tool_calls: events.filter((e) => e.kind === 'tool' && e.tool).length, llm_calls: events.filter((e) => e.tool === 'llm').length, rounds: 1 };
}

/** Real execution events: which agent did what, with which tool, and whether it worked. Collapsed by default. */
export function AgentActivity({ events, metrics, running, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen || running);
  const [tech, setTech] = useState(false);
  const m = metricsOf(events, metrics);
  const ev = events.filter((e) => tech || !['agent_start'].includes(e.kind));
  const failed = events.filter((e) => e.status === 'error').length;
  return (
    <div className="card flat agent-activity">
      <button className="agent-head" onClick={() => setOpen(!open)} aria-expanded={open}>
        <Bot size={16} /> <strong>Agent activity</strong>
        <span className="agent-chips">{(m.agents || []).map((a) => <Chip key={a} agent={a} small />)}</span>
        <span className="grow" />
        <span className="muted small">{m.tool_calls} tool call{m.tool_calls === 1 ? '' : 's'} · {m.llm_calls} model call{m.llm_calls === 1 ? '' : 's'}{m.rounds > 1 ? ` · ${m.rounds} rounds` : ''}{m.parallel_batches ? ` · ${m.parallel_batches} parallel` : ''}{failed ? ` · ${failed} issue${failed > 1 ? 's' : ''}` : ''}</span>
        {open ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
      </button>
      {open && (
        <div className="agent-body">
          <label className="row small muted" style={{ justifyContent: 'flex-end' }}><input type="checkbox" checked={tech} onChange={(e) => setTech(e.target.checked)} /> Technical details</label>
          <div className="timeline">
            {ev.map((e, i) => {
              const I = ICON[e.kind] || Search;
              const last = running && i === ev.length - 1;
              const st = last ? 'running' : e.status;
              const showAgent = i === 0 || ev[i - 1].agent !== e.agent;
              return (
                <div key={i}>
                  {showAgent && <div className="agent-lane"><Chip agent={e.agent || 'Orchestrator'} /><span className="muted small">{(AGENT_STYLE[e.agent] || [])[1]}</span></div>}
                  <div className="tl-item">
                    <div className={`dot ${st}`}>{last ? <Loader2 size={12} className="spin" /> : st === 'error' ? <XCircle size={12} /> : st === 'warning' ? <AlertTriangle size={12} /> : <I size={12} />}</div>
                    <div>
                      <div style={{ fontWeight: 600 }}>{e.tool && e.kind === 'tool' && <code className="tool-chip">{e.tool}</code>} {e.title}{e.cached ? <span className="badge" style={{ marginLeft: 6 }}>cached</span> : null}</div>
                      {e.detail && <div className="muted small">{e.detail}</div>}
                      {e.plan && (
                        <div className="plan-chain">{e.plan.map((t) => <span key={t.id} className={`plan-step ${t.status}`} title={t.goal}><Chip agent={({ discovery: 'File Discovery Agent', retrieval: 'Retrieval Agent', analysis: 'Analysis Agent', verification: 'Verification Agent', response: 'Response Agent' })[t.agent]} small /> {t.op.replace(/_/g, ' ')}</span>)}</div>
                      )}
                      {tech && (e.ms != null || e.args) && <div className="muted small mono">{e.ms != null ? `${e.ms} ms` : ''} {e.args ? JSON.stringify(e.args) : ''}</div>}
                    </div>
                    <div className="muted small mono">{e.t?.toFixed?.(2)}s</div>
                  </div>
                </div>
              );
            })}
            {running && !ev.length && <div className="muted"><Loader2 size={14} className="spin" /> Starting…</div>}
          </div>
        </div>
      )}
    </div>
  );
}

export function ActivityLog({ events, running }) {
  return <AgentActivity events={events} running={running} defaultOpen />;
}

export function RefCard({ r, onPreview, toast, id }) {
  const act = async (fn, ok) => { try { await fn(); toast(ok); } catch (e) { toast(errMsg(e), true); } };
  return (
    <div className="card flat ref" id={id}>
      <div className="row">
        <FileIcon type={r.file_type} />
        <div className="grow" style={{ minWidth: 0 }}>
          <div className="row wrap" style={{ gap: 6 }}><strong className="trunc">{r.filename}</strong><span className="cite">{r.ref}</span><TypeBadge t={r.file_type} />{r.language && <span className="badge info">{r.language}</span>}
            {r.stale && <span className="badge warn" title={r.stale}>changed on disk</span>}{r.injection_suspected && <span className="badge err">instruction-like text</span>}</div>
          <div className="muted small trunc" title={r.abs_path}>{r.folder_label} › {r.rel_path} · {fmtBytes(r.size_bytes)}</div>
        </div>
        <div className="row" title="Relevance"><span className="rel"><i style={{ width: `${Math.round(r.relevance * 100)}%` }} /></span><span className="small muted">{Math.round(r.relevance * 100)}%</span></div>
      </div>
      <div className="small"><strong>Why it matches:</strong> {r.reason}</div>
      {r.duplicates?.length > 0 && <div className="small muted">{r.duplicates.length} identical cop{r.duplicates.length > 1 ? 'ies' : 'y'} elsewhere were collapsed into this result.</div>}
      {r.excerpts.map((e, i) => (
        <div key={i} className={`excerpt ${r.language ? 'code' : ''}`}>
          {e.line_start && <div className="muted small" style={{ marginBottom: 4 }}>lines {e.line_start}–{e.line_end}</div>}
          <Highlight text={e.text} terms={e.matched_terms} />
        </div>
      ))}
      <div className="row wrap">
        <button className="btn sm" onClick={() => onPreview(r.file_id)}><Eye size={14} /> Preview</button>
        <button className="btn sm" onClick={() => act(() => api.openFile(r.file_id), 'Opened with your default app')}><ExternalLink size={14} /> Open</button>
        <button className="btn sm" onClick={() => act(() => api.reveal(r.file_id), 'Opened containing folder')}><FolderOpen size={14} /> Show in folder</button>
        <button className="btn sm" onClick={() => act(() => navigator.clipboard.writeText(r.abs_path), 'Path copied')}><Copy size={14} /> Copy path</button>
      </div>
    </div>
  );
}

function Analyses({ a }) {
  return (
    <>
      {a.resume && (
        <div className="card flat">
          <div className="card-head"><h3>Résumé review</h3><span className={`badge ${a.resume.score >= 75 ? 'ok' : a.resume.score >= 50 ? 'warn' : 'err'}`}>{a.resume.score}/100</span></div>
          {a.resume.checks.map((c, i) => (
            <div className="check" key={i}>{c.ok ? <CheckCircle2 size={18} color="var(--ok)" /> : <AlertTriangle size={18} color="var(--warn)" />}
              <div><strong>{c.title}</strong> <span className="muted">— {c.detail}</span>{c.evidence?.map((x, j) => <div key={j} className="excerpt" style={{ maxHeight: 60 }}>{x}</div>)}</div></div>
          ))}
        </div>
      )}
      {a.tables?.map((t) => t.sheets.map((s) => (
        <div className="card flat" key={t.ref + s.sheet}>
          <div className="card-head"><h3>{t.filename} · {s.sheet}</h3><span className="badge">{s.rows.toLocaleString()} rows</span></div>
          <div className="table-wrap"><table className="t"><thead><tr><th>Column</th><th>Type</th><th>Min</th><th>Mean</th><th>Max</th><th>Missing</th></tr></thead>
            <tbody>{s.columns.map((c) => <tr key={c.name} style={{ cursor: 'default' }}><td>{c.name}</td><td><span className="badge">{c.type}</span></td>
              <td>{c.min ?? '—'}</td><td>{typeof c.mean === 'number' ? c.mean.toLocaleString(undefined, { maximumFractionDigits: 2 }) : '—'}</td><td>{c.max ?? '—'}</td><td>{c.missing}</td></tr>)}</tbody></table></div>
        </div>
      )))}
    </>
  );
}

function Excerpt({ e, code }) {
  const hl = e.highlight && e.highlight[1] > e.highlight[0] ? e.highlight : null;
  return (
    <div className={`excerpt ${code ? 'code' : ''}`}>
      <span className="loc">{e.location}{e.verified === false ? ' · could not be re-verified' : ''}</span>
      {hl ? <>{e.text.slice(0, hl[0])}<mark>{e.text.slice(hl[0], hl[1])}</mark>{e.text.slice(hl[1])}</> : e.text}
    </div>
  );
}

export function SourceCard({ r, runId, onPreview, toast }) {
  const [more, setMore] = useState(false);
  const act = async (fn, ok) => { try { await fn(); toast(ok); } catch (e) { toast(errMsg(e), true); } };
  const ex = more ? r.excerpts : r.excerpts.slice(0, 1);
  return (
    <div className="card flat ref" id={`ref-${runId}-${r.ref}`} style={r.closest_only ? { opacity: .85, borderStyle: 'dashed' } : null}>
      <div className="src-head">
        <span className="src-num">{r.ref}</span><FileIcon type={r.file_type} />
        <div className="grow" style={{ minWidth: 0 }}>
          <div className="row wrap" style={{ gap: 6 }}><strong className="trunc">{r.filename}</strong>
            {r.computed && <span className="badge info">calculated from the data</span>}
            {r.closest_only && <span className="badge warn">closest match — does not answer the question</span>}
            {r.stale && <span className="badge warn" title={r.stale}>changed on disk</span>}
            {r.injection_suspected && <span className="badge err">instruction-like text ignored</span>}</div>
          <div className="muted small trunc" title={r.abs_path}>{r.folder_label} › {r.rel_path} · {fmtBytes(r.size_bytes)}</div>
        </div>
      </div>
      {r.reason && <div className="small muted">{r.reason}</div>}
      {ex.map((e, i) => <Excerpt key={i} e={e} code={!!r.language} />)}
      <div className="row wrap">
        {r.excerpts.length > 1 && <button className="btn sm ghost" onClick={() => setMore(!more)}>{more ? <ChevronUp size={14} /> : <ChevronDown size={14} />} {more ? 'Show fewer' : `${r.excerpts.length - 1} more excerpt${r.excerpts.length > 2 ? 's' : ''}`}</button>}
        <div className="grow" />
        <button className="btn sm" onClick={() => onPreview(r.file_id)}><Eye size={14} /> Preview</button>
        <button className="btn sm" onClick={() => act(() => api.openFile(r.file_id), 'Opened with your default app')}><ExternalLink size={14} /> Open</button>
        <button className="btn sm" onClick={() => act(() => api.reveal(r.file_id), 'Opened containing folder')}><FolderOpen size={14} /> Folder</button>
        <button className="btn sm" onClick={() => act(() => navigator.clipboard.writeText(r.abs_path), 'Path copied')}><Copy size={14} /> Path</button>
      </div>
    </div>
  );
}

function QAAnswer({ result, onPreview, toast, onAsk }) {
  const [showLog, setShowLog] = useState(false);
  const cite = (ref) => document.getElementById(`ref-${result.run_id}-${ref}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  const v = result.verification || {};
  const cov = result.coverage || {};
  const calc = result.analyses?.calc;
  const st = result.status;
  const cls = { insufficient: 'info', no_index: 'info', clarify: 'info', error: 'err', partial: 'warn' }[st];
  return (
    <div className="col" style={{ gap: 14 }}>
      <div className="card">
        <div className="row wrap" style={{ marginBottom: 8, gap: 6 }}>
          {result.followup && <span className="badge primary">follow-up</span>}
          {st === 'ok' && v.supported > 0 && <span className="badge ok"><ShieldCheck size={12} /> {v.supported} of {v.sentences_checked} statements checked against the files</span>}
          {st === 'ok' && v.sentences_checked === 0 && v.note && <span className="badge ok"><ShieldCheck size={12} /> grounded in the files</span>}
          {st === 'partial' && <span className="badge warn">partly answered</span>}
          {st === 'insufficient' && <span className="badge warn">not found in your files</span>}
          {result.llm?.used ? <span className={`badge ${result.llm.external ? 'warn' : ''}`}>written by {result.llm.model || result.llm.provider}</span> : <span className="badge">no language model used</span>}
          <span className="badge">{result.duration_s}s</span>
        </div>
        {cls && st !== 'ok' && st !== 'partial' ? <div className={`alert ${cls}`}><Info size={16} style={{ flex: 'none', marginTop: 2 }} /><div className="grow"><Markdown text={result.answer} onCite={cite} /></div></div> : <Markdown text={result.answer} onCite={cite} />}
        {result.clarify_options?.length > 0 && <div className="row wrap" style={{ marginTop: 8 }}>{result.clarify_options.map((o) => <button key={o} className="chip" onClick={() => onAsk(`Summarize ${o}`)}>{o}</button>)}</div>}
        {calc && (
          <div className="calc-table"><table><thead><tr>{calc.table[0].map((h, i) => <th key={i}>{h}</th>)}</tr></thead>
            <tbody>{calc.table.slice(1, 25).map((r, i) => <tr key={i}>{r.map((c, j) => <td key={j}>{c}</td>)}</tr>)}</tbody></table></div>
        )}
        {result.warnings?.map((w, i) => <div key={i} className="alert warn" style={{ marginTop: 8 }}><AlertTriangle size={16} style={{ flex: 'none', marginTop: 2 }} />{w}</div>)}
        <div className="cov">
          {cov.files_searched != null && <span>Searched {cov.files_searched.toLocaleString()} file{cov.files_searched === 1 ? '' : 's'}{cov.passages_searched ? ` · ${cov.passages_searched.toLocaleString()} passages` : ''}{cov.rows_used ? ` · ${cov.rows_used.toLocaleString()} rows` : ''}</span>}
          {cov.files_indexed != null && <span>{cov.files_indexed.toLocaleString()} indexed in total</span>}
          {cov.files_failed > 0 && <span style={{ color: 'var(--warn)' }}>{cov.files_failed} could not be read</span>}
          {v.excerpts_checked > 0 && <span>{v.excerpts_checked - (v.excerpts_failed || 0)}/{v.excerpts_checked} excerpts match the files</span>}
        </div>
      </div>
      {result.references.length > 0 && <h2>{st === 'insufficient' ? 'Closest material' : 'Sources'}</h2>}
      {result.references.map((r) => <SourceCard key={r.ref} r={r} runId={result.run_id} onPreview={onPreview} toast={toast} />)}
      <AgentActivity events={result.activity} metrics={result.agents} />
    </div>
  );
}

function FindAnswer({ result, onPreview, toast }) {
  const [showLog, setShowLog] = useState(false);
  const cite = (ref) => document.getElementById(`ref-${result.run_id}-${ref}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  const v = result.verification || {};
  return (
    <div className="col" style={{ gap: 14 }}>
      <div className="card">
        <div className="row wrap" style={{ marginBottom: 8, gap: 6 }}>
          <span className="badge primary">{result.intent.replace(/_/g, ' ')}</span>
          <span className="badge">{result.duration_s}s</span>
          {result.llm?.used ? <span className={`badge ${result.llm.external ? 'warn' : 'ok'}`}>{result.llm.external ? 'External model' : 'Local model'}: {result.llm.model || result.llm.provider}</span> : <span className="badge">Deterministic · no model call</span>}
          {v.excerpts_checked > 0 && <span className="badge ok"><ShieldCheck size={12} /> {v.excerpts_checked - v.excerpts_failed}/{v.excerpts_checked} excerpts verified</span>}
        </div>
        <Markdown text={result.answer} onCite={cite} />
        {v.unverified_quotes?.length > 0 && <div className="alert warn" style={{ marginTop: 8 }}><AlertTriangle size={16} />The model quoted text that is not verbatim in the files: treat those quotes as unverified.</div>}
      </div>
      <Analyses a={result.analyses || {}} />
      {result.references.length > 0 && <h2>Sources &amp; evidence</h2>}
      {result.references.map((r) => <RefCard key={r.ref} id={`ref-${result.run_id}-${r.ref}`} r={r} onPreview={onPreview} toast={toast} />)}
      <AgentActivity events={result.activity} metrics={result.agents} />
    </div>
  );
}

export default function AnswerView({ result, onPreview, toast, onAsk }) {
  return result.mode === 'qa'
    ? <QAAnswer result={result} onPreview={onPreview} toast={toast} onAsk={onAsk} />
    : <FindAnswer result={result} onPreview={onPreview} toast={toast} />;
}
