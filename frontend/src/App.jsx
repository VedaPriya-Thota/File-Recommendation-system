import React, { useCallback, useEffect, useState } from 'react';
import { LayoutDashboard, Sparkles, Files as FilesIcon, ThumbsUp, FolderTree, Settings as Cog, Moon, Sun, Search, CheckCircle2, XCircle } from 'lucide-react';
import { api } from './api/client';
import Dashboard from './pages/Dashboard.jsx';
import Assistant from './pages/Assistant.jsx';
import FilesPage from './pages/Files.jsx';
import Recommendations from './pages/Recommendations.jsx';
import Folders from './pages/Folders.jsx';
import SettingsPage from './pages/Settings.jsx';
import FileDrawer from './components/FileDrawer.jsx';

const NAV = [
  ['dashboard', 'Overview', LayoutDashboard], ['assistant', 'AI Assistant', Sparkles], ['files', 'All Files', FilesIcon],
  ['recommendations', 'Recommendations', ThumbsUp], ['folders', 'Connected Folders', FolderTree], ['settings', 'Settings', Cog],
];
const pageFromHash = () => { const h = window.location.hash.replace('#/', '').split('?')[0]; return NAV.some((n) => n[0] === h) ? h : 'dashboard'; };

export default function App() {
  const [page, setPage] = useState(pageFromHash());
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') || (window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'));
  const [toasts, setToasts] = useState([]);
  const [preview, setPreview] = useState(null);          // { id, query }
  const [pendingAsk, setPendingAsk] = useState(null);     // text to prefill in the assistant
  const [counts, setCounts] = useState({ files: 0, scanning: 0 });

  useEffect(() => { document.documentElement.dataset.theme = theme; localStorage.setItem('theme', theme); }, [theme]);
  useEffect(() => { const h = () => setPage(pageFromHash()); window.addEventListener('hashchange', h); return () => window.removeEventListener('hashchange', h); }, []);
  const go = useCallback((p) => { window.location.hash = `#/${p}`; }, []);

  const toast = useCallback((msg, err = false) => {
    const id = Math.random();
    setToasts((t) => [...t, { id, msg, err }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), err ? 6000 : 3200);
  }, []);

  useEffect(() => {
    let alive = true;
    const tick = () => api.stats().then((s) => alive && setCounts({ files: s.files, scanning: s.active_scans.length })).catch(() => {});
    tick(); const t = setInterval(tick, 3000);
    return () => { alive = false; clearInterval(t); };
  }, []);

  const ask = (text) => { setPendingAsk({ text, n: Date.now() }); go('assistant'); };
  const ctx = { toast, go, openPreview: (id, query) => setPreview({ id, query }), ask };

  return (
    <div className="app">
      <nav className="sidebar" aria-label="Main">
        <div className="brand"><div className="logo"><Search size={18} /></div>IntelliFind</div>
        {NAV.map(([id, label, I]) => (
          <button key={id} className={`nav-item ${page === id ? 'active' : ''}`} onClick={() => go(id)} aria-current={page === id ? 'page' : undefined}>
            <I size={18} />{label}
            {id === 'files' && counts.files > 0 && <span className="count">{counts.files.toLocaleString()}</span>}
            {id === 'folders' && counts.scanning > 0 && <span className="count">scanning</span>}
          </button>
        ))}
        <button className="btn icon ghost topbar-mobile" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} aria-label="Toggle theme">{theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}</button>
        <div className="sidebar-foot">
          <div className="small muted">Local &amp; read-only. Your files never leave this computer unless you enable an external model.</div>
          <button className="btn" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}>{theme === 'dark' ? <Sun size={16} /> : <Moon size={16} />} {theme === 'dark' ? 'Light' : 'Dark'} theme</button>
        </div>
      </nav>
      <main className="main">
        {page === 'dashboard' && <Dashboard {...ctx} />}
        {page === 'assistant' && <Assistant {...ctx} pendingAsk={pendingAsk} />}
        {page === 'files' && <FilesPage {...ctx} />}
        {page === 'recommendations' && <Recommendations {...ctx} />}
        {page === 'folders' && <Folders {...ctx} />}
        {page === 'settings' && <SettingsPage {...ctx} />}
      </main>
      {preview && <FileDrawer fileId={preview.id} query={preview.query} onClose={() => setPreview(null)} toast={toast} onAsk={(f) => { setPreview(null); ask(`Summarize ${f.filename}`); }} />}
      <div className="toasts" role="status" aria-live="polite">{toasts.map((t) => <div key={t.id} className={`toast ${t.err ? 'err' : ''}`}>{t.err ? <XCircle size={16} /> : <CheckCircle2 size={16} />}{t.msg}</div>)}</div>
    </div>
  );
}
