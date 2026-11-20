import axios from 'axios';

const http = axios.create({ baseURL: import.meta.env.VITE_API_BASE || '', timeout: 120000 });

export const errMsg = (e) =>
  e?.response?.data?.detail ? (typeof e.response.data.detail === 'string' ? e.response.data.detail : JSON.stringify(e.response.data.detail))
    : e?.message || 'Request failed';

const get = (u, params) => http.get(u, { params }).then((r) => r.data);
const post = (u, body) => http.post(u, body).then((r) => r.data);

export const api = {
  health: () => get('/api/health'),
  embedding: () => get('/api/embedding'),
  stats: () => get('/api/stats'),
  folders: () => get('/api/folders'),
  addFolder: (body) => post('/api/folders', body),
  patchFolder: (id, body) => http.patch(`/api/folders/${id}`, body).then((r) => r.data),
  removeFolder: (id) => http.delete(`/api/folders/${id}`).then((r) => r.data),
  scan: (id) => post(`/api/folders/${id}/scan`),
  cancelScan: (id) => post(`/api/folders/${id}/cancel`),
  rebuild: (id) => post(`/api/folders/${id}/rebuild`),
  browse: (path) => get('/api/fs/browse', path ? { path } : {}),
  pick: () => post('/api/fs/pick'),
  files: (params) => get('/api/files', params),
  facets: () => get('/api/files/facets'),
  file: (id) => get(`/api/files/${id}`),
  preview: (id, q) => get(`/api/files/${id}/preview`, q ? { q } : {}),
  openFile: (id) => post(`/api/files/${id}/open`),
  reveal: (id) => post(`/api/files/${id}/reveal`),
  rawUrl: (id, inline = false) => `/api/files/${id}/raw${inline ? '?inline=true' : ''}`,
  settings: () => get('/api/settings'),
  saveSettings: (patch) => http.put('/api/settings', patch).then((r) => r.data),
  resetSettings: () => post('/api/settings/reset'),
  setKey: (provider, key) => post('/api/settings/llm-key', { provider, key }),
  rescanAll: () => post('/api/settings/rescan-all'),
  ask: (query, conversation_id, mode) => post('/api/assistant/runs', { query, conversation_id: conversation_id || undefined, mode: mode || undefined }),
  poll: (id, since) => get(`/api/assistant/runs/${id}`, { since }),
  history: (limit = 30) => get('/api/assistant/history', { limit }),
  clearHistory: () => http.delete('/api/assistant/history').then((r) => r.data),
};
