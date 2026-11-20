// @vitest-environment jsdom
// @vitest-environment-options {"url": "http://localhost:5173"}
// Smoke test against a RUNNING backend (set FDS_URL, default http://127.0.0.1:8000): renders every page with real data.
import React from 'react';
import { describe, it, expect, afterEach } from 'vitest';
import { render, screen, cleanup, waitFor, fireEvent } from '@testing-library/react';
import axios from 'axios';
import App from '../App.jsx';

// Run with: VITE_API_BASE=http://127.0.0.1:8000 npx vitest run


afterEach(cleanup);

const pages = ['dashboard', 'assistant', 'files', 'recommendations', 'folders', 'settings'];
describe('pages render with real backend data', () => {
  for (const p of pages) {
    it(p, async () => {
      window.location.hash = `#/${p}`;
      const errors = [];
      const orig = console.error; console.error = (...a) => { errors.push(a.join(' ')); };
      render(<App />);
      await waitFor(() => expect(document.querySelector('.page')).toBeTruthy(), { timeout: 8000 });
      await new Promise((r) => setTimeout(r, 1500));
      console.error = orig;
      expect(document.body.textContent).not.toContain('Something went wrong');
      expect(errors.filter((e) => !/act\(|not wrapped/.test(e))).toEqual([]);
    }, 20000);
  }
  it('files page lists indexed files and opens a preview', async () => {
    window.location.hash = '#/files';
    render(<App />);
    await waitFor(() => expect(screen.getAllByText(/login\.py/).length).toBeGreaterThan(0), { timeout: 8000 });
    fireEvent.click(screen.getAllByText(/login\.py/)[0]);
    await waitFor(() => expect(document.querySelector('.drawer')).toBeTruthy());
    await waitFor(() => expect(document.querySelector('.drawer pre.preview')?.textContent).toContain('authenticate_user'), { timeout: 8000 });
  }, 30000);
  it('assistant answers a real request with sources and a verified-evidence badge', async () => {
    window.location.hash = '#/assistant';
    render(<App />);
    const box = await screen.findByLabelText('Ask the assistant');
    fireEvent.change(box, { target: { value: 'Find my resume and suggest improvements' } });
    fireEvent.click(screen.getByText('Ask'));
    await waitFor(() => expect(document.body.textContent).toContain('Jane_Smith_Resume.docx'), { timeout: 20000 });
    await waitFor(() => expect(document.body.textContent).toMatch(/excerpts verified/), { timeout: 20000 });
    expect(document.body.textContent).toContain('Résumé review');
  }, 40000);
  it('grounded answer shows a computed result, numbered sources and the coverage line', async () => {
    window.location.hash = '#/assistant';
    render(<App />);
    const box = await screen.findByLabelText('Ask the assistant');
    fireEvent.change(box, { target: { value: 'What is the total revenue in the weekly sales data?' } });
    fireEvent.click(screen.getByText('Ask'));
    await waitFor(() => expect(document.body.textContent).toContain('1,450,371'), { timeout: 20000 });
    expect(document.body.textContent).toMatch(/Searched \d+ file/);
    expect(document.querySelector('.src-num')?.textContent).toBe('1');
    expect(document.body.textContent).toContain('calculated from the data');
  }, 40000);
  it('agent activity panel lists the agents and tools that really ran', async () => {
    window.location.hash = '#/assistant';
    render(<App />);
    const box = await screen.findByLabelText('Ask the assistant');
    fireEvent.change(box, { target: { value: 'Which month has the highest revenue in the weekly sales data?' } });
    fireEvent.click(screen.getByText('Ask'));
    await waitFor(() => expect(document.querySelector('.agent-activity')).toBeTruthy(), { timeout: 20000 });
    await waitFor(() => expect(document.body.textContent).toMatch(/Sources/), { timeout: 20000 });
    fireEvent.click(document.querySelector('.agent-head'));
    await waitFor(() => expect(document.querySelector('.agent-body')).toBeTruthy());
    const txt = document.querySelector('.agent-activity').textContent;
    for (const a of ['Orchestrator', 'File Discovery', 'Retrieval', 'Analysis', 'Verification', 'Response']) expect(txt).toContain(a);
    expect(txt).toContain('analyze_table');
    expect(txt).toMatch(/\d+ tool calls?/);
  }, 40000);
});
