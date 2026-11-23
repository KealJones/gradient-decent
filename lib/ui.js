// Page pieces shared by the canvas projects.
import { race } from './llama.js?v=6';
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };

// A small live tile for one matrix; returns the 2D context the model draws its product into.
export function tileMaker(grid) {
  return (key, label) => { const t = el('div', 'tile'), c = el('canvas'); c.width = c.height = 64; t.append(c, el('div', '', label)); grid.append(t); return c.getContext('2d'); };
}

// Wire a "race" button to four metric slots: canvas time, JavaScript time, how much slower, and agreement.
export function wireRace(button, getModel, getIds, slots) {
  button.addEventListener('click', async () => {
    const M = getModel(); if (!M) return;
    button.disabled = true; button.textContent = 'Racing…'; await new Promise(r => setTimeout(r, 30));
    const r = await race(M, getIds());
    slots.canvas.textContent = r.canvasMs.toFixed(1) + ' ms';
    slots.js.textContent = r.jsMs < 1 ? (r.jsMs * 1000).toFixed(0) + ' µs' : r.jsMs.toFixed(2) + ' ms';
    slots.x.textContent = Math.round(r.slower) + '×';
    slots.ok.textContent = `${r.agree}/${r.of} · ${r.maxLogitGap.toFixed(3)}`;
    button.disabled = false; button.textContent = 'Race again';
  });
}

// The weight pictures themselves: rows of matrices drawn at a shared scale, click to zoom.
//   rows: [{ label, items: [{ key, label, turn }] }]   turn draws a tall matrix on its side
export function showPictures(M, box, rows) {
  const dlg = el('dialog'), bar = el('div', 'bar'), title = el('span'), close = el('button', 'ghost', 'Close'), scroll = el('div', 'scroll'), zoom = el('canvas');
  close.type = 'button'; bar.append(title, close); scroll.append(zoom); dlg.append(bar, scroll); document.body.append(dlg);
  close.addEventListener('click', () => dlg.close()); dlg.addEventListener('click', e => { if (e.target === dlg) dlg.close(); });
  for (const row of rows) {
    const r = el('div', 'prow'); box.append(el('div', 'plab', row.label), r);
    for (const it of row.items) {
      const b = M.mats[it.key], { n, m } = b, c = el('canvas'), g = (c.width = it.turn ? n : m, c.height = it.turn ? m : n, c.getContext('2d'));
      // Only the real columns (the picture is padded to a power of two).
      if (it.turn) { g.translate(0, m); g.rotate(-Math.PI / 2); }
      g.drawImage(b.src, 0, 0, m, n, 0, 0, m, n);
      const f = el('figure'), btn = el('button');
      f.style.flex = `${c.width} 0 0`; f.style.maxWidth = c.width * 3 + 'px'; btn.type = 'button'; btn.setAttribute('aria-label', `${it.label}, ${n} by ${m} weights, open larger`);
      btn.addEventListener('click', () => { const k = Math.max(1, Math.min(4, Math.floor(900 / c.width))); zoom.width = c.width; zoom.height = c.height; zoom.getContext('2d').drawImage(c, 0, 0); zoom.style.width = c.width * k + 'px';
        title.textContent = `${row.label}, ${it.label}: ${n} rows × ${m} columns, ${(n * m).toLocaleString()} weights${it.turn ? ', turned on its side' : ''}`; dlg.showModal(); });
      btn.append(c); f.append(btn, el('figcaption', '', `${it.label} ${n}×${m}`)); r.append(f);
    }
  }
}
export const LAYER_ITEMS = [['q', 'query'], ['k', 'key'], ['v', 'value'], ['o', 'attention out'], ['gate', 'gate'], ['up', 'up'], ['down', 'down']];
export const layerRows = M => Array.from({ length: M.L }, (_, l) => ({ label: `Layer ${l + 1}`, items: LAYER_ITEMS.map(([k, label]) => ({ key: l + '.' + k, label })) }));
