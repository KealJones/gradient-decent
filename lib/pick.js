// Picking the next token with the layout engine instead of arithmetic.
//   Softmax: the candidates are boxes in one flex row with flex-grow = exp(score - best score), so flexbox
//            divides the row between them and each box's share of the width is its probability.
//   Sort:    flex `order` puts the biggest first. The boxes sit in the document in token order.
//   Top-p:   a marker sits at 80% of the row. The box under it is the last one kept (the nucleus).
//   Pick:    a needle is placed at a random fraction of the kept width and the browser is asked what's under it
//            (document.elementFromPoint).
// JavaScript chooses the candidates, takes one exp() per box and rolls the random number. It never adds the
// widths up, divides by a total or compares cumulative sums, unless the hit test can't be used (see edges()).
const el = (cls, parent) => { const e = document.createElement('div'); e.className = cls; parent.append(e); return e; };

export function flexPicker(host, label, { topK = 40 } = {}) {
  host.classList.add('pick');
  const row = el('pick-row', host), crop = el('pick-crop', host), shade = el('pick-shade', host), win = el('pick-win', host), needle = el('pick-needle', win);
  const boxes = Array.from({ length: topK }, () => el('pick-box', row)), ids = new Int32Array(topK), slot = new Map(boxes.map((b, i) => [b, i]));
  const names = boxes.map(b => b.appendChild(document.createElement('span')));   // labels, so the page can style or hide them
  const stats = { picks: 0, hit: 0, edges: 0, ms: 0 };
  let lit = null;

  // The plain way to find the box holding x: walk the box edges layout produced. Only used when the hit test
  // can't see the row (scrolled out of view, or under a dialog) or disagrees with the edges.
  function edges(x) {
    let far = null, farRight = -Infinity;
    for (const b of boxes) { const r = b.getBoundingClientRect(); if (!r.width) continue; if (r.left <= x && x < r.right) return b; if (r.right > farRight) { far = b; farRight = r.right; } }
    return far;   // x was past the end of the row
  }
  // Which box is at layout position x? Chrome's hit test takes whole-pixel coordinates and returns the last
  // box touching that pixel, so the row is nudged by a fraction of a pixel (never painted that way) to put x
  // just inside the pixel's far edge. That makes the hit test good to 1/64 px, the grid layout works on.
  function under(x, y) {
    const n = Math.floor(x); host.style.transform = `translateX(${n + 1 - x - 1 / 128}px)`;
    const b = document.elementFromPoint(n, y); host.style.transform = '';
    if (slot.has(b)) { const r = b.getBoundingClientRect(); if (r.left <= x && x < r.right) { stats.hit++; return b; } }
    stats.edges++; return edges(x);
  }

  // lg: a score for every token (-Infinity for blocked ones). r: where the needle goes, 0 to 1.
  function pick(lg, r = Math.random()) {
    const t0 = performance.now();
    // Bookkeeping: the topK highest scores, kept in token order so the sorting is left to flex `order`.
    const cut = Float32Array.from(lg).sort()[lg.length - topK]; let n = 0, max = -Infinity;
    for (let i = 0; i < lg.length && n < topK; i++) if (lg[i] > cut) ids[n++] = i;
    for (let i = 0; i < lg.length && n < topK; i++) if (lg[i] === cut && cut > -Infinity) ids[n++] = i;
    for (let i = 0; i < n; i++) if (lg[ids[i]] > max) max = lg[ids[i]];
    boxes.forEach((b, i) => {
      if (i >= n) { b.style.flexGrow = 0; names[i].textContent = ''; return; }
      const gap = max - lg[ids[i]];
      b.style.flexGrow = Math.exp(-gap); b.style.order = Math.round(Math.min(gap * 1e6, 2e9));
      const text = label(ids[i]); if (names[i].textContent !== text) names[i].textContent = text;
    });
    if (lit) lit.classList.remove('hit');
    const R = row.getBoundingClientRect(), y = Math.floor(R.top + R.height / 2);
    // Top-p: the box under the 80% marker is the last one in. The needle's run ends at that box's right edge.
    const last = under(crop.getBoundingClientRect().left, y), end = (last ? last.getBoundingClientRect().right : R.right) - R.left;
    win.style.width = shade.style.left = end + 'px';
    needle.style.left = r * 100 + '%';   // layout turns the fraction into a position
    const b = under(needle.getBoundingClientRect().left, y) || last || boxes[0];
    (lit = b).classList.add('hit');
    stats.picks++; stats.ms += performance.now() - t0;
    return ids[slot.get(b)];
  }
  return { pick, stats, row, boxes, ids };
}

// Check the layout pick against plain arithmetic. For each set of scores, n needle positions spread evenly from
// 0 to 1 are picked both ways: by layout, and by reference(scores, r), a cumulative sum in JavaScript.
// Also reads the box widths through a ResizeObserver and compares them with a real softmax.
export async function samplingTest(picker, sets, reference, n = 400) {
  picker.row.scrollIntoView({ block: 'center' });   // a hit test only works on screen
  const out = { agree: 0, of: 0, miss: [], tokensInPlay: sets.map(() => 0), rowPx: picker.row.getBoundingClientRect().width };
  const before = { ...picker.stats }, t0 = performance.now();
  for (const [s, lg] of sets.entries()) for (let i = 0, prev = -1; i < n; i++) {
    const r = (i + 0.5) / n, got = picker.pick(lg, r), want = reference(lg, r); out.of++;
    if (want !== prev) { out.tokensInPlay[s]++; prev = want; }   // how many different tokens the needle can reach
    if (got === want) out.agree++; else if (out.miss.length < 20) out.miss.push({ set: s, r, got, want });
  }
  out.msPerPick = (performance.now() - t0) / out.of; out.byHitTest = picker.stats.hit - before.hit; out.byEdges = picker.stats.edges - before.edges;
  // Widths as probabilities, read the odd way: a ResizeObserver reports every box's size when it starts watching.
  out.widthMaxErr = 0;
  for (const lg of sets) {
    picker.pick(lg, 0.5);
    const w = await new Promise(done => {
      const seen = new Map(), ro = new ResizeObserver(es => { for (const e of es) seen.set(e.target, e.contentRect.width); if (seen.size === picker.boxes.length) { ro.disconnect(); done(seen); } });
      picker.boxes.forEach(b => ro.observe(b)); setTimeout(() => { ro.disconnect(); done(null); }, 3000);   // a hidden tab never delivers them
    });
    if (!w) { out.widthMaxErr = null; break; }
    let total = 0, z = 0, max = -Infinity; for (const v of w.values()) total += v;
    for (const id of picker.ids) if (lg[id] > max) max = lg[id];
    for (const id of picker.ids) z += Math.exp(lg[id] - max);
    picker.boxes.forEach((b, i) => { out.widthMaxErr = Math.max(out.widthMaxErr, Math.abs(w.get(b) / total - Math.exp(lg[picker.ids[i]] - max) / z)); });
  }
  return out;
}
