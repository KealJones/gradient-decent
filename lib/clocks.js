// Rotary positions done by CSS transforms, on clock faces.
// One face per rotation frequency. Each face holds a rotor element with `rotate: <pos × freq>rad`, and each
// pair of numbers to be rotated is a zero-size child placed with `translate`. The browser's transform engine
// turns the rotor; getBoundingClientRect() says where each child ended up, and that is the answer.
const el = (tag, cls, text) => { const e = document.createElement(tag); if (cls) e.className = cls; if (text != null) e.textContent = text; return e; };
const short = n => n < 10 ? n.toFixed(1) : n < 1000 ? n.toFixed(0) : n < 1e4 ? (n / 1000).toFixed(1) + 'k' : (n / 1000).toFixed(0) + 'k';

// Builds the faces inside `box` and returns { rotate(list, pos, show) }, where list is [[vector, heads], ...]
// (query first, then key). rotate returns the rotated copies, or null if the faces aren't laid out.
export function clockRig(box, M, { size = 48, posLabel } = {}) {
  const { HD, freq } = M, h2 = HD / 2, R = size / 2 - 3, groups = [['q', M.H], ['k', M.KVH]];
  const rot = [], pts = groups.map(() => []), was = [];
  box.style.setProperty('--clock', size + 'px'); box.style.setProperty('--hand', R + 'px');
  for (let i = 0; i < h2; i++) {
    const face = el('div', 'clock'), dial = el('div', 'dial'), ctr = el('div', 'ctr'), r = el('div', 'rot');
    for (let h = 0; h < M.H; h++) { const p = el('i', 'pt was'); was[h * h2 + i] = p; ctr.append(p); }
    r.append(el('i', 'hand'));
    groups.forEach(([cls, nh], g) => { for (let h = 0; h < nh; h++) { const p = el('i', 'pt ' + cls); pts[g][h * h2 + i] = p; r.append(p); } });
    ctr.append(r); dial.append(ctr); face.append(dial, el('div', 'per', short(2 * Math.PI / freq[i])));
    face.title = `One full turn every ${(2 * Math.PI / freq[i]).toPrecision(3)} tokens`;
    box.append(face); rot.push(r);
  }
  const cx = new Float64Array(h2), cy = new Float64Array(h2);
  return {
    box,
    rotate(list, pos, show) {
      // Write every style first, then read every position, so the browser lays out once per call.
      for (let i = 0; i < h2; i++) rot[i].style.rotate = pos * freq[i] + 'rad';
      const scale = list.map(([x], g) => {
        // Pixels per unit, one per face. A power of two, so value × scale is still an exact float; the longest pair
        // on a face lands in its outer half.
        const p = pts[g], nh = groups[g][1], S = new Float64Array(h2);
        for (let i = 0; i < h2; i++) {
          let m = 0; for (let h = 0; h < nh; h++) m = Math.max(m, Math.hypot(x[h * HD + i], x[h * HD + h2 + i]));
          const s = S[i] = m > 0 && isFinite(m) ? Math.pow(2, Math.floor(Math.log2(R / m))) : m === 0 ? 1 : NaN;
          for (let h = 0; h < nh; h++) {
            const t = x[h * HD + i] * s + 'px ' + x[h * HD + h2 + i] * s + 'px';
            p[h * h2 + i].style.translate = t; if (show && g === 0) was[h * h2 + i].style.translate = t;
          }
        }
        return S;
      });
      if (scale.some(S => S.some(isNaN)) || !rot[0].getClientRects().length) return null;
      for (let i = 0; i < h2; i++) { const c = rot[i].getBoundingClientRect(); cx[i] = c.left; cy[i] = c.top; }
      if (show && posLabel) posLabel.textContent = pos;
      return list.map(([x], g) => {
        const o = new Float32Array(x.length), p = pts[g], nh = groups[g][1], S = scale[g];
        for (let h = 0; h < nh; h++) for (let i = 0; i < h2; i++) {
          const c = p[h * h2 + i].getBoundingClientRect();
          o[h * HD + i] = (c.left - cx[i]) / S[i]; o[h * HD + h2 + i] = (c.top - cy[i]) / S[i];
        }
        return o;
      });
    }
  };
}

// Check the faces against Math.cos and Math.sin: n rounds of random vectors (sizes from 0.01 to 100) at random
// positions. Returns the largest error, as a number and as a share of the longest pair in its vector.
export function rigCheck(M, n = 200) {
  const { HD, H, KVH } = M, h2 = HD / 2; let maxErr = 0, maxRel = 0, pairs = 0, ms = 0;
  for (let r = 0; r < n; r++) {
    const pos = Math.floor(Math.random() * M.ctx), list = [H, KVH].map(nh => { const mag = Math.pow(10, Math.random() * 4 - 2); return [Float32Array.from({ length: nh * HD }, () => (Math.random() * 2 - 1) * mag), nh]; });
    const t0 = performance.now(), got = M.rig.rotate(list, pos, false); ms += performance.now() - t0; if (!got) return null;
    list.forEach(([x, nh], g) => {
      const want = M.rope(x, pos, nh); let big = 0, e = 0;
      for (let h = 0; h < nh; h++) for (let i = 0; i < h2; i++) big = Math.max(big, Math.hypot(x[h * HD + i], x[h * HD + h2 + i]));
      for (let j = 0; j < x.length; j++) e = Math.max(e, Math.abs(got[g][j] - want[j]));
      maxErr = Math.max(maxErr, e); maxRel = Math.max(maxRel, e / big); pairs += x.length / 2;
    });
  }
  return { pairs, maxErr, maxRel, msPerCall: ms / n, scrollY, clocksTop: M.rig.box.getBoundingClientRect().top };
}

// What the CSS rotation costs with the page live: feeds the same tokens with the engine switched every `per`
// tokens (so both see the same machine load and the same context lengths) and compares the medians.
export async function rigCost(M, ids, rounds = 12, per = 8) {
  const t = { css: [], js: [] }, rot = [], mid = a => a.sort((x, y) => x - y)[a.length >> 1]; let st = M.newState();
  for (let r = 0; r < rounds * 2; r++) {
    const e = M.ropeEngine = r % 2 ? 'js' : 'css', r0 = M.ropeMs, t0 = performance.now();
    for (let n = 0; n < per; n++) { if (st.pos >= ids.length) st = M.newState(); M.step(st, ids[st.pos]); await new Promise(f => setTimeout(f, 0)); }
    t[e].push((performance.now() - t0) / per); if (e === 'css') rot.push((M.ropeMs - r0) / per);
  }
  M.ropeEngine = 'css'; const css = mid(t.css), js = mid(t.js);
  return { cssMsPerToken: css, jsMsPerToken: js, cssTokensPerSecond: 1000 / css, jsTokensPerSecond: 1000 / js, rotationMsPerToken: mid(rot) };
}
