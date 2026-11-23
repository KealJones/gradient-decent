// A small Llama-style transformer whose matrix multiplies all go through the canvas engine.
import { Blend, f32, loadImage, floatPicture, makeSiluGradient } from './blend.js?v=6';

export const MATS = ['q', 'k', 'v', 'o', 'gate', 'up', 'down'];

export class Model {
  // tile(key, label) may return a 2D context to draw that matrix's live product into.
  static async load(url, { tile, gradientCanvas } = {}) {
    const B = await (await fetch(url)).json(), M = new Model();
    Object.assign(M, { KVH: B.cfg.H, eps: 1e-6, theta: 10000, ctx: 512 }, B.cfg);
    M.bundle = B; M.emb = f32(B.emb); M.n1 = B.n1.map(f32); M.n2 = B.n2.map(f32); M.nf = f32(B.nf);
    M.mats = {}; M.tiles = {}; M.show = true; M.ropeEngine = 'css'; M.rig = null; M.ropeMs = 0;
    const add = async (key, label, e) => {
      M.mats[key] = new Blend(await loadImage(e.png), e.n, e.m, e.scale ? f32(e.scale) : e.alpha);
      if (tile) M.tiles[key] = tile(key, label);
    };
    for (let l = 0; l < M.L; l++) for (const nm of MATS) await add(l + '.' + nm, 'L' + (l + 1) + ' ' + nm, B.mats[l + '.' + nm]);
    if (B.mats.head) await add('head', 'head', B.mats.head);
    else { const p = floatPicture(M.emb, M.V, M.D); M.mats.head = new Blend(p.src, M.V, M.D, p.scale); if (tile) M.tiles.head = tile('head', 'head'); }
    M.silu = makeSiluGradient(gradientCanvas);
    M.freq = new Float32Array(M.HD / 2); for (let i = 0; i < M.HD / 2; i++) M.freq[i] = 1 / Math.pow(M.theta, 2 * i / M.HD);
    M.count = Object.keys(M.mats).length;
    return M;
  }
  mul(key, x) {
    const b = this.mats[key], js = this.engine === 'js', y = js ? b.mulJS(x) : b.mul(x), t = !js && this.show && this.tiles[key];
    if (t) { t.globalCompositeOperation = 'copy'; t.imageSmoothingEnabled = false; t.filter = 'brightness(4)'; t.drawImage(b.work, 0, 0, b.mp, b.n, 0, 0, t.canvas.width, t.canvas.height); }
    return y;
  }
  newState() { return { pos: 0, K: Array.from({ length: this.L }, () => []), V: Array.from({ length: this.L }, () => []) }; }
  rope(x, pos, nh) {
    const { HD, freq } = this, h2 = HD / 2, o = new Float32Array(x.length);
    for (let h = 0; h < nh; h++) for (let i = 0; i < h2; i++) { const c = Math.cos(pos * freq[i]), s = Math.sin(pos * freq[i]), a = x[h * HD + i], b = x[h * HD + h2 + i]; o[h * HD + i] = a * c - b * s; o[h * HD + h2 + i] = b * c + a * s; }
    return o;
  }
  // Rotate several vectors for the same position: list is [[vector, heads], ...]. With ropeEngine 'css' and a
  // clock rig attached (lib/clocks.js) the turning is done by CSS transforms; otherwise, and in the plain
  // JavaScript pass of the race, by rope() above.
  rotate(list, pos, last) {
    const t0 = performance.now(), css = this.engine !== 'js' && this.ropeEngine === 'css' && this.rig;
    const out = (css && this.rig.rotate(list, pos, this.show && last)) || list.map(([x, nh]) => this.rope(x, pos, nh));
    this.ropeMs += performance.now() - t0; return out;
  }
  rms(x, w, out) { let s = 0; for (let i = 0; i < x.length; i++) s += x[i] * x[i]; const r = 1 / Math.sqrt(s / x.length + this.eps); for (let i = 0; i < x.length; i++) out[i] = x[i] * r * w[i]; return out; }
  // Feed one token. Returns the logits for the next one, or null when wantLogits is false.
  step(st, tok, wantLogits = true) {
    const { D, H, KVH, HD, L, FF } = this, pos = st.pos++, per = H / KVH, isq = 1 / Math.sqrt(HD);
    const x = Float32Array.from(this.emb.subarray(tok * D, tok * D + D)), h = new Float32Array(D), a = new Float32Array(D), t = new Float32Array(FF);
    for (let l = 0; l < L; l++) {
      this.rms(x, this.n1[l], h);
      const [q, k] = this.rotate([[this.mul(l + '.q', h), H], [this.mul(l + '.k', h), KVH]], pos, l === L - 1), v = Float32Array.from(this.mul(l + '.v', h));
      st.K[l].push(k); st.V[l].push(v);
      const T = st.K[l].length, sc = new Float32Array(T);
      for (let hh = 0; hh < H; hh++) {
        const g = Math.floor(hh / per) * HD; let mx = -Infinity;
        for (let i = 0; i < T; i++) { const kk = st.K[l][i]; let s = 0; for (let d = 0; d < HD; d++) s += kk[g + d] * q[hh * HD + d]; sc[i] = s * isq; if (sc[i] > mx) mx = sc[i]; }
        let z = 0; for (let i = 0; i < T; i++) { sc[i] = Math.exp(sc[i] - mx); z += sc[i]; }
        for (let d = 0; d < HD; d++) { let s = 0; for (let i = 0; i < T; i++) s += sc[i] * st.V[l][i][g + d]; a[hh * HD + d] = s / z; }
      }
      const o = this.mul(l + '.o', a); for (let i = 0; i < D; i++) x[i] += o[i];
      this.rms(x, this.n2[l], h);
      const g = this.mul(l + '.gate', h); for (let i = 0; i < FF; i++) t[i] = this.silu(g[i]);
      const u = this.mul(l + '.up', h); for (let i = 0; i < FF; i++) t[i] *= u[i];
      const dn = this.mul(l + '.down', t); for (let i = 0; i < D; i++) x[i] += dn[i];
    }
    return wantLogits ? Float32Array.from(this.mul('head', this.rms(x, this.nf, h))) : null;
  }
}

// r is the random number, 0 to 1. Passing it in lets a test ask what a given roll would pick.
export function sample(lg, { temp = 1, topK = 40, topP = 1, r = Math.random() } = {}) {
  if (!temp) { let b = 0; for (let i = 1; i < lg.length; i++) if (lg[i] > lg[b]) b = i; return b; }
  const idx = Array.from(lg.keys()).sort((a, b) => lg[b] - lg[a]).slice(0, topK);
  let p = idx.map(i => Math.exp((lg[i] - lg[idx[0]]) / temp)); const z = p.reduce((a, b) => a + b, 0); p = p.map(v => v / z);
  let keep = p.length; if (topP < 1) { let c = 0; for (let i = 0; i < p.length; i++) { c += p[i]; if (c >= topP) { keep = i + 1; break; } } }
  r *= p.slice(0, keep).reduce((a, b) => a + b, 0);
  for (let i = 0; i < keep; i++) { r -= p[i]; if (r <= 0) return idx[i]; }
  return idx[0];
}

// Compare the canvas against the reference run stored in the bundle.
export function selfTest(M, ids, greedy, logits) {
  M.show = false; const st = M.newState(); let lg;
  ids.forEach((id, i) => { lg = M.step(st, id, i === ids.length - 1); });
  let err = 0, mag = 0; for (let i = 0; i < lg.length; i++) { err = Math.max(err, Math.abs(lg[i] - logits[i])); mag = Math.max(mag, Math.abs(logits[i])); }
  const got = [], t0 = performance.now();
  for (let i = 0; i < greedy.length; i++) { const t = sample(lg, { temp: 0 }); got.push(t); if (t === 2) break; lg = M.step(st, t); }
  let same = 0; while (same < got.length && got[same] === greedy[same]) same++;
  let lutErr = 0; for (let x = -11.9; x < 11.9; x += 0.0137) lutErr = Math.max(lutErr, Math.abs(M.silu(x) - x / (1 + Math.exp(-x))));
  M.show = true;
  return { logitMaxErr: err, logitMag: mag, tokensMatched: same, of: greedy.length, msPerToken: (performance.now() - t0) / Math.max(1, got.length - 1), lutErr, got };
}

// Race the canvas against a plain JavaScript loop over the same weights (read back out of the pictures),
// feeding both the same tokens and comparing their choice at every position.
export async function race(M, ids) {
  const run = engine => { M.engine = engine; M.show = false; const st = M.newState(), out = [], t0 = performance.now(); for (const id of ids) out.push(M.step(st, id)); return { out, ms: (performance.now() - t0) / ids.length }; };
  for (const b of Object.values(M.mats)) b.floats();
  const c = run('canvas'); await new Promise(r => setTimeout(r, 0)); const j = run('js'); M.engine = 'canvas'; M.show = true;
  let agree = 0, gap = 0;
  c.out.forEach((lg, i) => { const rf = j.out[i]; let a = 0, b = 0; for (let k = 0; k < lg.length; k++) { if (lg[k] > lg[a]) a = k; if (rf[k] > rf[b]) b = k; gap = Math.max(gap, Math.abs(lg[k] - rf[k])); } if (a === b) agree++; });
  return { canvasMs: c.ms, jsMs: j.ms, slower: c.ms / j.ms, agree, of: ids.length, maxLogitGap: gap };
}
