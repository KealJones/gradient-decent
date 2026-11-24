// The canvas engine: matrix multiplies done by colour blending on a 2D canvas.
//
// The idea in one paragraph: a weight matrix is a picture (one pixel per weight). The input vector is a strip of
// pixels stretched down over every row. Drawing the strip on top of the picture with the `multiply` blend mode
// multiplies every weight by its input in one draw call. Then shrinking the picture to half its width, again and
// again, averages each row down to a single pixel, and that pixel is the row's sum divided by its width.
// One getImageData at the end reads all the answers. Used by stories/ and chat/.
//
// Colours can't be negative, so every number is split into two channels: red holds the positive part, green the
// negative part. Ordinary 8-bit canvases round to 1/255, which is about 5% off after a few layers, so everything
// here asks for float16 canvases.

// Canvas options for float16 pixels: drawing (colorType) and reading back (pixelFormat).
export const F16 = { colorType: 'float16' },
  RD = { pixelFormat: 'rgba-float16' };

// Base64 text to a Float32Array (the model files store raw float32 bytes this way).
export const f32 = s => {
  const b = atob(s),
    u = new Uint8Array(b.length);
  for (let i = 0; i < b.length; i++) u[i] = b.charCodeAt(i);
  return new Float32Array(u.buffer);
};
// The smallest power of two at least m: rows are padded to it so the halving chain ends at exactly one pixel.
const pow2 = m => {
  let p = 1;
  while (p < m) p *= 2;
  return p;
};
// Load an image (a weight PNG) and wait until it can be drawn.
export async function loadImage(src) {
  const img = new Image();
  img.src = src;
  await img.decode();
  return img;
}

// A full-precision matrix as a float16 picture: red holds positive weights, green negative, scaled by the largest.
export function floatPicture(w, n, m) {
  let max = 0;
  for (const v of w) max = Math.max(max, Math.abs(v));
  const d = new ImageData(m, n, RD);
  for (let i = 0; i < n * m; i++) {
    const v = w[i] / max;
    d.data[i * 4] = v > 0 ? v : 0;
    d.data[i * 4 + 1] = v < 0 ? -v : 0;
    d.data[i * 4 + 3] = 1;
  }
  const c = new OffscreenCanvas(m, n);
  c.getContext('2d', F16).putImageData(d, 0, 0);
  return { src: c, scale: max };
}

// One weight matrix as a picture. mul(x) returns W·x.
//   src: n rows by m columns (padded to a power of two), red = positive weight, green = negative
//   scale: one number, or one per row, that turns pixel values back into weights
export class Blend {
  constructor(src, n, m, scale) {
    this.n = n;
    this.m = m;
    this.scale = scale;
    this.src = src;
    const mp = (this.mp = pow2(m));
    // w2: the weight picture twice, one copy above the other. The top copy gets multiplied by the input's
    // (positive, negative) pixels and the bottom copy by the swapped (negative, positive) pixels; see mul().
    this.w2 = new OffscreenCanvas(mp, 2 * n);
    const wg = this.w2.getContext('2d', F16);
    wg.imageSmoothingEnabled = false;
    wg.drawImage(src, 0, 0);
    wg.drawImage(src, 0, n);
    // work: where the multiply happens. strip: the input vector as a two-pixel-tall image.
    this.work = new OffscreenCanvas(mp, 2 * n);
    this.g = this.work.getContext('2d', F16);
    this.strip = new OffscreenCanvas(mp, 2);
    this.sg = this.strip.getContext('2d', F16);
    this.sd = new ImageData(mp, 2, RD);
    for (let i = 0; i < mp * 2; i++) this.sd.data[i * 4 + 3] = 1;
    // The halving chain: canvases mp/2, mp/4, ... 1 pixel wide. Each is drawn from the one before with smoothing
    // on, so every pixel becomes the average of the two under it. The last one is read back, so it is told so.
    this.chain = [];
    for (let w = mp / 2; w >= 1; w /= 2) {
      const c = new OffscreenCanvas(w, 2 * n),
        g = c.getContext('2d', w === 1 ? { ...F16, willReadFrequently: true } : F16);
      g.imageSmoothingEnabled = true;
      g.globalCompositeOperation = 'copy';
      this.chain.push({ c, g, w });
    }
    this.y = new Float32Array(n);
  }
  // The same weights read back out of the picture as plain numbers, for the race against a JavaScript loop.
  floats() {
    if (!this.wf) {
      const { n, m, mp, scale } = this,
        d = this.w2.getContext('2d', F16).getImageData(0, 0, mp, n, RD).data,
        w = (this.wf = new Float32Array(n * m));
      for (let i = 0; i < n; i++) {
        const k = typeof scale === 'number' ? scale : scale[i];
        for (let j = 0; j < m; j++) {
          const o = (i * mp + j) * 4;
          w[i * m + j] = (d[o] - d[o + 1]) * k;
        }
      }
    }
    return this.wf;
  }
  // The same multiply as a plain JavaScript loop, for the race and the self-tests.
  mulJS(x) {
    const { n, m } = this,
      w = this.floats(),
      y = this.yj || (this.yj = new Float32Array(n));
    for (let i = 0; i < n; i++) {
      let s = 0;
      const o = i * m;
      for (let j = 0; j < m; j++) s += w[o + j] * x[j];
      y[i] = s;
    }
    return y;
  }
  // W·x by blending. Returns a Float32Array of n numbers (reused between calls).
  mul(x) {
    const { n, m, mp, sd, g, scale, y } = this,
      d = sd.data;
    // Scale the input so its largest entry is 1: pixels hold 0..1 best.
    let s = 0;
    for (let j = 0; j < m; j++) {
      const a = Math.abs(x[j]);
      if (a > s) s = a;
    }
    if (s === 0) {
      y.fill(0);
      return y;
    }
    // The input as a two-row strip: (positive, negative) on top, swapped underneath.
    //   top rows:    red = w⁺·x⁺, green = w⁻·x⁻   both products count towards the sum
    //   bottom rows: red = w⁺·x⁻, green = w⁻·x⁺   both products count against it
    for (let j = 0; j < m; j++) {
      const v = x[j] / s,
        p = v > 0 ? v : 0,
        q = v < 0 ? -v : 0;
      d[j * 4] = p;
      d[j * 4 + 1] = q;
      d[(mp + j) * 4] = q;
      d[(mp + j) * 4 + 1] = p;
    }
    this.sg.putImageData(sd, 0, 0);
    // Fresh copy of the weights, then multiply-blend the strip over it, stretched down all 2n rows. With smoothing
    // off, the top n rows get strip row 0 and the bottom n rows get strip row 1.
    g.globalCompositeOperation = 'copy';
    g.drawImage(this.w2, 0, 0);
    g.imageSmoothingEnabled = false;
    g.globalCompositeOperation = 'multiply';
    g.drawImage(this.strip, 0, 0, mp, 2, 0, 0, mp, 2 * n);
    // Halve the width until each row is one pixel: that pixel is the row's average.
    let prev = this.work,
      pw = mp;
    for (const { c, g: cg, w } of this.chain) {
      cg.drawImage(prev, 0, 0, pw, 2 * n, 0, 0, w, 2 * n);
      prev = c;
      pw = w;
    }
    // Read the 2n one-pixel rows. Each is (sum of products) / mp, so multiply back by mp, by the input scale s and by
    // the weight scale. The answer is (top red + top green) − (bottom red + bottom green).
    const r = this.chain[this.chain.length - 1].g.getImageData(0, 0, 1, 2 * n, RD).data,
      k = mp * s;
    if (typeof scale === 'number')
      for (let i = 0; i < n; i++) y[i] = (r[i * 4] + r[i * 4 + 1] - (r[(n + i) * 4] + r[(n + i) * 4 + 1])) * k * scale;
    else
      for (let i = 0; i < n; i++)
        y[i] = (r[i * 4] + r[i * 4 + 1] - (r[(n + i) * 4] + r[(n + i) * 4 + 1])) * k * scale[i];
    return y;
  }
}

// SiLU (the activation x / (1 + e^-x)) as a canvas gradient: rendered once, then eyedropped.
// The gradient runs from x = -12 to 12 across 4096 pixels, red for positive outputs, green for negative. Looking up
// SiLU(x) means reading the two pixels either side of x and blending between them. Outside ±12, SiLU is x or 0.
// Draws a brightened copy into the canvas `vis` if one is given, so the page can show it.
export function makeSiluGradient(vis) {
  const R = 12,
    LW = 4096,
    K = 1024;
  const c = new OffscreenCanvas(LW, 1),
    g = c.getContext('2d', F16);
  const gr = g.createLinearGradient(0, 0, LW, 0);
  for (let i = 0; i <= K; i++) {
    const x = -R + (2 * R * i) / K,
      v = x / (1 + Math.exp(-x));
    gr.addColorStop(i / K, `color(srgb ${Math.max(v, 0) / R} ${Math.max(-v, 0) / R} 0)`);
  }
  g.fillStyle = gr;
  g.fillRect(0, 0, LW, 1);
  const d = g.getImageData(0, 0, LW, 1, RD).data;
  if (vis) {
    const v = vis.getContext('2d');
    v.drawImage(c, 0, 0, LW, 1, 0, 0, vis.width, 1);
    const b = v.getImageData(0, 0, vis.width, 1);
    for (let i = 0; i < vis.width; i++) {
      b.data[i * 4] = Math.min(255, Math.sqrt(b.data[i * 4] / 255) * 255);
      b.data[i * 4 + 1] = Math.min(255, Math.sqrt(b.data[i * 4 + 1] / 255) * 255 * 3);
    }
    v.putImageData(b, 0, 0);
  }
  // The lookup the model calls: linear interpolation between neighbouring pixels.
  return x => {
    if (x >= R) return x;
    if (x <= -R) return 0;
    const u = ((x + R) / (2 * R)) * LW - 0.5,
      i = Math.max(0, Math.min(LW - 2, Math.floor(u))),
      t = Math.max(0, Math.min(1, u - i));
    const a = d[i * 4] - d[i * 4 + 1],
      b = d[(i + 1) * 4] - d[(i + 1) * 4 + 1];
    return (a + (b - a) * t) * R;
  };
}
