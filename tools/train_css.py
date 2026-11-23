import numpy as np, json, time, sys
rng = np.random.default_rng(0)
raw = open(sys.argv[1]).read().lower()
VOCAB = list("\n abcdefghijklmnopqrstuvwxyz.,;:!?'-&")  # 36 = 12 colors
idx = {c: i for i, c in enumerate(VOCAB)}
data = np.array([idx[c] for c in raw if c in idx], dtype=np.int64)
V, CTX, E, H = len(VOCAB), 8, 9, 192
X = np.lib.stride_tricks.sliding_window_view(data, CTX + 1)
P = {
  'emb': rng.normal(0, 1, (V, E)),
  'W1': rng.normal(0, (2 / (CTX * E)) ** .5, (CTX * E, H)), 'b1': np.zeros(H),
  'W2': rng.normal(0, (1 / H) ** .5, (H, V)), 'b2': np.zeros(V),
}
M = {k: np.zeros_like(v) for k, v in P.items()}; S = {k: np.zeros_like(v) for k, v in P.items()}
steps, B, lr = int(sys.argv[2]), 512, 3e-3
t0 = time.time()
for step in range(1, steps + 1):
    batch = X[rng.integers(0, len(X), B)]
    ctx, y = batch[:, :CTX], batch[:, CTX]
    e = P['emb'][ctx].reshape(B, -1)
    z = e @ P['W1'] + P['b1']; h = np.maximum(z, 0)
    lg = h @ P['W2'] + P['b2']
    lg -= lg.max(1, keepdims=True); pr = np.exp(lg); pr /= pr.sum(1, keepdims=True)
    loss = -np.log(pr[np.arange(B), y] + 1e-9).mean()
    d = pr; d[np.arange(B), y] -= 1; d /= B
    G = {'W2': h.T @ d, 'b2': d.sum(0)}
    dh = d @ P['W2'].T * (z > 0)
    G['W1'] = e.T @ dh; G['b1'] = dh.sum(0)
    de = (dh @ P['W1'].T).reshape(B, CTX, E)
    G['emb'] = np.zeros_like(P['emb']); np.add.at(G['emb'], ctx, de)
    a = lr * (0.1 + 0.9 * (1 - step / steps))
    for k in P:
        M[k] = .9 * M[k] + .1 * G[k]; S[k] = .999 * S[k] + .001 * G[k] ** 2
        P[k] -= a * (M[k] / (1 - .9 ** step)) / (np.sqrt(S[k] / (1 - .999 ** step)) + 1e-8)
    if step % 1000 == 0:
        print(f'step {step} loss {loss:.3f} {time.time()-t0:.0f}s', flush=True)
json.dump({'vocab': VOCAB, 'ctx': CTX, 'E': E, 'H': H, **{k: v.tolist() for k, v in P.items()}}, open(sys.argv[3] if len(sys.argv) > 3 else 'model.json', 'w'))
print('saved')
