"""Train the tiny character model that css/ runs, with nothing but numpy.

    python3 tools/train_css.py shakespeare.txt 40000 [out.json]

Model: the last 8 characters -> a 9-number embedding each (72 numbers) -> 192 ReLU neurons -> 37 logits.
About 21K weights. Two minutes on a laptop CPU gets the loss to about 1.6 nats per character.

Every number count is a multiple of 3 on purpose (9, 72, 192): compile_css.py packs 3 numbers into each colour.
"""

import json
import sys
import time

import numpy as np

rng = np.random.default_rng(0)

# ---- Data: lowercase text, anything outside the vocabulary is dropped ----
raw = open(sys.argv[1]).read().lower()
VOCAB = list("\n abcdefghijklmnopqrstuvwxyz.,;:!?'-&")  # 37 characters; newline is token 0
idx = {c: i for i, c in enumerate(VOCAB)}
data = np.array([idx[c] for c in raw if c in idx], dtype=np.int64)

V, CTX, E, H = len(VOCAB), 8, 9, 192
X = np.lib.stride_tricks.sliding_window_view(data, CTX + 1)  # every 9-character window: 8 in, 1 to predict

# ---- Parameters (He init for the ReLU layer) ----
P = {
    "emb": rng.normal(0, 1, (V, E)),
    "W1": rng.normal(0, (2 / (CTX * E)) ** 0.5, (CTX * E, H)),
    "b1": np.zeros(H),
    "W2": rng.normal(0, (1 / H) ** 0.5, (H, V)),
    "b2": np.zeros(V),
}

# Adam state.
M = {k: np.zeros_like(v) for k, v in P.items()}
S = {k: np.zeros_like(v) for k, v in P.items()}
steps, B, lr = int(sys.argv[2]), 512, 3e-3

t0 = time.time()
for step in range(1, steps + 1):
    batch = X[rng.integers(0, len(X), B)]
    ctx, y = batch[:, :CTX], batch[:, CTX]

    # ---- Forward ----
    e = P["emb"][ctx].reshape(B, -1)  # (B, 72) concatenated embeddings
    z = e @ P["W1"] + P["b1"]
    h = np.maximum(z, 0)  # ReLU
    lg = h @ P["W2"] + P["b2"]  # logits
    lg -= lg.max(1, keepdims=True)
    pr = np.exp(lg)
    pr /= pr.sum(1, keepdims=True)  # softmax
    loss = -np.log(pr[np.arange(B), y] + 1e-9).mean()

    # ---- Backward (hand-written) ----
    d = pr
    d[np.arange(B), y] -= 1  # d loss / d logits for softmax + cross-entropy
    d /= B
    G = {"W2": h.T @ d, "b2": d.sum(0)}
    dh = d @ P["W2"].T * (z > 0)  # through the ReLU
    G["W1"] = e.T @ dh
    G["b1"] = dh.sum(0)
    de = (dh @ P["W1"].T).reshape(B, CTX, E)
    G["emb"] = np.zeros_like(P["emb"])
    np.add.at(G["emb"], ctx, de)  # scatter back into the embedding rows that were used

    # ---- Adam, learning rate decaying linearly to 10% ----
    a = lr * (0.1 + 0.9 * (1 - step / steps))
    for k in P:
        M[k] = 0.9 * M[k] + 0.1 * G[k]
        S[k] = 0.999 * S[k] + 0.001 * G[k] ** 2
        P[k] -= a * (M[k] / (1 - 0.9**step)) / (np.sqrt(S[k] / (1 - 0.999**step)) + 1e-8)

    if step % 1000 == 0:
        print(f"step {step} loss {loss:.3f} {time.time() - t0:.0f}s", flush=True)

# The page's compiler and the JS reference both read this file.
out = sys.argv[3] if len(sys.argv) > 3 else "model.json"
json.dump({"vocab": VOCAB, "ctx": CTX, "E": E, "H": H, **{k: v.tolist() for k, v in P.items()}}, open(out, "w"))
print("saved", out)
