"""Write the css/ character model as a sound file for audio/, with nothing but numpy.

    python3 tools/export_audio.py css/model.json audio/weights.wav audio/layout.json

The WAV is mono, 32-bit float, 8000 Hz. Every number in it is a model weight divided by SCALE (a power of two, so
the division is exact in float32) which keeps the file inside -1..1 like any other sound file. The sections are
laid out the way the page's audio graph wants them, so the script on the page only has to cut the file up:

    emb    37 x 9      the embedding table, one row per character
    ir1    192 x 72    layer 1 as an impulse response: neuron i's weights, reversed, start at sample i*72
    bias1  192 x 72    layer 1's biases as a track: bias i sits at sample i*72 + 71, silence in between
    ir2    192 x 72    layer 2 as an impulse response: neuron i's 37 outgoing weights start at sample (191-i)*72
    bias2  37          layer 2's biases

layout.json holds the section offsets and the two numbers the graph needs to keep the ReLU in range.
"""

import json
import struct
import sys

import numpy as np

src, wav_path, layout_path = sys.argv[1:4]
m = json.load(open(src))
emb, W1, b1, W2, b2 = (np.array(m[k], dtype=np.float32) for k in ("emb", "W1", "b1", "W2", "b2"))
V, E = emb.shape
IN, H = W1.shape  # 72 inputs, 192 neurons
assert W2.shape == (H, V) and V <= IN, "layer 2 needs its 37 outputs to fit in one 72-sample slot"

RATE = 8000
# Smallest power of two that brings every weight inside -1..1.
SCALE = 2.0 ** np.ceil(np.log2(max(np.abs(a).max() for a in (emb, W1, b1, W2, b2))))

# Layer 1: a convolution slides the reversed kernel over the input, so each row goes in back to front.
ir1 = np.zeros((H, IN), np.float32)
ir1[:, ::-1] = W1.T  # ir1[i, 71 - j] = W1[j, i]

# Bias 1: one value per 72-sample slot, lined up with the sample where the convolver's answer lands.
bias1 = np.zeros((H, IN), np.float32)
bias1[:, IN - 1] = b1

# Layer 2: the hidden values arrive 72 samples apart, so neuron i's weights sit 72 samples apart too,
# in reverse order. Each slot has room for 72 numbers and uses 37.
ir2 = np.zeros((H, IN), np.float32)
ir2[::-1, :V] = W2  # ir2[191 - i, o] = W2[i, o]

sections = [("emb", emb), ("ir1", ir1), ("bias1", bias1), ("ir2", ir2), ("bias2", b2)]
data = np.concatenate([a.ravel() for _, a in sections]) / np.float32(SCALE)
data = data.astype("<f4")
assert np.abs(data).max() <= 1

# The largest value any neuron can reach before the ReLU, over every possible 8-character context.
# The WaveShaper clamps its input to -1..1, so the graph divides by a power of two above this first.
per_slot = np.einsum("te,sej->stj", emb.astype(np.float64), W1.astype(np.float64).reshape(m["ctx"], E, H))
hmax = float((per_slot.max(axis=1).sum(axis=0) + b1).max())
relu_div = float(2.0 ** np.ceil(np.log2(hmax)))

# A float WAV: 'fmt ' with format tag 3 (IEEE float), a 'fact' chunk (required for non-PCM), then the samples.
fmt = struct.pack("<HHIIHH", 3, 1, RATE, RATE * 4, 4, 32)
fact = struct.pack("<I", len(data))
body = (
    b"WAVE"
    + b"fmt "
    + struct.pack("<I", len(fmt))
    + fmt
    + b"fact"
    + struct.pack("<I", 4)
    + fact
    + b"data"
    + struct.pack("<I", data.nbytes)
    + data.tobytes()
)
open(wav_path, "wb").write(b"RIFF" + struct.pack("<I", len(body)) + body)

layout, at = {}, 0
for name, a in sections:
    layout[name] = [at, a.size]
    at += a.size
json.dump(
    {
        "vocab": "".join(m["vocab"]),
        "ctx": m["ctx"],
        "E": E,
        "IN": IN,
        "H": H,
        "V": V,
        "rate": RATE,
        "scale": float(SCALE),
        "reluDiv": relu_div,
        "hmax": round(hmax, 3),
        "samples": int(len(data)),
        "sections": layout,
    },
    open(layout_path, "w"),
)
print(f"{wav_path}: {len(data)} samples, {len(data) / RATE:.2f} s at {RATE} Hz, weights / {SCALE:g}")
print(f"largest possible pre-ReLU value {hmax:.2f}, so the graph divides by {relu_div:g} before the WaveShaper")
