"""Compile a trained model.json into one HTML page whose forward pass is CSS colour maths.

    python3 tools/compile_css.py css/model.json tools/css_template.html css/index.html

The model is a tiny two-layer MLP (see train_css.py):

    8 characters -> embeddings (8 x 9 = 72 numbers) -> 192 ReLU neurons -> 37 logits

The trick: Chrome stores a registered `<color>` custom property as three unclamped floats (alpha is
clamped to 0..1, so it's useless). So every 3 numbers of the model become one colour:

    72 inputs  -> --x0 .. --x23   (24 colours)
    192 hidden -> --h0 .. --h63   (64 colours)
    37 logits  -> --o0 .. --o12   (13 colours, padded to 39 lanes)

and the maths becomes colour operations:

    3x3 weight block     color(from var(--x5) srgb calc(r*a + g*b + b*c) calc(...) calc(...))
    sum of many blocks   a binary tree of color-mix(in srgb, A, B), which averages, so weights are pre-scaled
    ReLU                 color(from <sum> srgb calc(max(r, 0)) ...)

The template's placeholders get filled in: /*CSS*/ (the generated stylesheet), /*VOCAB*/, /*META*/ (sizes and
stats the page shows) and <!--EXCERPT--> (real lines from the stylesheet, shown on the page).
"""

import html
import json
import sys

m = json.load(open(sys.argv[1]))
V, CTX, E, H = len(m["vocab"]), m["ctx"], m["E"], m["H"]

# 37 characters isn't a whole number of colours. Pad the output to 39 lanes; the padded logits get a huge
# negative bias so they can never be sampled (the page also trims them off before sampling).
Vp = -(-V // 3) * 3
m["W2"] = [row + [0.0] * (Vp - V) for row in m["W2"]]
m["b2"] = m["b2"] + [-1e4] * (Vp - V)


def f(x):
    """A number as short CSS text. 5 significant digits is plenty; Chrome keeps about 6."""
    return f"{x:.5g}"


def pow2(n):
    """Smallest power of two >= n, so the color-mix tree is a balanced binary tree."""
    p = 1
    while p < n:
        p *= 2
    return p


def tree(leaves, depth):
    """Sum `leaves` with nested color-mix() calls, indented so the tree shape is visible.

    color-mix(in srgb, A, B) with no percentages is (A + B) / 2, channel by channel. A balanced tree over N
    leaves therefore gives the mean, and since every weight was multiplied by N up front, the mean is the sum.
    """
    pad = "  " * depth
    if len(leaves) == 1:
        return pad + leaves[0]
    half = len(leaves) // 2
    return f"{pad}color-mix(in srgb,\n{tree(leaves[:half], depth + 1)},\n{tree(leaves[half:], depth + 1)}\n{pad})"


def layer(src, n_in, W, b, dst, n_out, relu, name):
    """CSS declarations for y = x @ W + b (and ReLU), x packed 3 per colour in --{src}i, y into --{dst}j."""
    K = n_in // 3  # input colours
    Kp = pow2(K + 1)  # leaves per output: one per input colour, one for the bias, then zero padding
    out = [
        f"  /* ---- {name}: {n_in} inputs ({K} colours) -> {n_out} outputs ({n_out // 3} colours).",
        f"     Each output colour sums {K} 3x3 blocks plus a bias leaf, padded to {Kp} leaves.",
        f"     Weights are pre-multiplied by {Kp} because the color-mix tree averages. ---- */",
    ]
    for j in range(n_out // 3):
        leaves = []
        for i in range(K):
            # One 3x3 block: output channel o = r*W[3i][3j+o] + g*W[3i+1][3j+o] + b*W[3i+2][3j+o].
            # (Inside the calc, r/g/b are the three numbers of input colour i, not colours.)
            ch = []
            for o in range(3):
                w = [W[i * 3 + c][j * 3 + o] * Kp for c in range(3)]
                ch.append(f"calc(r*{f(w[0])} + g*{f(w[1])} + b*{f(w[2])})")
            leaves.append(f"color(from var(--{src}{i}) srgb {' '.join(ch)})")
        # The bias is just a constant colour leaf.
        leaves.append("color(srgb " + " ".join(f(b[j * 3 + o] * Kp) for o in range(3)) + ")")
        # Zero leaves fill the tree up to a power of two.
        leaves += ["color(srgb 0 0 0)"] * (Kp - len(leaves))
        expr = tree(leaves, 3)
        if relu:
            # ReLU: clamp each channel of the summed colour at 0.
            expr = f"    color(from\n{expr}\n    srgb calc(max(r, 0)) calc(max(g, 0)) calc(max(b, 0)))"
        lanes = f"{dst}[{3 * j}..{3 * j + 2}]"
        out.append(f"\n  /* {lanes} */\n  --{dst}{j}:\n{expr};")
    return out


nx, nh, no = CTX * E // 3, H // 3, Vp // 3

# 1. Register every activation as a <color>. Registered properties get a real computed value (unregistered ones
#    are just token soup), so a layer can read the previous layer's result with var() and Chrome resolves the
#    whole chain in one style recalc.
props = ["/* ===== 1. Activation registers: every 3 numbers of the model live in one registered <color>. ===== */"]
for p, n, what in (("x", nx, "inputs (embeddings)"), ("h", nh, "hidden neurons"), ("o", no, "logits")):
    props.append(f"/* --{p}0 .. --{p}{n - 1}: {what} */")
    props += [f'@property --{p}{i} {{ syntax: "<color>"; inherits: true; initial-value: black; }}' for i in range(n)]

# 2. Embedding lookup as class selectors. The script puts e.g. "s3t12" on #m, meaning "token 12 at position 3",
#    and this rule writes that token's 9 embedding numbers into the 3 input colours for position 3.
emb = ["/* ===== 2. Embedding table. .s{position}t{token} sets the input colours for one position. ===== */"]
for s in range(CTX):
    emb.append(f"\n/* position {s}: --x{s * E // 3} .. --x{s * E // 3 + E // 3 - 1} */")
    for t in range(V):
        e = m["emb"][t]
        decls = "\n".join(
            f"  --x{s * E // 3 + k}: color(srgb {f(e[3 * k])} {f(e[3 * k + 1])} {f(e[3 * k + 2])});"
            for k in range(E // 3)
        )
        emb.append(f".s{s}t{t} {{ /* {json.dumps(m['vocab'][t])} */\n{decls}\n}}")

# 3. The network itself: both layers as declarations on #m.
net = [
    "/* ===== 3. The forward pass. Both layers are plain declarations on #m; reading one --o colour back",
    "          forces Chrome to evaluate the lot. ===== */",
    "#m {",
]
net += layer("x", CTX * E, m["W1"], m["b1"], "h", H, True, "Layer 1 + ReLU")
net += [""] + layer("h", H, m["W2"], m["b2"], "o", Vp, False, "Layer 2 (logits)")
net.append("}")
net = "\n".join(net)

# 4. The live swatches on the page: each square's background is a CSS expression on one neuron's colour,
#    squashed into a visible range. Nothing draws them from script.
sw = ["/* ===== 4. Live swatches: each square shows one activation colour, scaled to be visible. ===== */"]
sw += [
    f"#hv i:nth-child({i + 1}) {{ background: color(from var(--h{i}) srgb calc(r / 4) calc(g / 4) calc(b / 4)); }}"
    for i in range(nh)
]
sw += [
    f"#ov i:nth-child({i + 1}) {{ background: color(from var(--o{i}) srgb calc(r / 12 + .5) calc(g / 12 + .5) calc(b / 12 + .5)); }}"
    for i in range(no)
]

css = (
    "\n".join(
        ["", "/* ============ Everything below is generated by tools/compile_css.py. ============ */", ""]
        + props
        + [""]
        + emb
        + ["", net, ""]
        + sw
    )
    + "\n"
)

# Real lines for the "what the compiled stylesheet looks like" box on the page.
first_leaf = next(line.strip() for line in net.split("\n") if line.strip().startswith("color(from var(--x0)"))
excerpt = (
    "/* token 12 at position 0 sets three input colours */\n"
    + emb[13 + 1]
    + "\n\n/* one 3x3 block of layer 1: input colour 0 -> output colour 0 */\n"
    + first_leaf
)

stats = {
    "embRules": V * CTX,
    "props": nx + nh + no,
    "leaves": net.count("color(from var("),
    "mixes": net.count("color-mix("),
    "cssKB": round(len(css) / 1e3),
    "weights": V * E + CTX * E * H + H + H * V + V,
}

tpl = open(sys.argv[2]).read()
out = (
    tpl.replace("/*CSS*/", css)
    .replace("/*VOCAB*/ null", json.dumps("".join(m["vocab"])))
    .replace("/*META*/ null", json.dumps({"ctx": CTX, "no": no, "nh": nh, **stats}))
    .replace("<!--EXCERPT-->", html.escape(excerpt))
)
open(sys.argv[3], "w").write(out)
print(f"{len(out) / 1e3:.0f} KB, {len(css) / 1e3:.0f} KB css")
