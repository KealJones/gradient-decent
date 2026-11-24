"""Compile the tiny character model into a page whose matrix maths is done by CSS transforms.

    python3 tools/compile_cards.py css/model.json tools/cards_template.html cards/index.html

Same model as css/ (see train_css.py):

    8 characters -> embeddings (8 x 9 = 72 numbers) -> 192 ReLU neurons -> 37 logits

The trick: a CSS transform list is a product of 4x4 matrices, and Chrome multiplies it out to one matrix3d()
that script can read from the computed style. So a neuron is a transform list, built from matrices of this shape
(written the usual way, rows x y z w; matrix3d() lists them column by column):

        | 1   0   0   0 |      a   one input (a number, usually a var())          "z moves by a"
    M = | 0   1   0   0 |      p,q one pair of weights (constants)                the perspective row
        | 0   0   1   a |
        | p   q   0   1 |

Multiplying two of them gives

        | 1        0        0   0    |
        | 0        1        0   0    |      the z row now holds  a*p' and a*q' :
        | a*p'     a*q'     1   a+a' |      the first one's input times the second one's weights,
        | p+p'     q+q'     0   1    |      and everything else just adds.

and those z-row numbers keep adding up as more matrices follow. With inputs x0..xN in the `a` slots and the right
constants in the `p,q` slots, a list of N+2 matrices ends as

        | 1    0    0   0      |
        | 0    1    0   0      |      s1 = sum of x_k * w1_k   (one neuron)
        | s1   s2   1   sum x  |      s2 = sum of x_k * w2_k   (the neuron next to it)
        | 0    0    0   1      |

which is a card tilted by s1 left-to-right and s2 top-to-bottom, pushed towards you by the sum of its inputs.

The constants: a matrix's `a` multiplies the p,q of every matrix AFTER it. So for input k to end up multiplied
by its weight w_k, the p's from position k+1 to the end must sum to w_k. That makes each one a difference:

    position 0       a = 1  (the bias's "input")    p = -bias       (cancels the leftover; nothing is before it)
    position k+1     a = x_(k+1)                    p = w_k - w_(k+1)      where w_0 is the bias
    last position    a = 0                          p = w_N

The p's sum to zero overall, so the perspective row ends clean. The subtraction is done here, once, at build time.
In the browser every multiply and add is the transform engine's.

The template's placeholders get filled in: /*CSS*/ (the generated stylesheet), /*VOCAB*/, /*META*/ (sizes and
stats the page shows) and <!--EXCERPT--> (real lines from the stylesheet, shown on the page).
"""

import html
import json
import sys

m = json.load(open(sys.argv[1]))
V, CTX, E, H = len(m["vocab"]), m["ctx"], m["E"], m["H"]

# Two neurons per card, so 37 logits is padded to 38. The spare one has zero weights and is never read.
Vp = V + V % 2
m["W2"] = [row + [0.0] * (Vp - V) for row in m["W2"]]
m["b2"] = m["b2"] + [0.0] * (Vp - V)


Q = 10**6  # weights are rounded to 6 decimal places before the differences are taken (see card())


def f(n):
    """An integer count of millionths as short CSS text: 123456 -> .123456, -2500000 -> -2.5"""
    s = f"{abs(n) // Q}.{abs(n) % Q:06d}".rstrip("0").rstrip(".").removeprefix("0") or "0"
    return "-" + s if n < 0 and s != "0" else s


def q(x):
    return round(x * Q)


def card(inputs, W, b, j):
    """The transform list for card j: neurons 2j and 2j+1 of y = x @ W + b.

    `inputs` is the CSS text for each x_k (a var(), possibly wrapped in max()).

    The weights are rounded first and subtracted as whole numbers of millionths, so every difference is written
    out exactly and they still sum back to the rounded weights. Rounding the differences instead lets the
    errors pile up along the list (measured: 10 times the error on the hidden layer).
    """
    # w[k] is the pair of weights input k needs; k = 0 is the bias, whose input is the constant 1.
    w = [(q(b[2 * j]), q(b[2 * j + 1]))] + [(q(W[k][2 * j]), q(W[k][2 * j + 1])) for k in range(len(inputs))]
    a = ["1"] + inputs + ["0"]
    pq = [(-w[0][0], -w[0][1])]
    pq += [(w[k][0] - w[k + 1][0], w[k][1] - w[k + 1][1]) for k in range(len(w) - 1)]
    pq += [w[-1]]
    #                 column x        column y        column z  column w
    return [f"matrix3d(1,0,0,{f(p)},0,1,0,{f(r)},0,0,1,0,0,0,{x},1)" for x, (p, r) in zip(a, pq)]


def layer(sel, inputs, W, b, n_out, name):
    """One rule per card: `sel i:nth-child(j+1) { transform: ...; }`."""
    out = [
        f"/* ---- {name}: {len(inputs)} inputs -> {n_out} outputs on {n_out // 2} cards,",
        f"     {len(inputs) + 2} matrices per card. ---- */",
    ]
    for j in range(n_out // 2):
        fns = card(inputs, W, b, j)
        out.append(
            f"{sel} i:nth-child({j + 1}) {{ /* outputs {2 * j} and {2 * j + 1} */\n  transform:\n    "
            + "\n    ".join(fns)
            + ";\n}"
        )
    return out


# 1. Embedding lookup as class selectors, same as css/. The script puts e.g. "s3t12" on #m ("token 12 at
#    position 3") and this rule writes that token's 9 embedding numbers into the 9 inputs for position 3.
emb = ["/* ===== 1. Embedding table. .s{position}t{token} sets the 9 inputs for one position. ===== */"]
for s in range(CTX):
    emb.append(f"\n/* position {s}: --x{s * E} .. --x{s * E + E - 1} */")
    for t in range(V):
        decls = " ".join(f"--x{s * E + k}:{f(q(m['emb'][t][k]))};" for k in range(E))
        emb.append(f".s{s}t{t} {{ {decls} }} /* {json.dumps(m['vocab'][t])} */")

# 2. Layer 1: the inputs are the --x variables set by the class rules above.
l1 = ["/* ===== 2. Layer 1. Each card multiplies its list out; the tilt it ends up with is two neurons. ===== */"]
l1 += layer("#w1", [f"var(--x{k})" for k in range(CTX * E)], m["W1"], m["b1"], H, "Layer 1")

# 3. Layer 2: the inputs are the --h variables the script copies over from the layer 1 cards. The ReLU is the
#    max(0, ...) around each one. That is CSS maths, but not the transform engine.
l2 = [
    "/* ===== 3. Layer 2. Inputs are layer 1 tilts, copied into --h variables by the script. The ReLU is",
    "          the max(0, ...) on the way in. ===== */",
]
l2 += layer("#w2", [f"max(0,var(--h{k}))" for k in range(H)], m["W2"], m["b2"], Vp, "Layer 2 (logits)")

css = (
    "\n".join(
        ["", "/* ============ Everything below is generated by tools/compile_cards.py. ============ */", ""]
        + emb
        + [""]
        + l1
        + [""]
        + l2
    )
    + "\n"
)

# Real lines for the "what the compiled stylesheet looks like" box on the page.
first = l1[3].split("\n")
last2 = l2[4].split("\n")
excerpt = (
    "/* token 12 at position 0 sets nine inputs */\n"
    + emb[2 + 12]
    + "\n\n/* the first card of layer 1: 74 matrices, of which the first four and the last are */\n"
    + "\n".join(first[:6])
    + "\n    …\n"
    + "\n".join(first[-2:])
    + "\n\n/* the first card of layer 2 starts like this */\n"
    + "\n".join(last2[:4])
    + "\n    …"
)

n_mat = css.count("matrix3d(")
stats = {
    "embRules": V * CTX,
    "cards1": H // 2,
    "cards2": Vp // 2,
    "per1": CTX * E + 2,
    "per2": H + 2,
    "matrices": n_mat,
    "cssKB": round(len(css) / 1e3),
    "weights": V * E + CTX * E * H + H + H * V + V,
}
stats["bytesPerWeight"] = round(len(css) / stats["weights"])

tpl = open(sys.argv[2]).read()
out = (
    tpl.replace("/*CSS*/", css)
    .replace("/*VOCAB*/ null", json.dumps("".join(m["vocab"])))
    .replace("/*META*/ null", json.dumps({"ctx": CTX, "H": H, **stats}))
    .replace("<!--EXCERPT-->", html.escape(excerpt))
)
open(sys.argv[3], "w").write(out)
print(f"{len(out) / 1e3:.0f} KB, {len(css) / 1e3:.0f} KB css, {n_mat} matrices")
