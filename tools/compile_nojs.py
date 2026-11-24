"""Compile a trained model.json into a page with no script at all that writes text by itself.

    python3 tools/compile_nojs.py css/model.json no-js/index.html

Same model and same colour maths as tools/compile_css.py (8 characters -> 72 numbers -> 192 ReLU neurons -> 37
logits, three numbers per registered <color>). What the css/ page left to JavaScript is done here in CSS too:

    the loop        a CSS animation is the clock; two paused 1 ms animations latch the state between ticks
                    (the trick from Lyra Rebane's x86CSS)
    the embedding   if(style(--c3: 12): ...) picks the colours for "token 12 at position 3"
    sampling        Gumbel noise from sin() and log() of a seed is added to the logits, then argmax
    argmax          colour maths finds the largest lane; style() equality turns the winning colour into a number
    the text        each tick appends one string to a custom property, shown with content:
"""

import html
import json
import os
import sys

m = json.load(open(sys.argv[1]))
VOCAB, CTX, E, H = m["vocab"], m["ctx"], m["E"], m["H"]
V = len(VOCAB)
Vp = -(-V // 3) * 3  # 37 characters padded to 39 lanes = 13 colours
W2 = [row + [0.0] * (Vp - V) for row in m["W2"]]
b2 = m["b2"] + [-1e4] * (Vp - V)  # padded lanes can never win
PROMPT = "first citizen:\n"
# TICK_MS: one full clock cycle (4 phases). LIMIT: characters before it stops. EPS: slack for "is this zero?".
TICK_MS, LIMIT, EPS = int(os.environ.get("TICK", 1200)), 700, 1e-4
nx, nh, no = CTX * E // 3, H // 3, Vp // 3  # how many colours (3 numbers each) for inputs, hidden, logits


def f(x):
    """A number as short CSS text, 5 significant digits."""
    return f"{x:.5g}"


def pow2(n):
    """Smallest power of two at least n: a balanced mixing tree needs a power-of-two number of leaves."""
    p = 1
    while p < n:
        p *= 2
    return p


def tree(leaves):
    """Sum colours with a balanced color-mix() tree. Mixing averages, so weights are pre-scaled by the leaf count."""
    if len(leaves) == 1:
        return leaves[0]
    half = len(leaves) // 2
    return f"color-mix(in srgb,{tree(leaves[:half])},{tree(leaves[half:])})"


def layer(src, n_in, W, b, dst, n_out, relu):
    """One dense layer as CSS. Each output colour (3 neurons) is a mix of one leaf per input colour: a relative colour
    whose red, green and blue are 3x3 blocks of the weight matrix times the input's r, g, b. Plus a bias leaf, padded
    with black leaves to a power of two, then optionally ReLU (max(channel, 0))."""
    K = n_in // 3
    Kp = pow2(K + 1)
    out = []
    for j in range(n_out // 3):
        leaves = []
        for i in range(K):
            ch = [
                f"calc(r*{f(W[i * 3][j * 3 + o] * Kp)} + g*{f(W[i * 3 + 1][j * 3 + o] * Kp)} + b*{f(W[i * 3 + 2][j * 3 + o] * Kp)})"
                for o in range(3)
            ]
            leaves.append(f"color(from var(--{src}{i}) srgb {' '.join(ch)})")
        leaves.append("color(srgb " + " ".join(f(b[j * 3 + o] * Kp) for o in range(3)) + ")")
        leaves += ["color(srgb 0 0 0)"] * (Kp - len(leaves))
        expr = tree(leaves)
        if relu:
            expr = f"color(from {expr} srgb calc(max(r,0)) calc(max(g,0)) calc(max(b,0)))"
        out.append(f"  --{dst}{j}: {expr};")
    return out


def css_str(c):
    """A character as a CSS string, with newline as the escape '\\a '."""
    return '"\\a "' if c == "\n" else '"' + c.replace("\\", "\\\\").replace('"', '\\"') + '"'


tok = {c: i for i, c in enumerate(VOCAB)}
init = [tok[c] for c in PROMPT[-CTX:]]
prompt_css = " ".join(css_str(c) for c in PROMPT)

# ---- state: what survives from one tick to the next. name -> (syntax, initial value)
state = {f"t{i}": ("<integer>", str(init[i])) for i in range(CTX)}
state["seed"] = ("<integer>", "1234")
state["cnt"] = ("<integer>", "0")

# ---- registered properties. Registering a custom property gives it a type (number, colour), so animations can carry
# it and the maths can read it back as a value, not as text.
reg = []


def register(name, syntax, initial):
    reg.append(f'@property --{name} {{ syntax: "{syntax}"; inherits: true; initial-value: {initial}; }}')


register("clock", "<integer>", "0")
register("T", "<number>", "0.6")
for k, (syn, ini) in state.items():
    register(k, syn, ini)  # the next value, computed this tick
    register("c" + k, syn, ini)  # the current value, read from the latch
for p, n in (("x", nx), ("h", nh), ("o", no), ("p", no), ("w", no)):
    for i in range(n):
        register(f"{p}{i}", "<color>", "black")
for i in range(no):
    register(f"k{i}", "<integer>", "0")
register("nmx", "<color>", "black")
register("n", "<integer>", "0")
register("go", "<integer>", "1")

cpu = []
cpu.append("  /* 1. Read the state out of the latch. */")
for k, (_, ini) in state.items():
    cpu.append(f"  --__1{k}: var(--__2{k}, {ini}); --c{k}: var(--__1{k});")
cpu.append(f"  --__1text: var(--__2text, {prompt_css});")

cpu.append("\n  /* 2. Embedding: the last 8 characters pick 24 input colours. */")
for s in range(CTX):
    for k in range(E // 3):
        opts = " ".join(
            f"style(--ct{s}: {t}): color(srgb {f(m['emb'][t][3 * k])} {f(m['emb'][t][3 * k + 1])} {f(m['emb'][t][3 * k + 2])});"
            for t in range(V)
        )
        cpu.append(f"  --x{s * E // 3 + k}: if({opts} else: black);")

cpu.append("\n  /* 3. The network: two layers of colour maths, exactly as on the css page. */")
cpu += layer("x", CTX * E, m["W1"], m["b1"], "h", H, True)
cpu += layer("h", H, W2, b2, "o", Vp, False)

cpu.append("\n  /* 4. Sampling: add Gumbel noise (from sin and log of the seed) times the temperature to every logit.")
cpu.append("        The largest noisy logit is then a fair sample from the model. */")


def gumbel(n):
    """Gumbel noise for lane n, from the seed: a hash (fractional part of a big sin) gives u in (0, 1), and
    -T * log(-log(u)) is the noise. Adding it and taking the largest is the same as sampling from softmax(logits / T)."""
    u = f"clamp(0.000001, mod(abs(sin(var(--cseed) * 12.9898 + {n} * 78.233)) * 43758.5453, 1), 0.999999)"
    return f"calc(var(--T) * -1 * log(-1 * log({u})))"


for j in range(no):
    cpu.append(
        f"  --p{j}: color-mix(in srgb, var(--o{j}), color(srgb {gumbel(3 * j)} {gumbel(3 * j + 1)} {gumbel(3 * j + 2)}));"
    )

cpu.append(
    "\n  /* 5. Argmax, part one: the largest lane, by colour maths. max(A,B) = (A+B)/2 + |A-B|/2, lane by lane. */"
)


def cmax(a, b):
    """max(A, B) for colours, lane by lane: (A + B) / 2 + |A - B| / 2, with color-mix doing the halves."""
    neg = f"color(from {b} srgb calc(-1*r) calc(-1*g) calc(-1*b))"
    absd = f"color(from color-mix(in srgb, {a}, {neg}) srgb calc(abs(r)) calc(abs(g)) calc(abs(b)))"
    return f"color(from color-mix(in srgb, color-mix(in srgb, {a}, {b}), {absd}) srgb calc(r*2) calc(g*2) calc(b*2))"


level, names, lv = [f"var(--p{j})" for j in range(no)], [], 0
while len(level) > 1:
    nxt = []
    for i in range(0, len(level) - 1, 2):
        name = f"m{lv}_{i // 2}"
        register(name, "<color>", "black")
        names.append(name)
        cpu.append(f"  --{name}: {cmax(level[i], level[i + 1])};")
        nxt.append(f"var(--{name})")
    if len(level) % 2:
        nxt.append(level[-1])
    level, lv = nxt, lv + 1
cpu.append(f"  --nmx: color(from {level[0]} srgb calc(-1*max(r,g,b)) calc(-1*max(r,g,b)) calc(-1*max(r,g,b)));")

cpu.append("\n  /* 6. Argmax, part two: which lane was it? Subtract the max from every logit colour; the winner is the")
cpu.append("        lane that lands on zero. That makes a pure red, green or blue, and style() equality on that")
cpu.append("        colour is the one way to turn a colour back into a number. */")
step = lambda c: f"calc(max(0, sign({c} + {EPS})))"
for j in range(no):
    cpu.append(
        f"  --w{j}: color(from color-mix(in srgb, var(--p{j}), var(--nmx)) srgb {step('r')} {step('g')} {step('b')});"
    )
    cases = [("1 0 0", 1), ("0 1 0", 2), ("0 0 1", 3), ("1 1 0", 1), ("1 0 1", 1), ("0 1 1", 2), ("1 1 1", 1)]
    cpu.append(f"  --k{j}: if(" + " ".join(f"style(--w{j}: color(srgb {c})): {v};" for c, v in cases) + " else: 0);")
scores = ", ".join(f"calc(sign(var(--k{j})) * (1000 - {3 * j} - var(--k{j}) + 1))" for j in range(no))
cpu.append(f"  --n: calc(min({V - 1}, 1000 - max({scores})));")

cpu.append(
    "\n  /* 7. The next state: shift the context along, step the random seed, count, and append one character. */"
)
for i in range(CTX - 1):
    cpu.append(f"  --t{i}: var(--ct{i + 1});")
cpu.append(f"  --t{CTX - 1}: var(--n);")
cpu.append("  --seed: mod(var(--cseed) * 75 + 74, 65537);")
cpu.append("  --cnt: calc(var(--ccnt) + 1);")
cpu.append(f"  --go: calc(max(0, sign({LIMIT} - var(--ccnt))));")
cpu.append(
    '  --ch: if(style(--go: 0): ""; '
    + " ".join(f"style(--n: {t}): {css_str(c)};" for t, c in enumerate(VOCAB))
    + ' else: "");'
)
cpu.append("  --text: var(--__1text) var(--ch);")

store = (
    "\n".join(f"    --__2{k}: var(--__0{k}, {ini});" for k, (_, ini) in state.items())
    + f"\n    --__2text: var(--__0text, {prompt_css});"
)
execute = "\n".join(f"    --__0{k}: var(--{k});" for k in state) + "\n    --__0text: var(--text);"

machine = f"""
/* ================================================================================================================
   This stylesheet is the whole program. There is no script on this page.
   Registered properties (above) hold the state and the numbers. The clock, the latch, the model, the dice and the
   text are all below. Colours are used as registers: one colour holds three numbers, unclamped.
   ================================================================================================================ */

/* ===== The clock. One CSS animation steps a number through 0 1 2 3; the checkbox pauses it.
         Phases 1 and 3 do the work. Phases 0 and 2 are rests between them, and they are longer on purpose:
         if the browser ever skips a rest and goes straight from one working phase to the other, both
         latch animations update in the same pass and the state is lost. ===== */
.clock {{ --clock: 0; animation: tick {TICK_MS}ms step-end infinite paused; }}
.machine:has(#run:checked) .clock {{ animation-play-state: running; }}
.machine:has(#S0:checked) .clock {{ animation-duration: {TICK_MS * 2}ms; }}
.machine:has(#S2:checked) .clock {{ animation-duration: {TICK_MS * 6 // 10}ms; }}
@keyframes tick {{ 0% {{ --clock: 0 }} 35% {{ --clock: 1 }} 50% {{ --clock: 2 }} 85% {{ --clock: 3 }} }}

/* ===== Temperature: three radio buttons set one number. ===== */
.machine:has(#T0:checked) {{ --T: 0; }}
.machine:has(#T1:checked) {{ --T: 0.6; }}
.machine:has(#T2:checked) {{ --T: 1; }}

/* ===== The latch. A CSS property can't depend on itself, so the state goes round through two animations
         that are paused except for one clock phase each: "store" copies the saved state to where the
         maths reads it, "execute" saves what the maths produced. ===== */
.cpu {{
  animation: store 1ms infinite, execute 1ms infinite;
  animation-play-state: paused, paused;
  @container style(--clock: 1) {{ animation-play-state: running, paused }}
  @container style(--clock: 3) {{ animation-play-state: paused, running }}

{chr(10).join(cpu)}
}}
@keyframes store {{
  0%, 100% {{
{store}
  }}
}}
@keyframes execute {{
  0%, 100% {{
{execute}
  }}
}}

/* ===== What you see. None of it is drawn by script: every square and the text read the machine's properties. ===== */
#out::after {{ content: var(--text); }}
"""
sw = [
    f"#hv i:nth-child({i + 1}) {{ background: color(from var(--h{i}) srgb calc(r / 4) calc(g / 4) calc(b / 4)); }}"
    for i in range(nh)
]
sw += [
    f"#ov i:nth-child({i + 1}) {{ background: color(from var(--o{i}) srgb calc(r / 12 + .5) calc(g / 12 + .5) calc(b / 12 + .5)); }}"
    for i in range(no)
]
for n in range(V):
    c = "rgb"[n % 3]
    sw.append(
        f"#win i:nth-child({n + 1}) {{ background: color(from var(--w{n // 3}) srgb calc({c}) calc({c} * .82) calc({c} * .25)); color: color(from var(--w{n // 3}) srgb calc(.6 - {c} * .55) calc(.6 - {c} * .55) calc(.55 - {c} * .5)); }}"
    )

css = "\n".join(reg) + "\n" + machine + "\n".join(sw) + "\n"
weights = V * E + CTX * E * H + H + H * V + V
show = lambda c: {"\n": "↵", " ": "␣"}.get(c, c)
page = open(sys.argv[3] if len(sys.argv) > 3 else "tools/nojs_template.html").read()
excerpt = f"""/* the clock: an animation that counts */
.clock {{ animation: tick {TICK_MS}ms step-end infinite; }}
@keyframes tick {{ 0% {{ --clock: 0 }} 35% {{ --clock: 1 }} 50% {{ --clock: 2 }} 85% {{ --clock: 3 }} }}

/* the latch: two animations that only run on one clock phase each */
.cpu {{
  animation: store 1ms infinite, execute 1ms infinite;
  animation-play-state: paused, paused;
  @container style(--clock: 1) {{ animation-play-state: running, paused }}
  @container style(--clock: 3) {{ animation-play-state: paused, running }}
}}

/* a colour back into a number: the only bridge is asking "is it exactly this colour?" */
{next(l.strip() for l in cpu if l.strip().startswith("--k0:"))}

/* one character onto the end of the text */
--text: var(--__1text) var(--ch);"""
out = (
    page.replace("/*CSS*/", css)
    .replace("<!--EXCERPT-->", html.escape(excerpt))
    .replace("<!--HV-->", "<i></i>" * nh)
    .replace("<!--OV-->", "<i></i>" * no)
    .replace("<!--WIN-->", "".join(f"<i>{html.escape(show(c))}</i>" for c in VOCAB))
    .replace("{{cssKB}}", str(round(len(css) / 1e3)))
    .replace("{{weights}}", f"{weights:,}")
    .replace("{{tick}}", str(TICK_MS))
    .replace("{{limit}}", str(LIMIT))
    .replace("{{props}}", str(len(reg)))
)
assert "<script" not in out.lower()
open(sys.argv[2], "w").write(out)
print(f"{len(out) / 1e3:.0f} KB page, {len(css) / 1e3:.0f} KB css, {len(reg)} registered properties")
