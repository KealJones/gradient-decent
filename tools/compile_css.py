"""Compile model.json into a single HTML page whose forward pass is CSS color math."""
import json, sys
m = json.load(open(sys.argv[1]))
V, CTX, E, H = len(m['vocab']), m['ctx'], m['E'], m['H']
Vp = -(-V // 3) * 3  # pad the vocab to whole colours; padded logits get a huge negative bias
m['W2'] = [row + [0.0] * (Vp - V) for row in m['W2']]
m['b2'] = m['b2'] + [-1e4] * (Vp - V)
f = lambda x: f'{x:.5g}'

def pow2(n):
    p = 1
    while p < n: p *= 2
    return p

def tree(leaves):
    while len(leaves) > 1:
        leaves = [f'color-mix(in srgb,{leaves[i]},{leaves[i+1]})' for i in range(0, len(leaves), 2)]
    return leaves[0]

def layer(src, n_in, W, b, dst, n_out, relu):
    """y = W^T x + b, with x packed 3-per-color in --{src}i and y packed into --{dst}j."""
    K = n_in // 3; Kp = pow2(K + 1)  # color-mix averages, so pre-scale weights by Kp
    rules = []
    for j in range(n_out // 3):
        leaves = []
        for i in range(K):
            ch = []
            for o in range(3):
                w = [W[i * 3 + c][j * 3 + o] * Kp for c in range(3)]
                ch.append(f'calc(r*{f(w[0])} + g*{f(w[1])} + b*{f(w[2])})')
            leaves.append(f'color(from var(--{src}{i}) srgb {" ".join(ch)})')
        leaves.append('color(srgb ' + ' '.join(f(b[j * 3 + o] * Kp) for o in range(3)) + ')')
        leaves += ['color(srgb 0 0 0)'] * (Kp - len(leaves))
        expr = tree(leaves)
        if relu:
            expr = f'color(from {expr} srgb calc(max(r,0)) calc(max(g,0)) calc(max(b,0)))'
        rules.append(f'--{dst}{j}:{expr};')
    return rules

nx, nh, no = CTX * E // 3, H // 3, Vp // 3
props = [f'@property --{p}{i}{{syntax:"<color>";inherits:true;initial-value:black}}'
         for p, n in (('x', nx), ('h', nh), ('o', no)) for i in range(n)]
emb = []
for s in range(CTX):
    for t in range(V):
        e = m['emb'][t]
        decl = ''.join(f'--x{s*E//3+k}:color(srgb {f(e[3*k])} {f(e[3*k+1])} {f(e[3*k+2])});' for k in range(E // 3))
        emb.append(f'.s{s}t{t}{{{decl}}}')
net = '#m{' + ''.join(layer('x', CTX * E, m['W1'], m['b1'], 'h', H, True)
                     + layer('h', H, m['W2'], m['b2'], 'o', Vp, False)) + '}'
swatch = ''.join(f'#hv i:nth-child({i+1}){{background:color(from var(--h{i}) srgb calc(r/4) calc(g/4) calc(b/4))}}' for i in range(nh))
swatch += ''.join(f'#ov i:nth-child({i+1}){{background:color(from var(--o{i}) srgb calc(r/12 + .5) calc(g/12 + .5) calc(b/12 + .5))}}' for i in range(no))
css = '\n'.join(props + emb + [net, swatch])
tpl = open(sys.argv[2]).read()
out = (tpl.replace('/*CSS*/', css)
          .replace('/*VOCAB*/', json.dumps(''.join(m['vocab'])))
          .replace('/*META*/', json.dumps({'ctx': CTX, 'no': no, 'nh': nh})))
open(sys.argv[3], 'w').write(out)
print(f'{len(out)/1e3:.0f} KB, {len(css)/1e3:.0f} KB css')
