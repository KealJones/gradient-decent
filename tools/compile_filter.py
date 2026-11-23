"""Compile a trained model.json into one SVG <filter> and write the page around it.

    python3 tools/compile_filter.py filter/model.json tools/filter_template.html filter/index.html

The network (see train_filter.py):

    14x14 picture -> 8 kernels of 5x5, clamp -> 3x3 max -> 48 hidden units, clamp -> 10 scores, clamp

A filter's pixels are 8-bit colours clamped to 0..1, so "clamp" costs nothing: it's what happens to every
primitive's result anyway. Everything is grey (red = green = blue) and alpha stays 1.

Everything happens inside one 64 x 98 picture:

    rows 0..31    the working area: 8 tiles of 16x16 in a 4x2 grid. The digit sits in the middle 14x14 of a
                  tile, so every tile has a 1-pixel black frame and neighbouring digits are 2 pixels apart,
                  which is the zero padding the 5x5 kernels need.
    rows 32..63   a copy of the 8 feature maps before the max, only so they can be looked at
    rows 64..95   a copy of the 8 feature maps after the max, same reason
    row  96       the 48 hidden units, one pixel each
    row  97       the 10 scores, one pixel each. JavaScript reads these.

The primitives:

    feTile            copies the digit into all 8 tiles
    feConvolveMatrix  a kernel and a bias. One per feature map, limited to that map's 14x14 with x/y/width/height
    feMorphology      dilate with radius 1 = the largest value in each 3x3 window = max pooling (stride 1)
    feConvolveMatrix  a hidden unit is a 14x14 kernel worked out at ONE pixel (a 1x1 primitive subregion)
    feOffset          moves that pixel into row 96 (a kernel's target pixel has to be inside the kernel, so the
                      unit can't be written straight to where the next layer wants it)
    feConvolveMatrix  a score is a 48x1 kernel over row 96, again at one pixel

Two things about feConvolveMatrix in Chrome that shape this file:

    * kernelMatrix is applied rotated by 180 degrees (it's a true convolution), so kernels are written backwards.
    * at most 256 entries, and from 28 entries up the kernel is stored 8 bits per entry between its smallest and
      largest value. train_filter.py already put those kernels on that grid.

The template's placeholders: <!--FILTER--> (the filter), <!--EXCERPT--> (real lines from it, shortened, shown
on the page) and /*META*/ (sizes the page and its script use).
"""
import html
import json
import sys

m = json.load(open(sys.argv[1]))
K, M, S = m['K'], m['M'], m['size']
T = S + 2                      # tile pitch
COLS = 4
W, H = COLS * T, 3 * 2 * T + 2  # 64 x 98
HID_Y, OUT_Y = 3 * 2 * T, 3 * 2 * T + 1


def f(x):
    return f'{x:.7g}'


def kernel(rows):
    """Rows of cross-correlation weights -> kernelMatrix text. Reversed, because the filter flips it back."""
    flat = [v for r in rows for v in r]
    return ' '.join(f(v) for v in reversed(flat))


def tile(k):                   # top-left pixel of map k's 14x14
    return (k % COLS) * T + 1, (k // COLS) * T + 1


CONV = 'divisor="1" preserveAlpha="true" edgeMode="none"'
out = []
out.append(f'<filter id="net" filterUnits="userSpaceOnUse" x="0" y="0" width="{W}" height="{H}" color-interpolation-filters="sRGB">')
out.append('<!-- opaque black to build on, so alpha is 1 everywhere -->')
out.append('<feFlood flood-color="#000" result="black"/>')
out.append('<!-- the 16x16 input picture, copied into all 8 tiles -->')
out.append(f'<feOffset in="SourceGraphic" x="0" y="0" width="{T}" height="{T}" result="digit"/>')
out.append(f'<feTile in="digit" x="0" y="0" width="{W}" height="{2 * T}" result="copies"/>')

out.append('<!-- layer 1: one 5x5 kernel and bias per tile. Results below 0 or above 1 are clamped: the activation -->')
for k in range(K):
    x, y = tile(k)
    out.append(f'<feConvolveMatrix in="copies" x="{x}" y="{y}" width="{S}" height="{S}" order="5" '
               f'kernelMatrix="{kernel(m["conv"]["w"][k])}" bias="{f(m["conv"]["b"][k])}" {CONV} result="c{k}"/>')
out.append('<feMerge result="maps"><feMergeNode in="black"/>' + ''.join(f'<feMergeNode in="c{k}"/>' for k in range(K)) + '</feMerge>')

out.append('<!-- max pooling: the largest value in each 3x3 window -->')
out.append('<feMorphology in="maps" operator="dilate" radius="1" result="pooled"/>')

out.append('<!-- layer 2: 48 hidden units. Each is a 14x14 kernel over one map, worked out at one pixel, then moved to row 96 -->')
for k in range(K):
    x, y = tile(k)
    for j in range(M):
        n = k * M + j
        out.append(f'<feConvolveMatrix in="pooled" x="{x + j}" y="{y}" width="1" height="1" order="{S}" targetX="{j}" targetY="0" '
                   f'kernelMatrix="{kernel(m["hidden"]["w"][k][j])}" bias="{f(m["hidden"]["b"][k][j])}" {CONV} result="u{n}"/>')
        out.append(f'<feOffset in="u{n}" x="{n}" y="{HID_Y}" width="1" height="1" dx="{n - (x + j)}" dy="{HID_Y - y}" result="h{n}"/>')
out.append('<feMerge result="hidden"><feMergeNode in="black"/>' + ''.join(f'<feMergeNode in="h{n}"/>' for n in range(K * M)) + '</feMerge>')

out.append('<!-- layer 3: ten scores. Each is a 48x1 kernel over the hidden row, worked out at one pixel -->')
for c in range(10):
    out.append(f'<feConvolveMatrix in="hidden" x="{c}" y="{HID_Y}" width="1" height="1" order="{K * M} 1" targetX="{c}" targetY="0" '
               f'kernelMatrix="{kernel([m["out"]["w"][c]])}" bias="{f(m["out"]["b"][c])}" {CONV} result="s{c}"/>')
out.append('<feMerge result="scores"><feMergeNode in="s0"/>' + ''.join(f'<feMergeNode in="s{c}"/>' for c in range(1, 10)) + '</feMerge>')
out.append(f'<feOffset in="scores" x="0" y="{OUT_Y}" width="10" height="1" dx="0" dy="1" result="scoresRow"/>')

out.append('<!-- the picture that comes out: every stage, so the page can show them, and the ten scores in the last row -->')
out.append(f'<feOffset in="maps" x="0" y="{2 * T}" width="{W}" height="{2 * T}" dx="0" dy="{2 * T}" result="mapsShown"/>')
out.append(f'<feOffset in="pooled" x="0" y="{4 * T}" width="{W}" height="{2 * T}" dx="0" dy="{4 * T}" result="pooledShown"/>')
out.append(f'<feOffset in="hidden" x="0" y="{HID_Y}" width="{K * M}" height="1" result="hiddenRow"/>')
out.append('<feMerge><feMergeNode in="black"/><feMergeNode in="copies"/><feMergeNode in="mapsShown"/><feMergeNode in="pooledShown"/>'
           '<feMergeNode in="hiddenRow"/><feMergeNode in="scoresRow"/></feMerge>')
out.append('</filter>')
markup = '\n'.join(out)


def short(line, keep=6):
    """A real line with its kernel cut to the first few numbers."""
    a = line.index('kernelMatrix="') + 14
    b = line.index('"', a)
    nums = line[a:b].split(' ')
    return line[:a] + ' '.join(nums[:keep]) + f' … {len(nums) - keep} more' + line[b:] if len(nums) > keep else line


lines = markup.split('\n')
first = lambda s: next(l for l in lines if s in l)
excerpt = '\n'.join([
    first('<feTile'),
    '',
    '<!-- one of the 8 first-layer kernels, whole -->',
    first('result="c0"'),
    '',
    first('<feMorphology'),
    '',
    '<!-- one of the 48 hidden units: a 14x14 kernel worked out at a single pixel, then moved into row 96 -->',
    short(first('result="u0"')),
    first('result="h0"'),
    '',
    '<!-- the score for "0": a 48x1 kernel over row 96 -->',
    short(first('result="s0"')),
])

weights = K * 25 + K + K * M * S * S + K * M + 10 * K * M + 10
meta = {'K': K, 'M': M, 'size': S, 'tile': T, 'w': W, 'h': H, 'hidY': HID_Y, 'outY': OUT_Y,
        'weights': weights, 'primitives': markup.count('<fe') - markup.count('<feMergeNode'), 'kb': round(len(markup) / 1024)}

page = open(sys.argv[2]).read()
page = page.replace('<!--FILTER-->', markup).replace('<!--EXCERPT-->', html.escape(excerpt)).replace('/*META*/', json.dumps(meta))
open(sys.argv[3], 'w').write(page)
print(f'{meta["primitives"]} primitives, {meta["weights"]} weights, {meta["kb"]} KB of filter -> {sys.argv[3]}')
