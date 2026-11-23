# Gradient Decent

Small language models run by the parts of a browser that were built for drawing and styling, not for maths: canvas
blend modes, CSS colours, and whatever else turns out to be able to multiply. No WebGPU, WebGL or WASM in the
forward pass.

The name is the joke. Gradient descent is how a model gets trained. Nothing here is trained and nothing descends:
these only run models, and they do it with actual colours and gradients. So it's gradient, and it's decent.

Each page runs a real model, explains the trick step by step, and races it against a plain JavaScript loop.

There are four so far. The first two run on the canvas; the other two do it in CSS:

| | model | what it does |
|---|---|---|
| [`chat/`](chat/) | [raincandy-u/TinyChat-1776K](https://huggingface.co/raincandy-u/TinyChat-1776K) (Apache 2.0), 1.8M weights, 3 layers | Multi-turn chat at the level of a three-year-old. No world knowledge. |
| [`stories/`](stories/) | [hidude562/tiny-tiny-stories](https://huggingface.co/hidude562/tiny-tiny-stories) (MIT), 1M ternary weights, 5 layers | Wobbly children's stories from a prompt. |
| [`css/`](css/) | trained here on lowercase Tiny Shakespeare, 21K weights, one hidden layer | Shakespeare-shaped babble, one character at a time, with the whole forward pass in a stylesheet. |
| [`no-js/`](no-js/) | the same 21K-weight model as `css/` | The same babble from a page with no script at all: the clock, memory, sampling and text are CSS too. |

## The canvas ones

Every matrix multiply is done on a 2D canvas.

- **The weights are pictures.** One pixel per weight, red for positive and green for negative.
- **The multiply is a blend mode.** The input vector is a 2-pixel strip stretched over the weight picture with
  `multiply`.
- **The sum is a squash.** The product is halved in width over and over until each row is one pixel, then read
  back in a single `getImageData`.
- **The activation function is a gradient.** SiLU is a canvas linear gradient, rendered once and eyedropped.

The shared engine is two files: [`lib/blend.js`](lib/blend.js) (the canvas matrix multiply and the gradient) and
[`lib/llama.js`](lib/llama.js) (a small Llama-style forward pass built on it). [`lib/ui.js`](lib/ui.js) and
[`lib/page.css`](lib/page.css) hold the page pieces every project shares: the live tiles, the race, the weight
gallery and the explainer layout.

## The CSS one

[`css/`](css/) does the forward pass with no canvas and no script maths. Chrome stores a registered
`@property --h3 { syntax: '<color>' }` as unclamped floats, so `color(srgb -4.8 7.12 52.5)` is a three-lane vector
register. (Alpha clamps to 0 to 1, so it's no use as a lane.)

- **A 3×3 weight block is a relative colour.** `color(from var(--x5) srgb calc(r*a + g*b + b*c) …)`: a colour matrix.
- **The sum is a `color-mix()` tree.** Mixing averages, so the weights are pre-scaled by the number of leaves.
- **ReLU is `max(r, 0)`.** Both layers resolve in one style recalc, chained through the registered properties.
- **The embedding lookup is class names.** `.s3t12` sets the colours for "token 12 at position 3".

The script only puts the last 8 characters into class names, reads 12 computed colours back and samples. In
headless Chrome 153 it matches a plain JavaScript run of the same weights to 0.002 (top character 40 of 40) at
about 31 ms per character, which is 85 to 770 times slower than a plain JavaScript loop over the same weights (the
page has a live race and a scaling chart). The stylesheet is 480 KB for 21K weights, which is the catch: CSS is a terrible weight
format. `window.selfTest()` runs the comparison.

Rebuild it with numpy only (about two minutes on a laptop CPU):

```
curl -L https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt -o shakespeare.txt
python3 tools/train_css.py shakespeare.txt 40000                 # writes model.json
python3 tools/compile_css.py model.json tools/css_template.html css/index.html
```

`tools/check_css.cjs` runs the self-test in headless Chrome against a local server.

## The one with no script

[`no-js/`](no-js/) is the css model with the remaining JavaScript replaced by CSS, so the page has no script tag
and still writes text by itself.

- **The clock is an animation** stepping a registered integer through 0, 1, 2, 3.
- **The memory is a latch** made of two 1 ms animations that are paused except on one clock phase each (the trick
  from Lyra Rebane's x86CSS). The last 8 characters, a random seed and the text so far live in it.
- **Sampling is Gumbel noise** made from the seed with `sin()`, `mod()` and `log()`, added to the logits.
- **Argmax is colour maths**, then `if(style(--w4: color(srgb 0 1 0)): …)` turns the winning colour into a number.
  Asking whether a colour equals an exact colour is the only way found to get a number back out of one.
- **The text** is a custom property that gets one more string each tick, shown with `content`.

In headless Chrome 153 on a CPU it wrote 0.8 characters a second, and at temperature 0 all 37 characters of a
45-second run matched a Python run of the same weights. `python3 tools/compile_nojs.py css/model.json
no-js/index.html` rebuilds it from [`tools/nojs_template.html`](tools/nojs_template.html).

## Run it

```
python3 -m http.server
```

Then open http://localhost:8000. The canvas pages need a float16 2D canvas, so a recent Chrome or Edge. The CSS page needs relative colour syntax and registered properties; it's only been tried in Chrome.

`window.selfTest()` in the console on either page compares the canvas against a plain numpy run of the same model.

## How it did

Headless Chrome 153, CPU only, no GPU:

| check | chat | stories |
|---|---|---|
| greedy tokens matching the numpy reference | 14 of 14 | 60 of 60 |
| largest logit error | 0.07 on values up to 21 | 0.022 on values up to 14.5 |
| speed with the live tiles on | about 15 tokens/s | about 20 tokens/s |
| canvas multiplies per token | 22 | 36 |
| download | 5.9 MB | 743 KB |

The chat model's weights are full precision, so its pictures hold 8 bits of magnitude per weight with one scale per
row. Against the original float weights that costs a logit error of about 0.06 and no change in the greedy tokens.
The chat reference itself matches Hugging Face transformers to 0.00001.

A plain JavaScript loop does the same multiply about 25 times faster. The canvas is here because you can watch it.

Still plain JavaScript: the norms, rotary positions, attention over the cache, the tokenizers and sampling. Not
tried yet: Safari, phones, a machine with a real GPU.

## Rebuilding the weights

The exporters in [`tools/`](tools/) turn a checkpoint into `model.json` (the pictures, the tokenizer and a
reference run):

```
pip install torch sentencepiece tokenizers safetensors pillow numpy
python3 tools/export_stories.py path/to/tiny-tiny-stories stories/model.json
python3 tools/export_chat.py path/to/TinyChat-1776K chat/model.json
```

Each first argument is a folder holding that model's files from its Hugging Face repo. Add `--check` to the chat
exporter (needs `transformers`) to compare the numpy reference with the real model.

The notes on everything else that was tried on the way (CSS `calc()`, CSS layout, a Web Audio convolver, a
flexbox softmax) are in [lm-experiments](https://github.com/KealJones/lm-experiments/tree/main/experiments/gradient-decent).
