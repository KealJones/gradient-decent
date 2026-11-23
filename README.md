# Gradient Decent

Small neural networks, mostly language models, run by the parts of a browser that were built for drawing and
styling, not for maths: canvas blend modes, CSS colours and transforms, SVG filters, the audio engine, and whatever else turns out to be able to multiply. No WebGPU, WebGL or WASM in the
forward pass.

The name is the joke. Gradient descent is how a model gets trained. No page here trains anything and nothing descends: they only run models, and they do it with actual colours and gradients. So it's gradient, and it's decent.

Each page runs a real model, explains the trick step by step, and races it against a plain JavaScript loop.

There are seven so far:

| | model | what it does |
|---|---|---|
| [`chat/`](chat/) | [raincandy-u/TinyChat-1776K](https://huggingface.co/raincandy-u/TinyChat-1776K) (Apache 2.0), 1.8M weights, 3 layers | Multi-turn chat at the level of a three-year-old. No world knowledge. |
| [`stories/`](stories/) | [hidude562/tiny-tiny-stories](https://huggingface.co/hidude562/tiny-tiny-stories) (MIT), 1M ternary weights, 5 layers | Wobbly children's stories from a prompt. |
| [`css/`](css/) | trained here on lowercase Tiny Shakespeare, 21K weights, one hidden layer | Shakespeare-shaped babble, one character at a time, with the whole forward pass in a stylesheet. |
| [`no-js/`](no-js/) | the same 21K-weight model as `css/` | The same babble from a page with no script at all: the clock, memory, sampling and text are CSS too. |
| [`cards/`](cards/) | the same model as `css/` | The same babble, with both matrix multiplies done by CSS transforms on a wall of tilting cards. |
| [`audio/`](audio/) | the same model as `css/`, shipped as a 5-second float WAV | The same babble, with the whole forward pass in one Web Audio render per character. You can play the weights and each character's render. |
| [`filter/`](filter/) | trained here on MNIST shrunk to 14×14, 10K weights, three layers | Reads a digit you draw, with the whole forward pass in one SVG `<filter>`. |

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

## The transform one

[`cards/`](cards/) runs the same model (it reads `css/model.json`) with the matrix maths done by the transform
engine. A `transform` list is a product of 4×4 matrices, and the computed `transform` is that product multiplied
out to one `matrix3d()`.

- **A neuron is a transform list.** Every matrix has an input in its depth slot and two constants in its
  perspective row. Multiplying them down the list adds up input × weight in the two slots that tilt the element,
  so each card ends up tilted by two neurons. A layer-1 list is 74 matrices, a layer-2 list is 194.
- **The constants are differences of neighbouring weights**, worked out at build time, because an input gets
  multiplied by every constant after it. The bias is one more input fixed at 1.
- **The cards on the page are the calculation.** The script reads the result out of the same `transform` that
  tilts each card. The walls show the tilt scaled down (`scaleZ()` on the parent) so it stays on screen.
- **ReLU is not a transform.** A transform list is linear, so the script copies 192 numbers, as text, from the
  first wall's computed transforms into variables on the second, where each is used as `max(0, var(--h5))`.

So: transforms do every multiply, add and bias in both layers; class selectors do the embedding lookup; CSS
`max()` does the ReLU; JavaScript sets class names, copies the 192 numbers across, reads 19 matrices back and
samples. In headless Chrome 153 it matches a plain JavaScript run to about 0.0007 on logits from −270 to 13 (top
character 40 of 40; the limit is that Chrome prints a computed matrix with six significant digits) at 50 to 90 ms
per character, around a thousand times slower than the JavaScript loop. The stylesheet is 812 KB: 10,790
`matrix3d()` functions for 21K weights. `window.selfTest()` runs the comparison.

```
python3 tools/compile_cards.py css/model.json tools/cards_template.html cards/index.html
```


## The audio one

[`audio/`](audio/) runs the `css/` model as a Web Audio graph in an `OfflineAudioContext`. One render per character,
with no script touching the numbers between the layers.

- **The weights are a sound file.** `audio/weights.wav` is mono, 8 kHz, 32-bit float, every weight divided by 16.
  `decodeAudioData` returns it bit for bit, as long as the decoding context runs at 8 kHz too.
- **A matrix multiply is a `ConvolverNode`.** Each neuron's weights are a 72-sample slot of the impulse response,
  written back to front. The dot product is the last sample of the slot; the rest is junk.
- **ReLU is a `WaveShaperNode`** with the three-point curve `[0, 0, 1]`, after `GainNode`s bring the signal inside
  the −1…1 the node clamps to.
- **A click track picks out the answers.** A `GainNode` whose gain is driven by a track that is 1 on every 72nd
  sample mutes the junk, and layer 2's impulse response is spaced the same 72 samples apart, so layer 1 feeds
  layer 2 directly and the 37 logits come out as the last 37 samples.

The script builds the graph, copies 8 embedding rows into the input buffer, reads 37 samples and samples a
character. In headless Chrome 153 it matches a plain JavaScript run to about 0.0002 on logits as large as 400 (top
character 40 of 40) at 10 to 45 ms per character, a few hundred times slower than the plain loop.
`window.selfTest()` runs the comparison and the WAV check. Playback is there but nobody has heard it yet: the test
browser has no speakers.

Rebuild the sound file from the CSS page's model:

```
python3 tools/export_audio.py css/model.json audio/weights.wav audio/layout.json
```


## The filter one

[`filter/`](filter/) is a digit classifier, not a language model. Its forward pass is one SVG `<filter>` with
126 primitives, applied to a canvas with `ctx.filter = 'url(#net)'`.

- **A layer is `feConvolveMatrix`.** A kernel, a `bias`, and a result that is a colour, so it is clamped to 0..1.
  The clamp is the activation function.
- **Max pooling is `feMorphology`.** `dilate` with radius 1 is the largest value in each 3×3 window.
- **A hidden unit is a big kernel worked out at one pixel**, by giving the primitive a 1×1 output area. `feOffset`
  moves the pixel to where the next layer reads it, and `feTile` makes the 8 copies of the input.
- **The ten scores are ten pixels.** The script draws the 14×14 input, and reads those pixels back.

The network is 8 kernels of 5×5, a 3×3 max, 48 hidden units (6 per feature map) and 10 outputs: 10,154 weights.
Filter pictures are 8 bits per channel, `feConvolveMatrix` takes at most 256 kernel entries, and Chrome stores
kernels of 28 entries or more at 8 bits each. The network was trained with all of that in place, so the filter
loses almost nothing. On the 10,000 MNIST test digits, in headless Chrome 153 on a CPU:

| | of 10,000 |
|---|---|
| plain JavaScript, same weights | 9,824 (98.24%) |
| the filter | 9,822 (98.22%) |
| same answer from both | 9,996 |
| largest difference in a score (scores are 0 to 1) | 0.019 |

A JavaScript version that rounds each layer to 8 bits, as the filter has to, is never more than one 8-bit step
(0.0039) from the filter and gives the same answer on all 10,000. The filter takes about 6 ms a digit, about 40 times
slower than the plain JavaScript loop. `window.selfTest()` on the page runs the comparison on the first 1,000 test digits.
Filters ran in software here; a GPU path was not tested and could round differently.

Rebuild it (about 15 minutes on two CPU cores; the trainer downloads MNIST into the folder you name):

```
pip install torch numpy
python3 tools/train_filter.py mnist 12      # writes filter/model.json and filter/test.json
python3 tools/compile_filter.py filter/model.json tools/filter_template.html filter/index.html
```

`tools/check_filter.py` runs the self-test in headless Chrome against a local server.


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
| speed with the live tiles and clocks on | about 14 tokens/s | about 20 tokens/s |
| canvas multiplies per token | 22 | 36 |
| download | 5.9 MB | 743 KB |

The chat model's weights are full precision, so its pictures hold 8 bits of magnitude per weight with one scale per
row. Against the original float weights that costs a logit error of about 0.06 and no change in the greedy tokens.
The chat reference itself matches Hugging Face transformers to 0.00001.

A plain JavaScript loop does the same multiply about 25 times faster. The canvas is here because you can watch it.

The rotary positions are done by CSS transforms ([`lib/clocks.js`](lib/clocks.js)): each pair of numbers is a
zero-size element placed with `translate` inside an element with `rotate`, and `getBoundingClientRect()` reads the
answer. The elements are the dots on the clock faces next to the live tiles. Against `Math.cos`/`Math.sin` the
worst error is 0.0006% to 0.003% of the longest pair on a clock (it grows with how far the clocks sit from the corner
of the window), and it costs about 5 ms a token on chat and 7 ms on stories. `M.ropeEngine = 'js'` switches it off;
`window.rotaryTest()` and `window.rotaryCost()` measure it.

Still plain JavaScript: the norms, attention over the cache, the tokenizers and, on the stories
page, sampling. The chat page picks its token with layout: the candidates are boxes in a flex row whose widths are
their probabilities, and a hit test says which box a randomly placed needle is on ([`lib/pick.js`](lib/pick.js);
`window.samplingTest()` checks it against the JavaScript sampler). Not tried yet: Safari, phones, a machine with a
real GPU.

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
