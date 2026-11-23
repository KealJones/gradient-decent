# Gradient Decent

A language model that runs on colour blending. Every matrix multiply is done on a 2D canvas: no WebGPU, no WebGL,
no WASM. It's made of gradients, and it's decent. (Nothing is descended. This is inference.)

- **The weights are pictures.** One pixel per weight, red for +1 and green for -1.
- **The multiply is a blend mode.** The input vector is a 2-pixel strip stretched over the weight picture with
  `multiply`.
- **The sum is a squash.** The product is halved in width over and over until each row is one pixel, then read
  back in a single `getImageData`.
- **The activation function is a gradient.** SiLU is a canvas linear gradient, rendered once and eyedropped.

There are two so far, both live under one page:

| | model | what it does |
|---|---|---|
| [`chat/`](chat/) | [raincandy-u/TinyChat-1776K](https://huggingface.co/raincandy-u/TinyChat-1776K) (Apache 2.0), 1.8M weights, 3 layers | Multi-turn chat at the level of a three-year-old. No world knowledge. |
| [`stories/`](stories/) | [hidude562/tiny-tiny-stories](https://huggingface.co/hidude562/tiny-tiny-stories) (MIT), 1M ternary weights, 5 layers | Wobbly children's stories from a prompt. |

The shared engine is two files: [`lib/blend.js`](lib/blend.js) (the canvas matrix multiply and the gradient) and
[`lib/llama.js`](lib/llama.js) (a small Llama-style forward pass built on it).

## Run it

```
python3 -m http.server
```

Then open http://localhost:8000. It needs a float16 2D canvas, so a recent Chrome or Edge.

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
