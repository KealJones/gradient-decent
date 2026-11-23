# Gradient Decent

A language model that runs on colour blending. Every matrix multiply is done on a 2D canvas: no WebGPU, no WebGL,
no WASM. It's made of gradients, and it's decent. (Nothing is descended. This is inference.)

- **The weights are pictures.** One pixel per weight, red for +1 and green for -1.
- **The multiply is a blend mode.** The input vector is a 2-pixel strip stretched over the weight picture with
  `multiply`.
- **The sum is a squash.** The product is halved in width over and over until each row is one pixel, then read
  back in a single `getImageData`.
- **The activation function is a gradient.** SiLU is a canvas linear gradient, rendered once and eyedropped.

The model is [hidude562/tiny-tiny-stories](https://huggingface.co/hidude562/tiny-tiny-stories) (MIT): about a
million weights, 5 layers, ternary {-1, 0, +1}, trained on TinyStories. It writes wobbly children's stories.

## Run it

```
python3 -m http.server
```

Then open http://localhost:8000. It needs a float16 2D canvas, so a recent Chrome or Edge.

`window.selfTest()` in the console compares the canvas against a plain numpy run of the same model.

## How it did

Headless Chrome 153, CPU only, no GPU:

| check | result |
|---|---|
| greedy tokens matching the numpy reference | 60 of 60 |
| largest logit error | 0.022 on values up to 14.5 |
| speed | about 20 tokens/s with the live tiles, about 30 without |
| download | 743 KB for all 36 weight pictures and the tokenizer |

A plain JavaScript loop does the same multiply about 25 times faster. The canvas is here because you can watch it.

Still plain JavaScript: the norms, rotary positions, attention over the cache, and sampling. Not tried yet:
Safari, phones, a machine with a real GPU.

## Rebuilding the weights

`export.py` turns the checkpoint into `model.json` (the pictures, the tokenizer and a reference run):

```
pip install torch sentencepiece pillow numpy
python3 export.py path/to/tiny-tiny-stories model.json
```

The first argument is a folder holding `best.pt` and `tokenizer.model` from the model's Hugging Face repo.

The notes on everything else that was tried on the way (CSS `calc()`, CSS layout, a Web Audio convolver, a
flexbox softmax) are in [lm-experiments](https://github.com/KealJones/lm-experiments/tree/main/experiments/gradient-decent).
