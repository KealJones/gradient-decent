#!/usr/bin/env python3
"""
Gradient Decent / filter: train the digit classifier that tools/compile_filter.py turns into one SVG <filter>.

    python3 tools/train_filter.py [mnist_dir] [epochs]      # writes filter/model.json and filter/test.json

The network is shaped by what SVG filter primitives can do in Chrome (all of this was measured, see the page):

  * every intermediate picture is 8 bits per channel and clamped to 0..1, so every activation is
    clamp(x, 0, 1) rounded to a multiple of 1/255. The clamp is the activation function.
  * feConvolveMatrix takes at most 256 kernel entries, so no single primitive can see every feature map.
  * kernels of 28 entries or more are stored by Chrome as an 8-bit picture stretched between the kernel's
    smallest and largest entry. Those kernels are trained on that 256-level grid, so nothing is lost later.

  input   14x14 grey picture (MNIST 28x28, averaged 2x2)
  conv    K 5x5 kernels + bias, clamp          -> K maps of 14x14      (K feConvolveMatrix)
  pool    3x3 max, stride 1                    -> K maps of 14x14      (one feMorphology dilate)
  hidden  M units per map, each sees its own
          whole map (196 weights) + bias, clamp -> K*M numbers          (K*M feConvolveMatrix)
  out     10 units over the K*M numbers, clamp -> 10 scores in 0..1    (10 feConvolveMatrix)

Training rounds the activations and snaps the big kernels to their grid in the forward pass, and lets the
gradient straight through. The loss is cross-entropy on TEMP * score.
"""
import gzip, json, os, sys, base64, urllib.request
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

K, M, TEMP = 8, 6, 12.0
MIRROR = 'https://raw.githubusercontent.com/fgnt/mnist/master/'
FILES = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']


def load_mnist(d):
    os.makedirs(d, exist_ok=True)
    out = []
    for f in FILES:
        p = os.path.join(d, f)
        if not os.path.exists(p):
            print('downloading', f); urllib.request.urlretrieve(MIRROR + f, p)
        raw = gzip.open(p).read()
        out.append(np.frombuffer(raw, np.uint8, offset=16).reshape(-1, 28, 28) if 'images' in f else np.frombuffer(raw, np.uint8, offset=8))
    return out


def small(x28):
    """28x28 floats in 0..1 -> 14x14, each pixel the mean of a 2x2 block, rounded to 8 bits like a canvas would."""
    return torch.round(F.avg_pool2d(x28, 2) * 255) / 255


def augment(x28):
    """Small random shifts, scales, rotations and shears, so hand-drawn digits that aren't MNIST-neat still work."""
    n = x28.shape[0]
    ang = (torch.rand(n) - .5) * 2 * 0.21
    sc = 1 + (torch.rand(n) - .5) * 2 * 0.15
    sh = (torch.rand(n) - .5) * 2 * 0.15
    t = (torch.rand(n, 2) - .5) * 2 * 0.14          # up to 2 pixels of 28
    A = torch.zeros(n, 2, 3)
    A[:, 0, 0] = torch.cos(ang) / sc; A[:, 0, 1] = -torch.sin(ang) / sc + sh
    A[:, 1, 0] = torch.sin(ang) / sc; A[:, 1, 1] = torch.cos(ang) / sc
    A[:, :, 2] = t
    return F.grid_sample(x28, F.affine_grid(A, x28.shape, align_corners=False), align_corners=False).clamp(0, 1)


def ste(x, y):          # use y going forward, x's gradient going back
    return x + (y - x).detach()


def act(x, quant):      # what a filter primitive does to its result
    y = x.clamp(0, 1)
    return ste(y, torch.round(y * 255) / 255) if quant else y


def grid(w):            # snap each kernel (last 2 dims) to 256 levels between its min and max
    flat = w.flatten(-2)
    lo, hi = flat.min(-1, keepdim=True)[0], flat.max(-1, keepdim=True)[0]
    q = lo + torch.round((flat - lo) / (hi - lo) * 255) * (hi - lo) / 255
    return ste(w, q.view_as(w))


class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.conv = nn.Conv2d(1, K, 5, padding=2)
        s.w2 = nn.Parameter(torch.randn(K, M, 14, 14) * 0.05)
        s.b2 = nn.Parameter(torch.full((K, M), 0.1))
        s.w3 = nn.Parameter(torch.randn(10, 1, K * M) * 0.1)
        s.b3 = nn.Parameter(torch.full((10,), 0.1))

    def weights(s):
        return s.conv.weight, s.conv.bias, grid(s.w2), s.b2, grid(s.w3), s.b3

    def forward(s, x, quant=True):
        w1, b1, w2, b2, w3, b3 = s.weights()
        a = act(F.conv2d(x, w1, b1, padding=2), quant)
        p = F.max_pool2d(a, 3, 1, 1)
        h = act(torch.einsum('bkyx,kmyx->bkm', p, w2) + b2, quant)
        o = h.flatten(1) @ w3[:, 0].T + b3
        return ste(o, act(o, quant))             # clamped going forward, gradient passes through the clamp


def accuracy(net, x, y, quant):
    with torch.no_grad():
        return sum((net(x[i:i + 2000], quant).argmax(1) == y[i:i + 2000]).sum().item() for i in range(0, len(x), 2000)) / len(x)


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else 'mnist'
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    here = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'filter')
    torch.manual_seed(0); np.random.seed(0)
    xtr, ytr, xte, yte = load_mnist(d)
    xtr = torch.tensor(xtr / 255., dtype=torch.float32)[:, None]; ytr = torch.tensor(ytr.astype(np.int64))
    xte = small(torch.tensor(xte / 255., dtype=torch.float32)[:, None]); yte = torch.tensor(yte.astype(np.int64))

    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    steps = epochs * (len(xtr) // 128); sched = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=steps)
    for ep in range(epochs):
        perm = torch.randperm(len(xtr)); tot = 0
        for i in range(0, len(perm) - 127, 128):
            idx = perm[i:i + 128]
            loss = F.cross_entropy(TEMP * net(small(augment(xtr[idx]))), ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item()
        print(f'epoch {ep + 1}: loss {tot / (len(perm) // 128):.4f}  test, rounded like the filter {accuracy(net, xte, yte, True):.4f}  unrounded {accuracy(net, xte, yte, False):.4f}', flush=True)

    with torch.no_grad():
        w1, b1, w2, b2, w3, b3 = [t.double().numpy() for t in net.weights()]
    plain, rounded = accuracy(net, xte, yte, False), accuracy(net, xte, yte, True)
    r = lambda a: np.round(a, 7).tolist()
    model = {'K': K, 'M': M, 'size': 14, 'temp': TEMP,
             'conv': {'w': r(w1[:, 0]), 'b': r(b1)},            # K x 5 x 5
             'hidden': {'w': r(w2), 'b': r(b2)},                # K x M x 14 x 14, already on the 8-bit grid
             'out': {'w': r(w3[:, 0]), 'b': r(b3)},             # 10 x K*M, already on the 8-bit grid
             'test': {'n': len(xte), 'plain': plain, 'rounded': rounded}}
    json.dump(model, open(os.path.join(here, 'model.json'), 'w'), separators=(',', ':'))
    # The first 1000 held-out test images for window.selfTest(), one byte per pixel.
    n = 1000
    px = (xte[:n, 0] * 255).round().numpy().astype(np.uint8)
    json.dump({'n': n, 'size': 14, 'labels': ''.join(map(str, yte[:n].tolist())), 'pixels': base64.b64encode(px.tobytes()).decode()},
              open(os.path.join(here, 'test.json'), 'w'), separators=(',', ':'))
    if os.environ.get('FULL_TEST'):   # all 10,000, for a one-off full run
        px = (xte[:, 0] * 255).round().numpy().astype(np.uint8)
        json.dump({'n': len(xte), 'size': 14, 'labels': ''.join(map(str, yte.tolist())), 'pixels': base64.b64encode(px.tobytes()).decode()},
                  open(os.environ['FULL_TEST'], 'w'), separators=(',', ':'))
    print(f'test accuracy on {len(xte)} images: plain {plain:.4f}, with 8-bit rounding {rounded:.4f}')


if __name__ == '__main__':
    main()
