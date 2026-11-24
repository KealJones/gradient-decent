"""Train a tiny attention model on Cicero's De Finibus (the source of lorem ipsum), built so that page layout can run it:
no norms, ReLU attention with a sink of 1 (weights = relu(s) / (1 + sum relu(s))), rotary positions with angles that
repeat every 64 positions (so a 16-slot ring buffer works), sliding window of 16 tokens.

Needs the five books of De Finibus as corpus/fin1.txt .. fin5.txt, from the Latin Library via
https://github.com/cltk/lat_text_latin_library (cicero/fin1.txt and so on). Then:
    python3 tools/train_lorem.py model.json 6000 && python3 tools/export_lorem.py lorem/model.json"""

import re, json, math, time, sys, random
import torch, torch.nn as nn, torch.nn.functional as F
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

torch.manual_seed(0)
random.seed(0)
torch.set_num_threads(2)
OUT = sys.argv[1] if len(sys.argv) > 1 else "model.json"
V, D, H, HD, FF, W, PERIOD = 320, 32, 2, 16, 96, 16, 64
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 6000

# ---- text: lowercase Latin, letters, space, comma, period
raw = ""
for b in range(1, 6):
    t = open(f"corpus/fin{b}.txt", encoding="utf-8").read()
    t = t[t.index("[ 1 ]") :] if "[ 1 ]" in t else t
    raw += " " + t
raw = raw.lower().replace("j", "i")
raw = re.sub(r"\[[^\]]*\]", " ", raw)
raw = re.sub(r"[;:!?]", ".", raw)
raw = re.sub(r"[^a-z,. ]", " ", raw)
raw = re.sub(r"\s*([,.])", r"\1", raw)
raw = re.sub(r"([,.])[,.]+", r"\1", raw)
raw = re.sub(r" +", " ", raw).strip()
print("chars", len(raw), "words", len(raw.split()))

# ---- a 320-piece BPE vocabulary over that text, words split at spaces ('▁' marks a word start)
tok = Tokenizer(models.BPE())
tok.pre_tokenizer = pre_tokenizers.Metaspace()
tok.decoder = decoders.Metaspace()
tok.train_from_iterator(
    [raw[i : i + 2000] for i in range(0, len(raw), 2000)],
    trainers.BpeTrainer(vocab_size=V, initial_alphabet=list("abcdefghiklmnopqrstuvxyz,."), show_progress=False),
)
V = tok.get_vocab_size()
ids = torch.tensor(tok.encode(raw).ids)
print("vocab", V, "tokens", len(ids))
split = int(len(ids) * 0.95)
tr, va = ids[:split], ids[split:]


# ---- rotary positions. The turn counts are whole numbers of turns per 64 positions, so the angles repeat every 64
# and the page's 16-slot ring buffer can use (position mod 64) without ever shifting anything.
def rope_tables(n):
    k = torch.tensor([1, 2, 3, 5, 8, 12, 17, 23], dtype=torch.float)  # integer turns per 64 positions
    ang = torch.arange(n)[:, None].float() * k[None] * 2 * math.pi / PERIOD
    return ang.cos(), ang.sin()


def rope(x, cos, sin):  # x: B,T,H,HD ; pairs (2i, 2i+1)
    a, b = x[..., 0::2], x[..., 1::2]
    c, s = cos[None, :, None], sin[None, :, None]
    return torch.stack([a * c - b * s, a * s + b * c], -1).flatten(-2)


# ---- the model: embedding, one attention layer (2 heads), one feed-forward layer, output head. No layer norms, and
# attention shares are relu(score) / (1 + sum of relu(scores)) over a 16-token window: both things page layout can do.
class M(nn.Module):
    def __init__(s):
        super().__init__()
        s.emb = nn.Embedding(V, D)
        nn.init.normal_(s.emb.weight, std=0.3)
        s.q, s.k, s.v, s.o = (nn.Linear(D, D, bias=False) for _ in range(4))
        s.f1, s.f2 = nn.Linear(D, FF), nn.Linear(FF, D)
        s.head = nn.Linear(D, V)

    def forward(s, x, off=0):
        B, T = x.shape
        h = s.emb(x)
        cos, sin = rope_tables(off + T)
        cos, sin = cos[off:], sin[off:]
        q = rope(s.q(h).view(B, T, H, HD), cos, sin)
        k = rope(s.k(h).view(B, T, H, HD), cos, sin)
        v = s.v(h).view(B, T, H, HD)
        sc = torch.einsum("bthd,bshd->bhts", q, k)
        i = torch.arange(T)
        mask = (i[None, :] <= i[:, None]) & (i[:, None] - i[None, :] < W)
        r = F.relu(sc) * mask
        w = r / (1 + r.sum(-1, keepdim=True))
        a = torch.einsum("bhts,bshd->bthd", w, v).reshape(B, T, D)
        h = h + s.o(a)
        h = h + s.f2(F.relu(s.f1(h)))
        return s.head(h)


m = M()
print("params", sum(p.numel() for p in m.parameters()))
# ---- training: AdamW with warmup and cosine decay, random 64-token windows, a random rotary offset each batch
opt = torch.optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.05)
T = 64


def batch(src, n=64):
    ix = torch.randint(0, len(src) - T - 1, (n,))
    return (
        torch.stack([src[i : i + T] for i in ix]),
        torch.stack([src[i + 1 : i + T + 1] for i in ix]),
        int(torch.randint(0, PERIOD, ())),
    )


@torch.no_grad()
def val():
    m.eval()
    L = []
    for _ in range(20):
        x, y, off = batch(va)
        L.append(F.cross_entropy(m(x, off).reshape(-1, V), y.reshape(-1)).item())
    m.train()
    return sum(L) / len(L)


t0 = time.time()
for step in range(STEPS + 1):
    lr = 3e-3 * min(1, step / 200) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / STEPS)))
    for g in opt.param_groups:
        g["lr"] = lr
    x, y, off = batch(tr)
    loss = F.cross_entropy(m(x, off).reshape(-1, V), y.reshape(-1))
    opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(m.parameters(), 1.0)
    opt.step()
    if step % 500 == 0:
        print(f"step {step} train {loss.item():.3f} val {val():.3f} {time.time() - t0:.0f}s", flush=True)

# unigram baseline for comparison
cnt = torch.bincount(tr, minlength=V).float() + 1
p = cnt / cnt.sum()
print("unigram val loss", float(-(p.log()[va]).mean()))


# ---- a quick look at what it writes, then save the weights
@torch.no_grad()
def sample(prompt, n=60, temp=0.7):
    x = tok.encode(prompt).ids
    for i in range(n):
        ctx = x[-W:]
        off = (len(x) - len(ctx)) % PERIOD
        lg = m(torch.tensor([ctx]), off)[0, -1] / max(temp, 1e-6)
        x.append(int(torch.multinomial(F.softmax(lg, -1), 1)) if temp > 0 else int(lg.argmax()))
    return tok.decode(x)


for t in (0, 0.7):
    print(f"T={t}:", sample("lorem ipsum dolor sit amet,", temp=t))

sd = {k: v.tolist() for k, v in m.state_dict().items()}
json.dump(
    {
        "vocab": [tok.id_to_token(i) for i in range(V)],
        "merges": json.loads(tok.to_str())["model"]["merges"],
        "D": D,
        "H": H,
        "HD": HD,
        "FF": FF,
        "W": W,
        "PERIOD": PERIOD,
        "turns": [1, 2, 3, 5, 8, 12, 17, 23],
        "w": sd,
    },
    open(OUT, "w"),
)
print("saved", OUT)
