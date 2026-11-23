"""Turn raincandy-u/TinyChat-1776K into weight pictures for the canvas engine.

  python3 export_chat.py path/to/TinyChat-1776K chat/model.json

The folder holds model.safetensors, config.json and tokenizer.json from the model's Hugging Face repo.
"""
import sys, json, base64, io, math, os
import numpy as np, torch
from safetensors.torch import load_file
from tokenizers import Tokenizer
from PIL import Image

MD, OUT = sys.argv[1], sys.argv[2]
cfg = json.load(open(MD + "/config.json"))
sd = {k: v.float().numpy().astype(np.float64) for k, v in load_file(MD + "/model.safetensors").items()}
D, H, KVH, L, FF, V = cfg["hidden_size"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["num_hidden_layers"], cfg["intermediate_size"], cfg["vocab_size"]
HD, EPS = D // H, cfg["rms_norm_eps"]
NAMES = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.o_proj", "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}
emb = sd["model.embed_tokens.weight"]
W = {f"{l}.{n}": sd[f"model.layers.{l}.{p}.weight"] for l in range(L) for n, p in NAMES.items()}
W["head"] = emb
n1 = [sd[f"model.layers.{l}.input_layernorm.weight"] for l in range(L)]
n2 = [sd[f"model.layers.{l}.post_attention_layernorm.weight"] for l in range(L)]
nf = sd["model.norm.weight"]

def quant(w):
    """8-bit magnitude per weight with one scale per row: what the pictures hold."""
    s = np.abs(w).max(1, keepdims=True); s[s == 0] = 1
    q = np.round(np.abs(w) / s * 255).astype(np.uint8)
    return q, np.sign(w).astype(np.int8), s[:, 0]
Q = {k: quant(w) for k, w in W.items()}
WQ = {k: q.astype(np.float64) / 255 * sg * s[:, None] for k, (q, sg, s) in Q.items()}

freqs = 1.0 / (cfg["rope_theta"] ** (np.arange(0, HD, 2) / HD))
def rms(x, w): return x / np.sqrt((x * x).mean() + EPS) * w
def rope(x, pos, nh):
    x = x.reshape(nh, HD); c, s = np.cos(pos * freqs), np.sin(pos * freqs); a, b = x[:, :HD // 2], x[:, HD // 2:]
    return np.concatenate([a * c - b * s, b * c + a * s], 1)
def step(Wt, tok, pos, cache):
    x = emb[tok].copy()
    for l in range(L):
        h = rms(x, n1[l])
        q = rope(Wt[f"{l}.q"] @ h, pos, H); k = rope(Wt[f"{l}.k"] @ h, pos, KVH); v = (Wt[f"{l}.v"] @ h).reshape(KVH, HD)
        cache[l]["k"].append(k); cache[l]["v"].append(v)
        K = np.stack(cache[l]["k"]); Vv = np.stack(cache[l]["v"]); a = np.zeros((H, HD))
        for hh in range(H):
            g = hh // (H // KVH); sc = K[:, g] @ q[hh] / math.sqrt(HD); p = np.exp(sc - sc.max()); p /= p.sum(); a[hh] = p @ Vv[:, g]
        x = x + Wt[f"{l}.o"] @ a.reshape(D); h = rms(x, n2[l]); g = Wt[f"{l}.gate"] @ h
        x = x + Wt[f"{l}.down"] @ (g / (1 + np.exp(-g)) * (Wt[f"{l}.up"] @ h))
    return Wt["head"] @ rms(x, nf)
def run(Wt, ids, n):
    cache = [{"k": [], "v": []} for _ in range(L)]
    for i, t in enumerate(ids): lg = step(Wt, t, i, cache)
    first = lg.copy(); out = []
    for i in range(n):
        t = int(lg.argmax()); out.append(t)
        if t == 2: break
        lg = step(Wt, t, len(ids) + i, cache)
    return out, first

tok = Tokenizer.from_file(MD + "/tokenizer.json")
convo = "<A>Hi, Tom. How are you? <end>\n<B>I'm fine, thank you. And you? <end>\n<A>Fine. What's your favorite color? <end>\n<B>"
ids = tok.encode(convo).ids
out_f, first_f = run(W, ids, 40); out_q, first_q = run(WQ, ids, 40)
print("float :", tok.decode(out_f)); print("8-bit :", tok.decode(out_q))
print("8-bit vs float: max logit diff %.3f, greedy tokens equal: %s" % (np.abs(first_f - first_q).max(), out_f == out_q))
if "--check" in sys.argv:
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(MD, torch_dtype=torch.float32).eval()
    with torch.no_grad(): ref = m(torch.tensor([ids])).logits[0, -1].numpy()
    print("numpy vs transformers: max logit diff %.5f (logits up to %.1f)" % (np.abs(ref - first_f).max(), np.abs(ref).max()))

def png(k):
    q, sg, s = Q[k]; n, m = q.shape; mp = 1 << (m - 1).bit_length(); im = np.zeros((n, mp, 4), np.uint8); im[..., 3] = 255
    im[:, :m, 0] = q * (sg > 0); im[:, :m, 1] = q * (sg < 0)
    b = io.BytesIO(); Image.fromarray(im, "RGBA").save(b, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
b64 = lambda a: base64.b64encode(np.ascontiguousarray(a, dtype=np.float32).tobytes()).decode()
tj = json.load(open(MD + "/tokenizer.json"))
vocab = [None] * V
for p, i in tj["model"]["vocab"].items(): vocab[i] = p
tests = ["<A>Hi, Tom. How are you? <end>\n<B>", "<A>Do you like cats?  I do! <end>\n<B>Yes. <end>\n<A>why not, it's 3 o'clock <end>\n<B>", convo]
bundle = {"name": "TinyChat-1776K", "cfg": {"D": D, "H": H, "KVH": KVH, "HD": HD, "L": L, "FF": FF, "V": V, "eps": EPS, "theta": cfg["rope_theta"], "ctx": cfg["max_position_embeddings"]},
  "mats": {k: {"png": png(k), "n": W[k].shape[0], "m": W[k].shape[1], "scale": b64(Q[k][2])} for k in W},
  "emb": b64(emb), "n1": [b64(a) for a in n1], "n2": [b64(a) for a in n2], "nf": b64(nf),
  "tok": {"vocab": vocab, "merges": [m if isinstance(m, str) else " ".join(m) for m in tj["model"]["merges"]]},
  "test": {"ids": ids, "greedy": out_f, "greedy8": out_q, "firstLogits": [float(x) for x in first_f], "firstLogits8": [float(x) for x in first_q],
           "encode": [{"text": t, "ids": tok.encode(t).ids} for t in tests]}}
os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True); json.dump(bundle, open(OUT, "w"))
print("bundle KB", os.path.getsize(OUT) // 1024, "pictures KB", sum(len(m["png"]) for m in bundle["mats"].values()) * 3 // 4 // 1024)
