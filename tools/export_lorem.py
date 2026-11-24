"""Turn the trained model.json into lorem/model.json: rounded weights, the rotation table and the dial angles."""
import json, math, sys
J = json.load(open("model.json"))
r = lambda v: float(f"{v:.6g}") if abs(v) > 1e-9 else 0
R = lambda a: [R(x) for x in a] if isinstance(a, list) else r(a)
w = J["w"]; P = J["PERIOD"]
trig = [[[r(math.cos(p*t*2*math.pi/P)), r(math.sin(p*t*2*math.pi/P))] for t in J["turns"]] for p in range(P)]
out = {"vocab": J["vocab"], "merges": J["merges"], "D": J["D"], "H": J["H"], "HD": J["HD"], "FF": J["FF"], "W": J["W"], "PERIOD": P, "turns": J["turns"],
       "trig": trig, "ang": [[f"{(p*t*360/P)%360:.4g}deg" for t in J["turns"]] for p in range(P)], "emb": R(w["emb.weight"]), "q": R(w["q.weight"]), "k": R(w["k.weight"]), "v": R(w["v.weight"]), "o": R(w["o.weight"]),
       "f1": R(w["f1.weight"]), "f1b": R(w["f1.bias"]), "f2": R(w["f2.weight"]), "f2b": R(w["f2.bias"]), "head": R(w["head.weight"]), "headb": R(w["head.bias"])}
json.dump(out, open(sys.argv[1], "w"), separators=(",", ":"))
