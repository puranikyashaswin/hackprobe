"""Exploratory: stronger pre-shortcut readouts on the S3.2 per-token activations (after D3b (e)).

The registered readout was a linear probe on one token's state at one layer (train CV 0.563;
test AUROC 0.750 at k = 1, L(0.8) = 0). Same pairs, same train / test rows, same curve and
L(theta), with stronger readouts:
  linear_L      the registered form, re-fit here as the reference
  linear_cat6   linear probe on the 6 stored layers concatenated (24,576 dims)
  linear_ctx    linear probe on [token state, mean of the previous 16 tokens] at the registered layer
  mlp_L         one-hidden-layer MLP (256 units) at the registered layer
  mlp_cat6      the same MLP on the 6 layers concatenated
Hyper-parameters by grouped 5-fold CV on the train pairs (mean curve AUROC, as S3.6 chose its
layer); each readout is then fit on all train pairs and scored once on test stratum A.
Exploratory: the registered result stands.

  python -m diag.strong_probe   # GPU machine with the per-token activations
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from detect import metrics as M
from detect.leadtime import SPL, K, curve, lead_L, load_acts, pairs_for
from detect.probe import fit_logreg
from extract.pertoken import LAYERS

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/qwen3-8b/diag"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
SEED, MAX_TRAIN = 20261016, 40000
CTX = 16


def features(acts, items, kind, layer):
    """Rows for (key, idx) items, gathered per key so each mmap slice is read once."""
    by = defaultdict(list)
    for n, (key, idx) in enumerate(items):
        by[key].append((idx, n))
    dim = {"L": 4096, "cat6": 4096 * len(LAYERS), "ctx": 8192}[kind]
    X = np.zeros((len(items), dim), np.float16)
    for key, lst in by.items():
        a = acts[key]
        idxs = np.array([i for i, _ in lst])
        rows = np.array([n for _, n in lst])
        if kind == "L":
            X[rows] = np.asarray(a["L"][layer])[idxs]
        elif kind == "cat6":
            X[rows] = np.concatenate([np.asarray(a["L"][l])[idxs] for l in LAYERS], 1)
        else:
            L = np.asarray(a["L"][layer])
            cur = L[idxs]
            prev = np.stack([L[max(0, i - CTX): i].mean(0) if i > 0 else L[i] for i in idxs])
            X[rows] = np.concatenate([cur, prev], 1)
    return X


class MLP(torch.nn.Module):
    def __init__(self, d, h=256):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.ReLU(), torch.nn.Dropout(0.1), torch.nn.Linear(h, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def fit_mlp(X, y, wd, epochs, mu, sd, seed=0):
    torch.manual_seed(seed)
    m = MLP(X.shape[1]).to(DEV)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=wd)
    n = len(X)
    idx = np.arange(n)
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        rng.shuffle(idx)
        m.train()
        for s in range(0, n, 512):
            b = idx[s: s + 512]
            xb = (torch.tensor(X[b], device=DEV, dtype=torch.float32) - mu) / sd
            yb = torch.tensor(y[b], device=DEV, dtype=torch.float32)
            opt.zero_grad()
            torch.nn.functional.binary_cross_entropy_with_logits(m(xb), yb).backward()
            opt.step()
    m.eval()
    return m


def predict(model, X, mu, sd, linear):
    out = []
    with torch.no_grad():
        for s in range(0, len(X), 8192):
            xb = (torch.tensor(X[s: s + 8192], device=DEV, dtype=torch.float32) - mu) / sd
            out.append((xb @ model[0] + model[1] if linear else model(xb)).cpu().numpy())
    return np.concatenate(out)


def fit(kind_model, X, y, hp):
    mu = torch.tensor(X.astype(np.float32).mean(0), device=DEV)
    sd = torch.tensor(X.astype(np.float32).std(0) + 1e-6, device=DEV)
    if kind_model == "linear":
        w, b = fit_logreg((X.astype(np.float32) - mu.cpu().numpy()) / sd.cpu().numpy(), y, hp, device=DEV)
        return (torch.tensor(w, device=DEV), float(b)), mu, sd, True
    return fit_mlp(X, y, hp, 12, mu, sd), mu, sd, False


def pair_items(pairs):
    pos = sorted({(p[2], p[3]) for p in pairs})
    neg = sorted({(p[4], p[5]) for p in pairs})
    return pos, neg


def score_pairs(acts, pairs, kind, layer, model, mu, sd, linear):
    pos, neg = pair_items(pairs)
    sp, sn = {}, {}
    for items, store in ((pos, sp), (neg, sn)):
        for s in range(0, len(items), 20000):
            chunk = items[s: s + 20000]
            sc = predict(model, features(acts, chunk, kind, layer), mu, sd, linear)
            store.update(zip(chunk, sc.astype(float)))
    return sp, sn


def main():
    tr_rows = [r for r in pq.read_table(SPL / "anchor_pertoken_train.parquet").to_pylist() if r["stratum"] == "A"]
    te_rows = [r for r in pq.read_table(SPL / "anchor_pertoken.parquet").to_pylist() if r["stratum"] == "A"]
    tr_acts, te_acts = load_acts("train"), load_acts("test")
    layer = int(np.load(ROOT / "results/qwen3-8b/anchor/pertoken_probe.npz")["layer"])
    tr_pairs, _ = pairs_for(tr_rows, tr_acts, kstep=2)
    te_pairs, _ = pairs_for(te_rows, te_acts)
    rng = np.random.default_rng(SEED)
    if len(tr_pairs) > MAX_TRAIN:
        tr_pairs = [tr_pairs[i] for i in sorted(rng.choice(len(tr_pairs), MAX_TRAIN, replace=False))]
    probs = sorted({p[0] for p in tr_pairs})
    fold = {p: i % 5 for i, p in enumerate(probs)}
    configs = {"linear_L": ("linear", "L", (0.01, 0.1, 1.0)), "linear_cat6": ("linear", "cat6", (0.01, 0.1, 1.0)),
               "linear_ctx": ("linear", "ctx", (0.01, 0.1, 1.0)), "mlp_L": ("mlp", "L", (1e-3, 1e-2, 1e-1)),
               "mlp_cat6": ("mlp", "cat6", (1e-3, 1e-2, 1e-1))}
    report = {"layer": layer, "n_train_pairs": len(tr_pairs), "n_test_pairs": len(te_pairs), "readouts": {}}
    for name, (kind_model, kind, grid) in configs.items():
        print(name, flush=True)
        items_pos, items_neg = pair_items(tr_pairs)
        Xtr = np.concatenate([features(tr_acts, items_pos, kind, layer), features(tr_acts, items_neg, kind, layer)])
        ytr = np.array([1] * len(items_pos) + [0] * len(items_neg))
        key_row = {it: i for i, it in enumerate(items_pos + items_neg)}
        best = (-1, None)
        for hp in grid:
            sp, sn = {}, {}
            for f in range(5):
                tr = [p for p in tr_pairs if fold[p[0]] != f]
                va = [p for p in tr_pairs if fold[p[0]] == f]
                rows_tr = sorted({key_row[(p[2], p[3])] for p in tr} | {key_row[(p[4], p[5])] for p in tr})
                model, mu, sd, lin = fit(kind_model, Xtr[rows_tr], ytr[rows_tr], hp)
                vp, vn = pair_items(va)
                for items, store in ((vp, sp), (vn, sn)):
                    rows = [key_row[it] for it in items]
                    store.update(zip(items, predict(model, Xtr[rows], mu, sd, lin).astype(float)))
            res, _ = curve(tr_pairs, sp, sn)
            auc = float(np.mean([v["auroc"] for v in res.values()])) if res else -1
            print(f"  hp {hp}: cv {auc:.3f}", flush=True)
            if auc > best[0]:
                best = (auc, hp)
        model, mu, sd, lin = fit(kind_model, Xtr, ytr, best[1])
        del Xtr
        sp, sn = score_pairs(te_acts, te_pairs, kind, layer, model, mu, sd, lin)
        res, by_k = curve(te_pairs, sp, sn)
        ks = sorted(res)
        report["readouts"][name] = {"hp": best[1], "cv_auroc_train": best[0],
                                    "curve": {int(k): res[k] for k in ks},
                                    "L": {str(th): lead_L(by_k, ks, th) for th in (0.8, 0.9)}}
        print(f"  test: {{k: AUROC}} {{{', '.join(f'{k}: {res[k]['auroc']:.3f}' for k in (1, 2, 4, 8, 16, 32, 64) if k in res)}}}  "
              f"L(0.8) {report['readouts'][name]['L']['0.8']['L']}", flush=True)
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / "strong_probe.json").write_text(json.dumps(report, indent=2) + "\n")
    print("->", OUT / "strong_probe.json")


if __name__ == "__main__":
    main()
