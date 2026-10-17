"""E8: does the disguise signal come from the probed model's own activations, or from any good
representation of code? (exploratory, added after the registered results)

Baselines on the same code, same train rows (anchor train, read-only, 283 problems), same cells
(disguise: original / unparse / surface / full; attack ladder level 3), within-problem AUROC with
problem-bootstrap CIs:
  ast        logistic regression on AST-structure features (node-type counts, depth, constant counts,
             branch and comparison counts, literal lengths): no model at all
  encoder    mean-pooled embeddings of a code encoder (microsoft/unixcoder-base, 125M) + LR
  decoder    code-only mean over code tokens at a chosen layer of another decoder model (e.g. Gemma-4-12B-it,
             which never wrote this code) + LR, layer by grouped CV on train
If these match the Qwen probe (0.998 under full disguise), the result is a property of code
representations in general; if they fall where the Qwen probe holds, the signal is specific to the
model that wrote the code.

  python -m diag.other_model_probe ast
  python -m diag.other_model_probe encoder --model microsoft/unixcoder-base
  python -m diag.other_model_probe decoder --model google/gemma-4-12B-it --rev <sha>
"""

import ast
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from detect import metrics as M
from detect.fit import rows_with_text
from detect.probe import fit_and_score, select

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/qwen3-8b/diag/other_model"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
NODE_TYPES = ["If", "For", "While", "Compare", "Constant", "Dict", "Subscript", "Call", "Return", "BoolOp", "Assign", "FunctionDef", "Try", "Lambda", "ListComp", "Name"]


def cells():
    dg = {r["id"]: r for r in pq.read_table(ROOT / "results/hacktrace/crux/disguised.parquet").to_pylist()}
    te = {r["id"]: r["final_solution"] for r in pq.read_table(ROOT / "data/raw/hacktrace/trajectories/test.parquet", columns=["id", "final_solution"]).to_pylist() if r["id"] in dg}
    ids = sorted(i for i, r in dg.items() if r["split"] == "test" and r.get("executed") and r["keep_full"])
    out = {"original": [(i, te[i] or "") for i in ids], "unparse": [(i, dg[i]["code_unparse"]) for i in ids],
           "surface": [(i, dg[i]["code_surface"]) for i in ids], "full": [(i, dg[i]["code_full"]) for i in ids]}
    lad = ROOT / "results/qwen3-8b/crux/hard_ladder"
    if (lad / "exec.jsonl").exists():
        rw = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "results/hacktrace/crux/hard_disguise_rewrites.jsonl")}
        kept = {json.loads(l)["item"] for l in open(lad / "exec.jsonl") if json.loads(l)["keep"]}
        # level 3 of the ladder needs the probe scores; use the blind level (seeded random kept rewrite) here
        import random
        rng = random.Random(20261016)
        lvl = []
        for i in ids:
            cands = [f"{i}|cand{k}" for k in range(4) if f"{i}|cand{k}" in kept]
            if cands:
                lvl.append((i, rw[i]["candidates"][int(rng.choice(cands).split("cand")[1])]))
        out["rewrite_blind"] = lvl
    y = {i: dg[i]["ACT"] for i in ids}
    g = {i: dg[i]["problem_name"] for i in ids}
    return out, y, g


def train_rows():
    tr = [r for r in rows_with_text("anchor_train.parquet", "train") if r["variant"] == "read-only-test"]
    return [(r["final_solution"] or "") for r in tr], np.array([int(r["ACT"]) for r in tr]), np.array([r["problem_name"] for r in tr]), np.array([r["fold"] for r in tr])


def ast_features(code):
    try:
        t = ast.parse(code)
    except Exception:
        return np.zeros(len(NODE_TYPES) + 6, np.float32)
    c = Counter(type(n).__name__ for n in ast.walk(t))
    consts = [n.value for n in ast.walk(t) if isinstance(n, ast.Constant) and isinstance(n.value, (str, int)) and not isinstance(n.value, bool)]

    def depth(n, d=0):
        return max([depth(ch, d + 1) for ch in ast.iter_child_nodes(n)] + [d])
    f = [c.get(k, 0) for k in NODE_TYPES] + [depth(t), len(consts), sum(len(str(v)) for v in consts), max([len(str(v)) for v in consts] + [0]),
                                             sum(1 for v in consts if isinstance(v, str) and ("\n" in v or " " in v)), len(code)]
    return np.log1p(np.array(f, np.float32))


def summarise(name, feats_fn, embed_train, embed_cell, report):
    codes, ytr, gtr, folds = train_rows()
    Xtr = embed_train(codes)
    C, y, g = cells()
    for cell, items in C.items():
        ids = [i for i, _ in items]
        Xte = embed_cell([c for _, c in items])
        yy = np.array([y[i] for i in ids])
        gg = np.array([g[i] for i in ids])
        if Xtr.ndim == 3:
            sel = select(Xtr, ytr, gtr, folds, layers=list(range(2, Xtr.shape[1], 2)), device=DEV)
            sc = fit_and_score(Xtr, ytr, Xte, sel["layer"], sel["C"], device=DEV)["probe"]
            extra = {"layer": sel["layer"], "C": sel["C"]}
        else:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
            s = StandardScaler().fit(Xtr)
            lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000).fit(s.transform(Xtr), ytr)
            sc = lr.decision_function(s.transform(Xte))
            extra = {}
        lo, hi = M.bootstrap_ci(yy, sc, gg)
        report[cell] = {"within": M.within_problem_auroc(yy, sc, gg), "ci": [lo, hi], "n": int(len(yy)), **extra}
        print(name, cell, round(report[cell]["within"], 3), [round(lo, 3), round(hi, 3)], flush=True)


def main():
    kind = sys.argv[1]
    model = sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else None
    rev = sys.argv[sys.argv.index("--rev") + 1] if "--rev" in sys.argv else None
    OUT.mkdir(parents=True, exist_ok=True)
    report = {"kind": kind, "model": model}
    if kind == "ast":
        f = lambda codes: np.stack([ast_features(c) for c in codes])
        summarise("ast", None, f, f, report)
    elif kind == "encoder":
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model)
        enc = AutoModel.from_pretrained(model).to(DEV).eval()

        def embed(codes):
            out = []
            for i in range(0, len(codes), 32):
                b = tok(codes[i: i + 32], return_tensors="pt", padding=True, truncation=True, max_length=1024).to(DEV)
                with torch.no_grad():
                    h = enc(**b).last_hidden_state
                m = b["attention_mask"].unsqueeze(-1).float()
                out.append(((h * m).sum(1) / m.sum(1)).cpu().numpy())
            return np.concatenate(out)
        summarise("encoder", None, embed, embed, report)
    else:
        from env.hacktrace_gen import load_any_model, render
        from env.second_model import Rendered, decoder, pooled_all_layers
        tok, mdl = load_any_model(model, rev)
        dec = decoder(mdl)
        sys_user = [{"role": "system", "content": "You are a helpful coding assistant."}, {"role": "user", "content": "Write the solution."}]

        def embed(codes):
            out = None
            for k, c in enumerate(codes):
                rr = Rendered(tok, sys_user, f"```python\n{c.rstrip()}\n```")
                pooled = pooled_all_layers(dec, rr.ids, rr.code_mask)
                if out is None:
                    out = np.zeros((len(codes), pooled.shape[0], pooled.shape[1]), np.float16)
                out[k] = pooled.half().cpu().numpy()
                del pooled
                torch.cuda.empty_cache()
            return out
        summarise("decoder", None, embed, embed, report)
    (OUT / f"{kind}{'_' + model.split('/')[-1] if model else ''}.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
