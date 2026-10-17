"""D5 (exploratory): code-only probe. Is the disguise result about the code, or about the transcript around it?

Each program is rendered as the first user turn plus a single assistant turn holding only
its final code block (no reasoning prose, no earlier turns, no feedback). Pooled meancode
activations, all 37 layers. The probe is fit on read-only anchor-train programs rendered
the same way (layer and C by grouped CV, S3.3 recipe and weights) and scored on the
original and full-set disguised test programs. Compared with TF-IDF and the judge on the
same files (grid.json).

  python -m diag.code_only extract   # GPU
  python -m diag.code_only score
"""

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from common.render import render_turn
from detect import metrics as M
from detect.probe import fit_and_score, select

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/hacktrace"
OUT = ROOT / "acts/qwen3-8b/hacktrace/code_only"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def jobs():
    tr = [r for r in pq.read_table(ROOT / "results/hacktrace/splits/anchor_train.parquet").to_pylist() if r["variant"] == "read-only-test"]
    dg = {r["id"]: r for r in pq.read_table(ROOT / "results/hacktrace/crux/disguised.parquet").to_pylist()}
    te = sorted(i for i, r in dg.items() if r["split"] == "test" and r.get("executed") and r["keep_full"])
    return tr, dg, te


def extract():
    from extract.pooled import load_model, pooled_means
    tr, dg, te = jobs()
    trraw = {r["id"]: r for r in pq.read_table(RAW / "trajectories/train.parquet", columns=["id", "messages", "final_solution"]).to_pylist()
             if r["id"] in {x["id"] for x in tr}}
    teraw = {r["id"]: r for r in pq.read_table(RAW / "trajectories/test.parquet", columns=["id", "messages", "final_solution"]).to_pylist() if r["id"] in set(te)}
    c_names = set(json.loads((ROOT / "private/confirmation_set_C.json").read_text()))
    assert not {r["problem_name"] for r in tr} & c_names
    tok, model = load_model()
    sets = {"train": [(r["id"], trraw[r["id"]]["messages"], trraw[r["id"]]["final_solution"]) for r in tr],
            "orig": [(i, teraw[i]["messages"], teraw[i]["final_solution"]) for i in te],
            "full": [(i, teraw[i]["messages"], dg[i]["code_full"]) for i in te]}
    OUT.mkdir(parents=True, exist_ok=True)
    only = sys.argv[2] if len(sys.argv) > 2 else None  # extract one set (parallel machines)
    for name, rows in sets.items():
        if (only and name != only) or (OUT / f"{name}.npy").exists():
            continue
        X = np.zeros((len(rows), 37, 4096), np.float16)
        for k, (rid, msgs, code) in enumerate(rows):
            m = msgs[:2] + [{"role": "assistant", "content": f"```python\n{(code or '').rstrip()}\n```"}]
            rr = render_turn(tok, m, 1)
            cm = torch.tensor(rr.code_mask, device="cuda")
            if not cm.any():
                cm = torch.zeros_like(cm)
                cm[rr.assistant_start:] = True
            X[k] = pooled_means(model, rr.ids, [cm])[0].half().cpu().numpy()
            if k % 500 == 0:
                print(name, k, len(rows), flush=True)
        np.save(OUT / f"{name}.npy", X)
        (OUT / f"{name}.ids.json").write_text(json.dumps([r[0] for r in rows]))


def score():
    tr, dg, te = jobs()
    ded = {r["id"]: r["cluster_size"] for r in pq.read_table(ROOT / "results/hacktrace/labels/dedup_train.parquet", columns=["id", "cluster_size"]).to_pylist()}
    Xtr = np.load(OUT / "train.npy")
    y = np.array([int(r["ACT"]) for r in tr])
    g = np.array([r["problem_name"] for r in tr])
    folds = np.array([r["fold"] for r in tr])
    w = np.array([1.0 / ded.get(r["id"], 1) for r in tr])
    sel = select(Xtr, y, g, folds, weights=w, device=DEV)
    yt = np.array([dg[i]["ACT"] for i in te])
    gt = np.array([dg[i]["problem_name"] for i in te])
    out = {"layer": sel["layer"], "C": sel["C"], "cv_within_train": sel["cv_auroc"]}
    for cell in ("orig", "full"):
        Xte = np.load(OUT / f"{cell}.npy")
        sc = fit_and_score(Xtr, y, Xte, sel["layer"], sel["C"], weights=w, device=DEV)["probe"]
        sc19 = fit_and_score(Xtr, y, Xte, 19, sel["C"], weights=w, device=DEV)["probe"]
        out[cell] = {"probe_code_only": M.within_problem_auroc(yt, sc, gt), "ci": M.bootstrap_ci(yt, sc, gt),
                     "probe_code_only_layer19": M.within_problem_auroc(yt, sc19, gt)}
        print("D5", cell, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[cell].items()}, flush=True)
    p = ROOT / "results/qwen3-8b/diag/diagnostics.json"
    rep = json.loads(p.read_text())
    rep["D5_code_only_probe"] = out
    p.write_text(json.dumps(rep, indent=2, default=float) + "\n")


if __name__ == "__main__":
    {"extract": extract, "score": score}[sys.argv[1]]()
