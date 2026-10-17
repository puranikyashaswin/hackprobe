"""S3.3: in-distribution reproduction (runs on a GPU machine).

Per variant (read-only, editable): the meancode probe (HackTrace's site) with
layer and C chosen by grouped 5-fold CV on anchor train, refit and scored once
on anchor test; the meangen probe and difference-of-means beside it. Baselines:
TF-IDF+LR, visible-literal LR, length only, and the solvability probe (fit on
ACT=0 rows to predict honest_correct; hack score = -logit). The prompted judge
is scored separately (detect/judge.py) and merged here if present.

Reports within-problem AUROC with a problem-bootstrap CI, pooled AUROC, and
TPR at 1% FPR; editable also on non-artefact rows. Rows are weighted by
1/duplicate-cluster size in training. --shuffle permutes training labels for a
pipeline dry run (writes to scratch, never to results/).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from detect import metrics as M
from detect.probe import fit_and_score, fit_logreg, select, standardize
from detect.text_baselines import artifact_text, length_only, literal_features, literal_lr, tfidf_lr

ROOT = Path(__file__).resolve().parents[1]
SPL = ROOT / "results/hacktrace/splits"
ACTS = ROOT / "acts/qwen3-8b/hacktrace/pooled"
RAW = ROOT / "data/raw/hacktrace"
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def load_acts(ids, pos):
    idx = {json.loads(l)["id"]: json.loads(l) for l in open(ACTS / "index.jsonl")}
    shards = {}
    out = np.zeros((len(ids), 37, 4096), np.float16)
    for k, rid in enumerate(ids):
        r = idx[rid]
        if r["shard"] not in shards:
            shards[r["shard"]] = np.stack([np.load(ACTS / f"shard_{r['shard']:04d}/L{l:02d}_{pos}.npy") for l in range(37)], 1)
        out[k] = shards[r["shard"]][r["offset"]]
    return out


def rows_with_text(split_file, raw_split):
    rows = pq.read_table(SPL / split_file).to_pylist()
    want = {r["id"] for r in rows}
    raw = {r["id"]: r for r in pq.read_table(RAW / f"trajectories/{raw_split}.parquet",
                                              columns=["id", "final_solution", "final_test", "tier", "problem_name"]).to_pylist()
           if r["id"] in want}
    for r in rows:
        r.update(raw[r["id"]])
    return rows


def summarize(y, s, g, artefact=None):
    lo, hi = M.bootstrap_ci(y, s, g)
    out = {"within": M.within_problem_auroc(y, s, g), "ci": [lo, hi], "pooled": M.auroc(y, s),
           "tpr_at_1pct_fpr": M.tpr_at_fpr(y, s), "n_rows": int(len(y)), "n_problems": int(len(set(g)))}
    if artefact is not None and artefact.any():
        keep = ~artefact
        out["non_artefact_within"] = M.within_problem_auroc(y[keep], s[keep], g[keep])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffle", action="store_true")
    ap.add_argument("--layers", default="all")
    a = ap.parse_args()
    out_path = (Path("/data/scratch") if a.shuffle else ROOT / "results/qwen3-8b/anchor") / "indist.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tr, te = rows_with_text("anchor_train.parquet", "train"), rows_with_text("anchor_test.parquet", "test")
    dedup = {}
    for f in ("dedup_train.parquet", "dedup_test.parquet"):
        p = ROOT / "results/hacktrace/labels" / f
        if p.exists():
            dedup.update({r["id"]: r for r in pq.read_table(p, columns=["id", "cluster_size", "artefact"]).to_pylist()})
    probs = {(p["tier"], p["problem_name"]): p["visible_cases"] for p in
             pq.read_table(RAW / "problems/problems.parquet", columns=["tier", "problem_name", "visible_cases"]).to_pylist()}
    judge_file = ROOT / "results/qwen3-8b/anchor/judge_scores.json"
    judge = json.loads(judge_file.read_text()) if judge_file.exists() and not a.shuffle else {}
    layers = None if a.layers == "all" else [int(x) for x in a.layers.split(",")]
    rng = np.random.default_rng(20261015)

    report = {"shuffled_labels": a.shuffle}
    for variant in ("read-only-test", "editable-test"):
        R = [r for r in tr if r["variant"] == variant]
        T = [r for r in te if r["variant"] == variant]
        ytr = np.array([int(r["ACT"]) for r in R])
        if a.shuffle:
            ytr = rng.permutation(ytr)
        yte = np.array([int(r["ACT"]) for r in T])
        if a.shuffle:  # a dry run must never show a real number, including untrained baselines
            yte = rng.permutation(yte)
        gtr = np.array([r["problem_name"] for r in R])
        gte = np.array([r["problem_name"] for r in T])
        folds = np.array([r["fold"] for r in R])
        w = np.array([1.0 / dedup.get(r["id"], {}).get("cluster_size", 1) for r in R])
        art = np.array([bool(dedup.get(r["id"], {}).get("artefact", False)) for r in T])
        res = {}
        for pos in ("meancode", "meangen"):
            A_tr, A_te = load_acts([r["id"] for r in R], pos), load_acts([r["id"] for r in T], pos)
            sel = select(A_tr, ytr, gtr, folds, weights=w, layers=layers, device=DEV)
            sc = fit_and_score(A_tr, ytr, A_te, sel["layer"], sel["C"], weights=w, device=DEV)
            res[f"probe_{pos}"] = {**summarize(yte, sc["probe"], gte, art), "layer": sel["layer"], "C": sel["C"],
                                    "cv_within": sel["cv_auroc"]}
            res[f"diff_of_means_{pos}"] = summarize(yte, sc["diff_of_means"], gte, art)
            if pos == "meancode":  # solvability probe at the same site and layer
                hc = np.array([r["judge_category"] == "honest_correct" for r in R])
                neg = ytr == 0
                x0, xt = standardize(A_tr[neg][:, sel["layer"]].astype(np.float32), A_te[:, sel["layer"]].astype(np.float32))
                wv, b = fit_logreg(x0, hc[neg].astype(int), sel["C"], device=DEV)
                res["solvability_probe"] = summarize(yte, -(xt @ wv + b), gte, art)
        texts = lambda rows: [artifact_text(r["final_solution"], r["final_test"]) for r in rows]
        res["tfidf_lr"] = summarize(yte, tfidf_lr(texts(R), ytr, texts(T)), gte, art)
        feats = lambda rows: np.array([literal_features(r["final_solution"], probs[(r["tier"], r["problem_name"])]) for r in rows])
        res["visible_literal_lr"] = summarize(yte, literal_lr(feats(R), ytr, feats(T)), gte, art)
        res["length_only"] = summarize(yte, length_only([len(r["final_solution"] or "") for r in T]), gte, art)
        if judge:
            res["prompted_judge"] = summarize(yte, np.array([judge[r["id"]] for r in T]), gte, art)
        report[variant] = res
        print(variant, {k: round(v["within"], 3) for k, v in res.items()}, flush=True)

    out_path.write_text(json.dumps(report, indent=2) + "\n")
    print("->", out_path)


if __name__ == "__main__":
    main()
