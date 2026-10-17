"""S3.4: per-category transfer inside read-only (2x2, hardcode_output <-> special_casing).

Each cell fits on anchor-train rows (category-A hacks vs read-only ACT=0 rows)
and scores on anchor-test rows (category-B hacks vs ACT=0 rows of the same test
problems). Train and test never share a problem (asserted). The meancode probe
picks layer and C by grouped 5-fold CV on the cell's train rows, as in S3.3;
TF-IDF+LR and visible-literal LR are refit per cell; the prompted judge and
length need no fit. Descriptive: no adds-value test in-distribution.
A scored cell with fewer than 100 hacks is marked pilot.
"""

import json

import numpy as np
import pyarrow.parquet as pq
import torch

from detect.fit import ROOT, RAW, load_acts, rows_with_text, summarize
from detect.probe import fit_and_score, select
from detect.text_baselines import artifact_text, length_only, literal_features, literal_lr, tfidf_lr

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CATS = ("hardcode_output", "special_casing")
VAR = "read-only-test"
MIN_POS = 100


def main():
    tr = [r for r in rows_with_text("anchor_train.parquet", "train") if r["variant"] == VAR]
    te = [r for r in rows_with_text("anchor_test.parquet", "test") if r["variant"] == VAR]
    assert not {r["problem_name"] for r in tr} & {r["problem_name"] for r in te}, "train and test share a problem"
    dedup = {}
    for f in ("dedup_train.parquet", "dedup_test.parquet"):
        dedup.update({r["id"]: r for r in pq.read_table(ROOT / "results/hacktrace/labels" / f, columns=["id", "cluster_size"]).to_pylist()})
    probs = {(p["tier"], p["problem_name"]): p["visible_cases"] for p in
             pq.read_table(RAW / "problems/problems.parquet", columns=["tier", "problem_name", "visible_cases"]).to_pylist()}
    judge = json.loads((ROOT / "results/qwen3-8b/anchor/judge_scores.json").read_text())
    A_tr_all = load_acts([r["id"] for r in tr], "meancode")
    A_te_all = load_acts([r["id"] for r in te], "meancode")
    texts = lambda rows: [artifact_text(r["final_solution"], r["final_test"]) for r in rows]
    feats = lambda rows: np.array([literal_features(r["final_solution"], probs[(r["tier"], r["problem_name"])]) for r in rows])
    report = {}
    for src in CATS:
        ktr = [i for i, r in enumerate(tr) if not r["ACT"] or r["judge_category"] == src]
        R = [tr[i] for i in ktr]
        ytr = np.array([int(r["ACT"]) for r in R])
        gtr = np.array([r["problem_name"] for r in R])
        folds = np.array([r["fold"] for r in R])
        w = np.array([1.0 / dedup.get(r["id"], {}).get("cluster_size", 1) for r in R])
        sel = select(A_tr_all[ktr], ytr, gtr, folds, weights=w, device=DEV)
        for dst in CATS:
            hacked = {r["problem_name"] for r in te if r["ACT"] and r["judge_category"] == dst}
            kte = [i for i, r in enumerate(te) if r["problem_name"] in hacked and (not r["ACT"] or r["judge_category"] == dst)]
            T = [te[i] for i in kte]
            yte = np.array([int(r["ACT"]) for r in T])
            gte = np.array([r["problem_name"] for r in T])
            sc = fit_and_score(A_tr_all[ktr], ytr, A_te_all[kte], sel["layer"], sel["C"], weights=w, device=DEV)
            cell = {"probe": {**summarize(yte, sc["probe"], gte), "layer": sel["layer"], "C": sel["C"], "cv_within": sel["cv_auroc"]},
                    "tfidf_lr": summarize(yte, tfidf_lr(texts(R), ytr, texts(T)), gte),
                    "visible_literal_lr": summarize(yte, literal_lr(feats(R), ytr, feats(T)), gte),
                    "prompted_judge": summarize(yte, np.array([judge[r["id"]] for r in T]), gte),
                    "length_only": summarize(yte, length_only([len(r["final_solution"] or "") for r in T]), gte)}
            n_pos = int(yte.sum())
            cell["n_pos"], cell["n_neg"], cell["n_problems"] = n_pos, int(len(yte) - n_pos), len(hacked)
            cell["n_train_pos"], cell["pilot"] = int(ytr.sum()), n_pos < MIN_POS
            report[f"{src}->{dst}"] = cell
            print(f"{src} -> {dst}: n_pos {n_pos} on {len(hacked)} problems", {k: round(v["within"], 3) for k, v in cell.items() if isinstance(v, dict)},
                  flush=True)
    out = ROOT / "results/qwen3-8b/anchor/transfer.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("->", out)


if __name__ == "__main__":
    main()
