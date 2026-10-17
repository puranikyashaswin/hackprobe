"""S3.5: H1a, problem-level propensity from the turn-1 prompt (runs on a GPU machine).

Every row of a problem shares one turn-1 prompt, so its `lastprompt` activation
is identical within the problem and the within-problem AUROC must be 0.50
(asserted; anything else is a pairing bug). The probe is fit on train-minus-C
rows (each row carries its problem's activation), layer and C chosen by grouped
CV, and scored once on all test rows: pooled AUROC per variant, against a
prompt-feature baseline (TF-IDF on statement + test.py, plus tier).
Never described as a decision. --shuffle writes a dry run to /data/scratch.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from detect import metrics as M
from detect.fit import DEV, ROOT, RAW
from detect.probe import fit_and_score

LP = ROOT / "acts/qwen3-8b/hacktrace/lastprompt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shuffle", action="store_true")
    a = ap.parse_args()
    groups = [json.loads(l) for l in open(LP / "groups.jsonl")]
    vec = np.load(LP / "lastprompt.npy")
    gidx = {(g["split"], g["variant"], g["tier"], g["problem_name"]): i for i, g in enumerate(groups)}
    c_names = set(json.loads((ROOT / "private/confirmation_set_C.json").read_text()))
    rng = np.random.default_rng(20261015)
    report = {"shuffled_labels": a.shuffle}
    for variant in ("read-only-test", "editable-test"):
        sets = {}
        for split in ("train", "test"):
            filt = [("problem_name", "not in", sorted(c_names))] if split == "train" else None
            t = pq.read_table(RAW / f"trajectories/{split}.parquet",
                              columns=["variant", "tier", "problem_name", "ACT", "judge_category", "messages"], filters=filt).to_pylist()
            t = [r for r in t if r["variant"] == variant and "conf" not in r["tier"]]
            sets[split] = t
        tr, te = sets["train"], sets["test"]
        assert not {r["problem_name"] for r in tr} & c_names
        # one row per problem group: label = cheat rate, weight = rows (same model as fitting every row)
        from collections import defaultdict
        agg, hc = defaultdict(list), defaultdict(list)
        for r in tr:
            agg[(r["tier"], r["problem_name"])].append(int(r["ACT"]))
            hc[(r["tier"], r["problem_name"])].append(int(r["judge_category"] == "honest_correct"))
        keys = sorted(agg)
        Xtr = vec[[gidx[("train", variant, t, p)] for t, p in keys]]
        ytr = np.array([np.mean(agg[k]) for k in keys])
        wtr = np.array([len(agg[k]) for k in keys], float)
        if a.shuffle:
            ytr = rng.permutation(ytr)
        Xte = vec[[gidx[("test", variant, r["tier"], r["problem_name"])] for r in te]]
        yte = np.array([int(r["ACT"]) for r in te])
        gte = np.array([r["problem_name"] for r in te])
        folds = np.arange(len(keys)) % 5
        best = (-1, None, None)
        for layer in range(37):
            for c in (0.01, 0.1, 1.0):
                s = np.zeros(len(keys))
                for f in range(5):
                    trm, vam = folds != f, folds == f
                    s[vam] = fit_and_score(Xtr[trm], ytr[trm], Xtr[vam], layer, c, weights=wtr[trm], device=DEV)["probe"]
                auc = M.auroc(np.repeat(ytr > np.median(ytr), 1), s)
                if auc > best[0]:
                    best = (auc, layer, c)
        # score each problem group once and copy it to its rows (GPU sums can differ in the last bits row to row)
        test_keys = sorted({(r["tier"], r["problem_name"]) for r in te})
        Xg = vec[[gidx[("test", variant, t, p)] for t, p in test_keys]]
        sg = fit_and_score(Xtr, ytr, Xg, best[1], best[2], weights=wtr, device=DEV)["probe"]
        by_key = dict(zip(test_keys, sg))
        sc = np.array([by_key[(r["tier"], r["problem_name"])] for r in te])
        # solvability probe: same site, layer and C, trained on the per-problem honest_correct rate; hack score = minus it
        yhc = np.array([np.mean(hc[k]) for k in keys])
        solv_g = -fit_and_score(Xtr, yhc, Xg, best[1], best[2], weights=wtr, device=DEV)["probe"]
        solv = np.array([dict(zip(test_keys, solv_g))[(r["tier"], r["problem_name"])] for r in te])
        if a.shuffle:
            yte = rng.permutation(yte)
        within = M.within_problem_auroc(yte, sc, gte)
        assert abs(within - 0.5) < 1e-9 or np.isnan(within), f"within-problem AUROC {within} != 0.5: pairing bug"
        prompt = lambda rows: [r["messages"][1]["content"] + " " + r["tier"] for r in rows]
        tf = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=50_000, sublinear_tf=True)
        lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(
            tf.fit_transform(prompt(tr)), [int(r["ACT"]) for r in tr] if not a.shuffle else rng.permutation([int(r["ACT"]) for r in tr]))
        base = lr.decision_function(tf.transform(prompt(te)))
        def boot(score_a, score_b, n=2000):
            """Problem-level bootstrap of pooled AUROC(a) - AUROC(b)."""
            probs_ = np.array(sorted(set(gte)))
            rows_of = {p_: np.where(gte == p_)[0] for p_ in probs_}
            rr = np.random.default_rng(0)
            diffs = []
            for _ in range(n):
                pick = np.concatenate([rows_of[p_] for p_ in rr.choice(probs_, len(probs_))])
                diffs.append(M.auroc(yte[pick], score_a[pick]) - M.auroc(yte[pick], score_b[pick]))
            return float(np.nanquantile(diffs, 0.025)), float(np.nanquantile(diffs, 0.975))
        g_lo, g_hi = boot(sc, base)
        s_lo, s_hi = boot(sc, solv)
        gain = M.auroc(yte, sc) - M.auroc(yte, base)
        matches = s_lo <= 0 <= s_hi or M.auroc(yte, solv) >= M.auroc(yte, sc)
        branch = ("(a)" if gain >= 0.10 and not matches else "(b)" if gain >= 0.05 and matches
                  else "(c)" if gain >= 0.05 else "(d)")
        report[variant] = {"probe_pooled": M.auroc(yte, sc), "probe_within": within, "layer": best[1], "C": best[2],
                           "cv_pooled": best[0], "prompt_feature_pooled": M.auroc(yte, base),
                           "gain_over_prompt_features": gain, "gain_ci": [g_lo, g_hi],
                           "solvability_pooled": M.auroc(yte, solv), "probe_minus_solvability_ci": [s_lo, s_hi],
                           "solvability_matches": bool(matches), "D3a_branch": branch,
                           "n_test_rows": len(te), "n_test_problems": len(set(gte))}
        print(variant, report[variant], flush=True)
    out = (Path("/data/scratch") if a.shuffle else ROOT / "results/qwen3-8b/anchor") / "h1a.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("->", out)


if __name__ == "__main__":
    main()
