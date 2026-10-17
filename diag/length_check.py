"""D4 (exploratory): is the disguised-cell probe reading response length? Frozen prose-length LR
(fit on anchor train, read-only) on disg_full, and the frozen probe on length-matched pairs
(hack and honest row of the same problem whose prose lengths differ by <= 25%)."""

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from sklearn.linear_model import LogisticRegression

from crux.crux_extract import disguise_messages, load_cell
from detect import metrics as M
from detect.score_crux import Frozen
from diag.diagnostics import full_transcript, prose

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/hacktrace"


def main():
    F = Frozen("read-only-test")
    trm = {r["id"]: r["messages"] for r in pq.read_table(RAW / "trajectories/train.parquet", columns=["id", "messages"]).to_pylist()
           if r["id"] in {x["id"] for x in F.R}}
    xtr = np.log1p([len(prose(full_transcript(trm[r["id"]]))) for r in F.R])[:, None]
    lr = LogisticRegression(class_weight="balanced").fit(xtr, F.y)
    dg = {r["id"]: r for r in pq.read_table(ROOT / "results/hacktrace/crux/disguised.parquet").to_pylist()}
    out = {}
    for cell, flag, how in (("disg_orig", None, None), ("disg_full", "keep_full", "full")):
        ids = sorted(i for i, r in dg.items() if r["split"] == "test" and r.get("executed") and (flag is None or r[flag]))
        te = {r["id"]: r for r in pq.read_table(RAW / "trajectories/test.parquet", columns=["id", "messages", "final_solution"]).to_pylist() if r["id"] in set(ids)}
        A, _ = load_cell(cell, ids, "meancode")
        probe = F.probe(A)
        L = []
        for i in ids:
            m = te[i]["messages"]
            if how:
                m, _ = disguise_messages(m, te[i]["final_solution"], dg[i]["code_full"], how, i)
            L.append(len(prose(full_transcript(m))))
        L = np.array(L, float)
        y = np.array([dg[i]["ACT"] for i in ids])
        g = np.array([dg[i]["problem_name"] for i in ids])
        lscore = lr.decision_function(np.log1p(L)[:, None])
        # length-matched pairs: every (hack, honest) pair of a problem within 25% prose length; AUROC = P(probe hack > probe honest)
        wins, n, probs = 0.0, 0, set()
        for p in set(g):
            hi = np.where((g == p) & (y == 1))[0]
            ni = np.where((g == p) & (y == 0))[0]
            for a in hi:
                for b in ni:
                    if max(L[a], L[b]) <= 1.25 * max(min(L[a], L[b]), 1):
                        wins += 1.0 if probe[a] > probe[b] else (0.5 if probe[a] == probe[b] else 0.0)
                        n += 1
                        probs.add(p)
        out[cell] = {"prose_length_lr": M.within_problem_auroc(y, lscore, g), "probe_all": M.within_problem_auroc(y, probe, g),
                     "probe_length_matched_pairs": wins / n if n else None, "n_matched_pairs": n, "n_problems_matched": len(probs)}
        print("D4", cell, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[cell].items()}, flush=True)
    p = ROOT / "results/qwen3-8b/diag/diagnostics.json"
    rep = json.loads(p.read_text())
    rep["D4_length"] = out
    p.write_text(json.dumps(rep, indent=2, default=float) + "\n")


if __name__ == "__main__":
    main()
