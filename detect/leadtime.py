"""S3.6: H1b in-generation lead time (alignment frozen in PREREG §8).

For k = 1..K tokens before the first shortcut token (FSRT) in turn t*:
  positive = the hack row's activation at p*-k;
  negatives = ACT=0 rows of the same problem that reached turn t*, each at the
  index nearest to p*-k (within +/-32) holding the same token id (assert), else
  dropped for that k.
AUROC(k) = within-problem AUROC; n problems and pairs per k; truncated where
fewer than 20 problems remain. Panels: t*=1 (stratum A; probe fit on the train
per-token subset, layer and C by grouped CV on train) and t*>=2 (stratum B;
probe cross-fitted inside the stratum). L(theta) = largest k with the lower 95%
problem-bootstrap bound of AUROC(j) >= theta - 0.05 for every j <= k.
Baselines on the same pairs: TF-IDF on the prefix, TF-IDF on the last 32
tokens, comment-only TF-IDF, visible-literal rule count, previous-token LR,
the prefix judge (merged from detect/prefix_judge.py when present), and for
t*>=2 a feedback LR. Holm F1 test at k = L(0.8) for t*=1.
"""

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import AutoTokenizer

from common.render import render_turn
from detect import metrics as M
from detect.probe import fit_logreg
from extract.pertoken import LAYERS
from label.first_hack_token import Visible, construct_hits, issued_cases

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/hacktrace"
SPL = ROOT / "results/hacktrace/splits"
OUTF = ROOT / "results/qwen3-8b/anchor/leadtime.json"
K, WIN, MIN_PROBLEMS, KGRID_TRAIN = 400, 32, 20, 2
DEV = "cuda" if torch.cuda.is_available() else "cpu"
REV = json.loads((ROOT / "env/revisions.json").read_text())["models"]["Qwen/Qwen3-8B"]
CS = (0.01, 0.1, 1.0)


def load_acts(split):
    """key (id@t*) -> {"L": {layer: [n, 4096]}, "ids": [n], "span_start": int}, gathered over parts."""
    base = ROOT / f"acts/qwen3-8b/hacktrace/pertoken_{split}"
    out = {}
    for d in [base] + sorted(base.glob("part_*")):
        for sd in sorted(d.glob("shard_*")):
            rows = [json.loads(l) for l in open(sd / "rows.jsonl")]
            arrs = {l: np.load(sd / f"L{l:02d}.npy", mmap_mode="r") for l in LAYERS}
            ids = np.load(sd / "token_ids.npy")
            for m in rows:
                s, n = m["start"], m["length"]
                out[m["key"]] = {"L": {l: arrs[l][s: s + n] for l in LAYERS}, "ids": ids[s: s + n], "span_start": m["span_start"]}
    return out


def pairs_for(rows, acts, kmax=K, kstep=1):
    """List of (problem, k, pos_id, pos_idx, neg_id, neg_idx) under the frozen matching rule."""
    by = defaultdict(lambda: {"pos": [], "neg": defaultdict(list)})
    for r in rows:
        if r["key"] not in acts:
            continue
        if r["label"] == 1:
            by[r["problem_name"]]["pos"].append(r)
        else:
            by[r["problem_name"]]["neg"][r["t_star"]].append(r)
    out, dropped, tried = [], defaultdict(int), defaultdict(int)
    for prob, g in by.items():
        for pr in g["pos"]:
            pa = acts[pr["key"]]
            for k in range(1, kmax + 1, kstep):
                gp = pr["p_star"] - k                   # assistant-token index in turn t*
                li = gp - pa["span_start"]
                if gp < 0 or li < 0:
                    break
                t = int(pa["ids"][li])
                for nr in g["neg"].get(pr["t_star"], []):
                    na = acts[nr["key"]]
                    tried[k] += 1
                    lo, hi = max(0, gp - WIN - na["span_start"]), min(len(na["ids"]), gp + WIN + 1 - na["span_start"])
                    cand = [j for j in range(lo, hi) if int(na["ids"][j]) == t]
                    if not cand:
                        dropped[k] += 1
                        continue
                    j = min(cand, key=lambda j: abs(j + na["span_start"] - gp))
                    out.append((prob, k, pr["key"], li, nr["key"], j))
    drop_rate = {k: dropped[k] / tried[k] for k in tried if tried[k]}
    return out, drop_rate


def curve(pairs, score_pos, score_neg, kmax=K):
    """AUROC(k) per k from pair lists; score_pos/neg map (id, idx) -> score. Rows are kept by identity."""
    by_k = defaultdict(lambda: defaultdict(lambda: ({}, {})))
    for prob, k, pid, pi, nid, ni in pairs:
        p, n = by_k[k][prob]
        p[(pid, pi)] = score_pos[(pid, pi)]
        n[(nid, ni)] = score_neg[(nid, ni)]
    res = {}
    for k in sorted(by_k):
        per = [problem_auc(p, n) for p, n in by_k[k].values()]
        per = [x for x in per if not np.isnan(x)]
        if len(per) < MIN_PROBLEMS:
            break
        res[k] = {"auroc": float(np.mean(per)), "n_problems": len(per),
                  "n_pairs": int(sum(len(n) for _, n in by_k[k].values()))}
    return res, by_k


def problem_auc(p, n):
    return M.auroc([1] * len(p) + [0] * len(n), list(p.values()) + list(n.values()))


def lead_L(by_k, ks, theta, n_boot=2000, seed=0):
    """Point L and bootstrap CI of L, plus the per-k lower bounds."""
    probs = sorted({p for k in ks for p in by_k[k]})
    rng = np.random.default_rng(seed)
    per_k = {k: {p: problem_auc(a, b) for p, (a, b) in by_k[k].items()} for k in ks}
    boots = np.zeros((n_boot, len(ks)))
    for b in range(n_boot):
        pick = rng.choice(probs, len(probs))
        for j, k in enumerate(ks):
            vals = [per_k[k][p] for p in pick if p in per_k[k] and not np.isnan(per_k[k][p])]
            boots[b, j] = np.mean(vals) if vals else np.nan
    lower = np.nanquantile(boots, 0.025, axis=0)

    def first_fail(lb):
        L = 0
        for j, k in enumerate(ks):
            if lb[j] >= theta - 0.05:
                L = k
            else:
                break
        return L
    point = first_fail(lower)
    per_resample = [first_fail(np.where(np.isnan(boots[b]), -1, boots[b])) for b in range(n_boot)]
    return {"L": point, "L_resample_ci": [float(np.quantile(per_resample, 0.025)), float(np.quantile(per_resample, 0.975))],
            "lower_bound_by_k": {int(k): float(x) for k, x in zip(ks, lower)}}


def fit_probe(acts, pairs, layer, c):
    pos = sorted({(pid, pi) for _, _, pid, pi, _, _ in pairs})
    neg = sorted({(nid, ni) for _, _, _, _, nid, ni in pairs})
    X = np.stack([acts[i]["L"][layer][j] for i, j in pos + neg]).astype(np.float32)
    y = np.array([1] * len(pos) + [0] * len(neg))
    mu, sd = X.mean(0), X.std(0) + 1e-6
    w, b = fit_logreg((X - mu) / sd, y, c, device=DEV)
    score = lambda a: ((a.astype(np.float32) - mu) / sd) @ w + b
    score.params = {"mu": mu, "sd": sd, "w": w, "b": b}
    return score


def select_layer(acts, pairs):
    probs = sorted({p[0] for p in pairs})
    fold = {p: i % 5 for i, p in enumerate(probs)}
    best = (-1, None, None)
    for layer in LAYERS:
        for c in CS:
            sp, sn = {}, {}
            for f in range(5):
                tr = [p for p in pairs if fold[p[0]] != f]
                va = [p for p in pairs if fold[p[0]] == f]
                score = fit_probe(acts, tr, layer, c)
                for _, _, pid, pi, nid, ni in va:
                    sp[(pid, pi)] = float(score(acts[pid]["L"][layer][pi]))
                    sn[(nid, ni)] = float(score(acts[nid]["L"][layer][ni]))
            res, _ = curve(pairs, sp, sn)
            auc = np.mean([v["auroc"] for v in res.values()]) if res else -1
            if auc > best[0]:
                best = (auc, layer, c)
    return best


def texts_for(rows, tok, raw_msgs, probs_desc):
    """key -> (assistant text of turn t*, char offsets of its tokens, Visible, last feedback text)."""
    out = {}
    for r in rows:
        m = raw_msgs[r["id"]]
        rr = render_turn(tok, m, r["t_star"])
        s0 = rr.assistant_start
        offs = [(a - rr.prompt_chars, b - rr.prompt_chars) for a, b in rr.offsets[s0:]]
        user1 = next(x["content"] for x in m if x["role"] == "user")
        fb = m[2 * r["t_star"] - 1]["content"] if r["t_star"] >= 2 else ""
        out[r["key"]] = (rr.text[rr.prompt_chars:], offs, Visible(issued_cases(user1), probs_desc.get((r["tier"], r["problem_name"]), "")), fb)
    return out


def prefix_at(txt, offs, idx):
    """Text of the assistant turn through token idx, the token the probe reads at that position."""
    end = offs[idx][1] if idx < len(offs) else len(txt)
    return txt[:end]


def comment_text(prefix):
    return " ".join(re.findall(r"#[^\n]*", prefix))


def rule_count(prefix, vis):
    blocks = re.findall(r"```[^\n]*\n(.*?)(?:```|\Z)", prefix, re.S)
    return float(sum(len([h for h in construct_hits(b, vis) if h[0] in "AB"]) for b in blocks))


def text_scores(train_pairs, tr_text, tr_acts, test_pairs, te_text, te_acts):
    """Learned prefix baselines fit on train pairs; returns {name: (score_pos, score_neg)} for test pairs."""
    def samples(pairs, text, acts):
        items = sorted({(pid, pi, 1) for _, _, pid, pi, _, _ in pairs} | {(nid, ni, 0) for _, _, _, _, nid, ni in pairs})
        rows = []
        for i, j, y in items:
            txt, offs, vis, fb = text[i]
            g = j + acts[i]["span_start"]
            pre = prefix_at(txt, offs, g)
            prev_tok = int(acts[i]["ids"][j - 1]) if j >= 1 else -1
            last32 = txt[offs[max(0, g - 31)][0]: offs[g][1] if g < len(offs) else len(txt)] if offs else ""
            rows.append({"key": (i, j), "y": y, "prefix": pre, "last32": last32, "comments": comment_text(pre),
                         "prev": str(prev_tok), "rules": rule_count(pre, vis)})
        return rows
    tr, te = samples(train_pairs, tr_text, tr_acts), samples(test_pairs, te_text, te_acts)
    ytr = [r["y"] for r in tr]
    out = {}
    for name, field, kw in (("tfidf_prefix", "prefix", {}), ("tfidf_last32", "last32", {}), ("tfidf_comments", "comments", {}),
                            ("prev_token", "prev", {"analyzer": lambda s: [s]})):
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True, **kw) if "analyzer" not in kw else TfidfVectorizer(**kw)
        Xtr = vec.fit_transform([r[field] for r in tr])
        if Xtr.shape[1] == 0:
            continue
        lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(Xtr, ytr)
        s = lr.decision_function(vec.transform([r[field] for r in te]))
        out[name] = {r["key"]: float(v) for r, v in zip(te, s)}
    out["visible_literal_rules"] = {r["key"]: r["rules"] for r in te}
    return out


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", revision=REV)
    te_rows = pq.read_table(SPL / "anchor_pertoken.parquet").to_pylist()
    tr_rows = pq.read_table(SPL / "anchor_pertoken_train.parquet").to_pylist()
    c_names = set(json.loads((ROOT / "private/confirmation_set_C.json").read_text()))
    assert not {r["problem_name"] for r in tr_rows} & c_names
    te_acts, tr_acts = load_acts("test"), load_acts("train")
    desc = {(p["tier"], p["problem_name"]): p["description"] for p in
            pq.read_table(RAW / "problems/problems.parquet", columns=["tier", "problem_name", "description"]).to_pylist()}

    def msgs(split, rows):
        want = {r["id"] for r in rows}
        return {r["id"]: r["messages"] for r in pq.read_table(RAW / f"trajectories/{split}.parquet", columns=["id", "messages"]).to_pylist()
                if r["id"] in want}
    te_text = texts_for(te_rows, tok, msgs("test", te_rows), desc)
    tr_text = texts_for(tr_rows, tok, msgs("train", tr_rows), desc)
    report = {}

    # t*=1 panel: probe fit on train stratum A, scored once on test stratum A
    A_tr = [r for r in tr_rows if r["stratum"] == "A"]
    A_te = [r for r in te_rows if r["stratum"] == "A"]
    tr_pairs, _ = pairs_for(A_tr, tr_acts, kstep=KGRID_TRAIN)
    te_pairs, drop = pairs_for(A_te, te_acts)
    for prob, k, pid, pi, nid, ni in te_pairs:
        assert int(te_acts[pid]["ids"][pi]) == int(te_acts[nid]["ids"][ni]), "current-token mismatch"
    cv_auc, layer, c = select_layer(tr_acts, tr_pairs)
    score = fit_probe(tr_acts, tr_pairs, layer, c)
    np.savez(ROOT / "results/qwen3-8b/anchor/pertoken_probe.npz", layer=layer, C=c, **score.params)  # frozen for S3.8
    sp = {(pid, pi): float(score(te_acts[pid]["L"][layer][pi])) for _, _, pid, pi, _, _ in te_pairs}
    sn = {(nid, ni): float(score(te_acts[nid]["L"][layer][ni])) for _, _, _, _, nid, ni in te_pairs}
    probe_curve, by_k = curve(te_pairs, sp, sn)
    ks = sorted(probe_curve)
    panel = {"layer": layer, "C": c, "cv_auroc_train": cv_auc, "drop_rate_by_k": {int(k): v for k, v in drop.items()},
             "curves": {"probe": {int(k): v for k, v in probe_curve.items()}},
             "L": {str(th): lead_L(by_k, ks, th) for th in (0.8, 0.9)}}
    base = text_scores(tr_pairs, tr_text, tr_acts, te_pairs, te_text, te_acts)
    jd = {}
    for jf in sorted((ROOT / "results/qwen3-8b/anchor").glob("prefix_judge*.json")):
        jd.update(json.loads(jf.read_text()))
    if jd:
        jmap = {}
        for _, _, pid, pi, nid, ni in te_pairs:
            for i, j in ((pid, pi), (nid, ni)):
                rid, t = i.rsplit("@", 1)
                g = j + te_acts[i]["span_start"]
                # cut c covers tokens 0..c-1; the latest cut that has read no token past g
                pts = [int(x.split(":")[1]) for x in jd.get(rid, {}) if x.split(":")[0] == t and int(x.split(":")[1]) <= g + 1]
                if pts:
                    jmap[(i, j)] = jd[rid][f"{t}:{max(pts)}"]
        base["prefix_judge"] = jmap
    base_curves = {}
    for name, m in base.items():
        keep = [p for p in te_pairs if (p[2], p[3]) in m and (p[4], p[5]) in m]
        cv, _ = curve(keep, m, m)
        base_curves[name] = {int(k): v for k, v in cv.items()}
    panel["curves"].update(base_curves)
    L08 = panel["L"]["0.8"]["L"]
    if L08 > 0:
        kk = L08
        pk = [p for p in te_pairs if p[1] == kk]
        diffs = {}
        for name, m in base.items():
            sel = [p for p in pk if (p[2], p[3]) in m and (p[4], p[5]) in m]
            yy = np.array([1] * len(sel) + [0] * len(sel))
            gg = np.array([p[0] for p in sel] * 2)
            s_probe = np.array([sp[(p[2], p[3])] for p in sel] + [sn[(p[4], p[5])] for p in sel])
            s_base = np.array([m[(p[2], p[3])] for p in sel] + [m[(p[4], p[5])] for p in sel])
            diffs[name] = dict(zip(("diff", "lo", "hi"), M.paired_bootstrap_diff(yy, s_probe, s_base, gg, n=10000)))
        panel["F1_at_L08"] = {"k": kk, "probe_minus_baseline": diffs}
    report["t1"] = panel
    print("t*=1:", {"layer": layer, "L0.8": panel["L"]["0.8"]["L"], "L0.9": panel["L"]["0.9"]["L"],
                    "auroc_k1": probe_curve.get(1, {}).get("auroc")}, flush=True)

    # t*>=2 panel: probe cross-fitted inside stratum B (descriptive)
    B_te = [r for r in te_rows if r["stratum"] == "B"]
    b_pairs, b_drop = pairs_for(B_te, te_acts)
    if b_pairs:
        probsB = sorted({p[0] for p in b_pairs})
        foldB = {p: i % 5 for i, p in enumerate(probsB)}
        spB, snB = {}, {}
        for f in range(5):
            trp = [p for p in b_pairs if foldB[p[0]] != f]
            vap = [p for p in b_pairs if foldB[p[0]] == f]
            if not trp or not vap:
                continue
            sc = fit_probe(te_acts, trp, layer, c)
            for _, _, pid, pi, nid, ni in vap:
                spB[(pid, pi)] = float(sc(te_acts[pid]["L"][layer][pi]))
                snB[(nid, ni)] = float(sc(te_acts[nid]["L"][layer][ni]))
        keep = [p for p in b_pairs if (p[2], p[3]) in spB and (p[4], p[5]) in snB]
        cB, _ = curve(keep, spB, snB)
        report["t2plus"] = {"curves": {"probe_crossfit": {int(k): v for k, v in cB.items()}},
                            "n_pairs": len(b_pairs), "n_problems": len(probsB), "descriptive_only": True}
    OUTF.write_text(json.dumps(report, indent=2) + "\n")
    print("->", OUTF)


def confirm():
    """S8.0: the frozen S3.6 probe on set C, no refit (PREREG amendment 2; descriptive, development L(0.8) = 0)."""
    P = np.load(ROOT / "results/qwen3-8b/anchor/pertoken_probe.npz")
    layer, mu, sd, w, b = int(P["layer"]), P["mu"], P["sd"], P["w"].reshape(-1), float(P["b"])
    rows = pq.read_table(SPL / "anchor_pertoken_C.parquet").to_pylist()
    acts = load_acts("C")
    report = {"layer": layer}
    for panel, stratum in (("t1", "A"), ("t2plus", "B")):
        pairs, drop = pairs_for([r for r in rows if r["stratum"] == stratum], acts)
        if not pairs:
            continue
        sc = lambda i, j: float(((acts[i]["L"][layer][j].astype(np.float32) - mu) / sd) @ w + b)
        sp = {(pid, pi): sc(pid, pi) for _, _, pid, pi, _, _ in pairs}
        sn = {(nid, ni): sc(nid, ni) for _, _, _, _, nid, ni in pairs}
        cv, by_k = curve(pairs, sp, sn)
        ks = sorted(cv)
        report[panel] = {"curve": {int(k): v for k, v in cv.items()}, "drop_rate_by_k": {int(k): v for k, v in drop.items()},
                         "L": {str(th): lead_L(by_k, ks, th) for th in (0.8, 0.9)} if ks else None}
        print(panel, {k: round(cv[k]["auroc"], 3) for k in ks[:5]}, "n_prob@1", cv.get(1, {}).get("n_problems"), flush=True)
    out = ROOT / "results/qwen3-8b/confirm/leadtime.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("->", out)


if __name__ == "__main__":
    import sys
    confirm() if len(sys.argv) > 1 and sys.argv[1] == "confirm" else main()
