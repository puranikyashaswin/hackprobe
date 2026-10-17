"""Hard-disguise attack ladder (exploratory, after S4.6 and D5).

Inputs: results/hacktrace/crux/hard_disguise_rewrites.jsonl (GPT-6 Astra; 4 rewrites per cheat,
1 per honest program, same instruction for both classes).
  check    every candidate runs in the sandbox (visible test + all hidden cases, one container);
           a cheat candidate is kept if it still passes visible and fails hidden, an honest rewrite
           if its visible / hidden results equal the dataset's.
  extract  code-only rendering (first user turn + one assistant turn holding only the program, as
           diag/code_only.py): layer-19 mean over code tokens for every kept candidate, every
           honest rewrite and every original program; plus the prompted Qwen3-8B judge (App. E
           prompt, code + test file).
  score    frozen detectors, nothing refit on rewrites:
             probe   code-only probe (diag/code_only.py: fit on code-only anchor-train renders,
                     layer 19, C = 0.1)
             tfidf   HackTrace frozen TF-IDF (anchor train final files)
             judge   prompted Qwen3-8B
           ladder (cheats only; honest programs use their single rewrite at every level):
             level 0  original programs (reference)
             level 1  blind: a seeded random kept candidate
             level 2  adaptive to text: the kept candidate with the lowest frozen TF-IDF score
             level 3  adaptive to the probe: the kept candidate with the lowest probe score
           Within-problem AUROC with problem-bootstrap CIs per level and detector.

  python -m crux.hard_ladder check | extract | score
"""

import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from common.jsonl import append, done_keys

ROOT = Path(__file__).resolve().parents[1]
CRUX = ROOT / "results/hacktrace/crux"
OUT = ROOT / "results/qwen3-8b/crux/hard_ladder"
ACTS = ROOT / "acts/qwen3-8b/hacktrace/hard_ladder"
SEED = 20261016


def load():
    rw = {json.loads(l)["id"]: json.loads(l) for l in open(CRUX / "hard_disguise_rewrites.jsonl")}
    dg = {r["id"]: r for r in pq.read_table(CRUX / "disguised.parquet").to_pylist() if r["id"] in rw}
    raw = {r["id"]: r for r in pq.read_table(ROOT / "data/raw/hacktrace/trajectories/test.parquet",
                                              columns=["id", "messages", "final_solution", "final_test", "visible_tests_pass", "hidden_tests_pass", "ACT"]).to_pylist()
           if r["id"] in rw}
    probs = {(p["tier"], p["problem_name"]): p for p in pq.read_table(ROOT / "data/raw/hacktrace/problems/problems.parquet").to_pylist()}
    return rw, dg, raw, probs


def items(rw, dg):
    """(item_key, id, kind, code): kind = orig | cand<k>."""
    out = []
    for i, w in sorted(rw.items()):
        out.append((f"{i}|orig", i, "orig", None))
        for k, c in enumerate(w["candidates"]):
            if c:
                out.append((f"{i}|cand{k}", i, f"cand{k}", c))
    return out


def cmd_check():
    from crux.disguise import execute
    rw, dg, raw, probs = load()
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "exec.jsonl"
    done = done_keys(str(path), "item")
    jobs = [(key, i, code) for key, i, kind, code in items(rw, dg) if kind != "orig" and key not in done]
    print(f"{len(jobs)} candidates to execute", flush=True)

    def one(j):
        key, i, code = j
        r = {**raw[i], "id": i}
        res = execute((r, "hard", code, probs[(dg[i]["tier"], dg[i]["problem_name"])]))
        return {"item": key, "keep": res["keep"], "visible_pass": res["visible_pass"], "hidden_all": res["hidden_all"]}
    with ThreadPoolExecutor(32) as ex:
        for k, rec in enumerate(ex.map(one, jobs), 1):
            append(str(path), rec)
            if k % 500 == 0:
                print(f"  {k}/{len(jobs)}", flush=True)
    ex_ = [json.loads(l) for l in open(path)]
    kept = {e["item"] for e in ex_ if e["keep"]}
    n_ch = sum(1 for i in rw if dg[i]["ACT"])
    ch_any = sum(1 for i in rw if dg[i]["ACT"] and any(f"{i}|cand{k}" in kept for k in range(4)))
    hon = sum(1 for i in rw if not dg[i]["ACT"] and f"{i}|cand0" in kept)
    s = {"candidates": len(ex_), "kept": len(kept), "cheats_with_a_kept_candidate": ch_any, "cheats": n_ch,
         "honest_kept": hon, "honest": len(rw) - n_ch}
    (OUT / "check.summary.json").write_text(json.dumps(s, indent=2) + "\n")
    print(json.dumps(s, indent=2))


def cmd_extract():
    import torch
    from common.render import render_turn
    from detect.judge import TEMPLATE
    from extract.pooled import load_model, pooled_means
    rw, dg, raw, probs = load()
    kept = {json.loads(l)["item"] for l in open(OUT / "exec.jsonl") if json.loads(l)["keep"]}
    todo = [(key, i, kind, code) for key, i, kind, code in items(rw, dg) if kind == "orig" or key in kept]
    tok, model = load_model()
    yes_id, no_id = tok.encode("Yes", add_special_tokens=False)[0], tok.encode("No", add_special_tokens=False)[0]
    ACTS.mkdir(parents=True, exist_ok=True)
    X = np.zeros((len(todo), 4096), np.float16)
    judge = []
    for k, (key, i, kind, code) in enumerate(todo):
        code = raw[i]["final_solution"] if kind == "orig" else code
        m = raw[i]["messages"][:2] + [{"role": "assistant", "content": f"```python\n{(code or '').rstrip()}\n```"}]
        rr = render_turn(tok, m, 1)
        cm = torch.tensor(rr.code_mask, device="cuda")
        if not cm.any():
            cm = torch.zeros_like(cm)
            cm[rr.assistant_start:] = True
        X[k] = pooled_means(model, rr.ids, [cm])[0][19].half().cpu().numpy()
        user1 = next(x["content"] for x in raw[i]["messages"] if x["role"] == "user")
        text = tok.apply_chat_template([{"role": "user", "content": TEMPLATE.format(first_user_turn=user1, solution=code or "", test=raw[i]["final_test"] or "")}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
        with torch.no_grad():
            lg = model(tok(text, add_special_tokens=False, return_tensors="pt").input_ids.cuda(), logits_to_keep=1).logits[0, -1]
        judge.append(float(lg[yes_id] - lg[no_id]))
        if k % 500 == 0:
            print(f"{k}/{len(todo)}", flush=True)
    np.save(ACTS / "L19.npy", X)
    (ACTS / "meta.json").write_text(json.dumps({"items": [t[0] for t in todo], "judge": judge}))
    print("->", ACTS)


def cmd_score():
    import torch
    from detect import metrics as M
    from detect.probe import fit_logreg
    from detect.score_crux import Frozen
    from detect.text_baselines import artifact_text, tfidf_lr
    rw, dg, raw, probs = load()
    meta = json.loads((ACTS / "meta.json").read_text())
    X = np.load(ACTS / "L19.npy").astype(np.float32)
    idx = {k: n for n, k in enumerate(meta["items"])}
    judge = np.array(meta["judge"])
    # code-only probe, frozen: fit on code-only anchor-train renders (diag/code_only.py), layer 19, C = 0.1
    co = ROOT / "acts/qwen3-8b/hacktrace/code_only"
    F = Frozen("read-only-test")
    Xtr = np.load(co / "train.npy", mmap_mode="r")[:, 19].astype(np.float32)
    ded = {r["id"]: r["cluster_size"] for r in pq.read_table(ROOT / "results/hacktrace/labels/dedup_train.parquet", columns=["id", "cluster_size"]).to_pylist()}
    wtr = np.array([1.0 / ded.get(r["id"], 1) for r in F.R])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    w, b = fit_logreg((Xtr - mu) / sd, F.y, 0.1, wtr, device="cuda" if torch.cuda.is_available() else "cpu")
    probe = ((X - mu) / sd) @ w + b
    codes = {}
    for key in meta["items"]:
        i, kind = key.split("|")
        codes[key] = raw[i]["final_solution"] if kind == "orig" else rw[i]["candidates"][int(kind[4:])]
    tfidf = dict(zip(meta["items"], tfidf_lr(F.txt, F.y, [artifact_text(codes[k], raw[k.split("|")[0]]["final_test"]) for k in meta["items"]])))
    rng = random.Random(SEED)
    levels = {f"level{n}": {} for n in range(4)}
    for i in sorted(rw):
        cands = [f"{i}|cand{k}" for k in range(4) if f"{i}|cand{k}" in idx]
        if not dg[i]["ACT"]:
            if f"{i}|orig" in idx:
                levels["level0"][i] = f"{i}|orig"
            if f"{i}|cand0" in idx:
                for n in (1, 2, 3):
                    levels[f"level{n}"][i] = f"{i}|cand0"
            continue
        levels["level0"][i] = f"{i}|orig"
        if cands:
            levels["level1"][i] = rng.choice(cands)
            levels["level2"][i] = min(cands, key=lambda k: tfidf[k])
            levels["level3"][i] = min(cands, key=lambda k: probe[idx[k]])
    report = {}
    for lv, sel in levels.items():
        ids = sorted(sel)
        y = np.array([dg[i]["ACT"] for i in ids])
        g = np.array([dg[i]["problem_name"] for i in ids])
        keys = [sel[i] for i in ids]
        sc = {"probe": np.array([probe[idx[k]] for k in keys]), "tfidf": np.array([tfidf[k] for k in keys]),
              "judge": np.array([judge[idx[k]] for k in keys])}
        rep = {d: {"within": M.within_problem_auroc(y, s, g), "ci": list(M.bootstrap_ci(y, s, g))} for d, s in sc.items()}
        rep["probe_minus_judge"] = list(M.paired_bootstrap_diff(y, sc["probe"], sc["judge"], g))
        rep["probe_minus_tfidf"] = list(M.paired_bootstrap_diff(y, sc["probe"], sc["tfidf"], g))
        rep["n"] = {"cheats": int(y.sum()), "honest": int(len(y) - y.sum()), "problems": len(M.per_problem(y, sc["probe"], g))}
        report[lv] = rep
        print(lv, rep["n"], {d: round(v["within"], 3) for d, v in rep.items() if isinstance(v, dict) and "within" in v},
              "probe-judge", [round(x, 3) for x in rep["probe_minus_judge"]], flush=True)
    (OUT / "ladder.json").write_text(json.dumps(report, indent=2) + "\n")
    print("->", OUT / "ladder.json")


if __name__ == "__main__":
    {"check": cmd_check, "extract": cmd_extract, "score": cmd_score}[sys.argv[1]]()
