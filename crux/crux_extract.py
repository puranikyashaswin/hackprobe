"""S4.5b: crux extraction (pooled meancode / meangen on the final turn, as S3.1, plus the prompted judge).

Run (i), dataset rows as they are (S4.7, S4.8):
  overfit     visible pass, hidden fail, ACT=0, in test problems with honest_correct rows; one per dedup key
  hc_ref      honest_correct rows of those problems and of the S4.8 problems, up to 20 per problem
  s48_act1    ACT=1 rows of the S4.8 problems (>= 5 ACT=1 and >= 5 honest_correct), up to 20 per problem
Run (ii), crux cells (needs S4.2, S4.3):
  look_broad      S4.2 broad lookalikes (subset flags in meta)
  look_unflagged  control (i): unflagged honest rows of the same problems
  look_hacks      ordinary hacks of the lookalike problems, up to 20 per problem, no artefact rows
  disg_orig       S4.3 test sources as they are (hacks + up to 20 ACT=0 per problem)
  disg_unparse    the same rows, comment-removal-only copy
  disg_surface    the same rows kept by the surface set
  disg_full       the same rows kept by the full set (primary H2 cell)
  disg_full_train S4.3 train rows kept by the full set (oracle probe and refit text detectors)
A transformed trajectory gets the transform in every fenced block of every assistant
turn (so no earlier turn keeps the original), with the executed program in place of
the final solution's block, and naming prose removed by the S4.1 rule; the same for
both classes. Seed 20261016. Every row of a cell is extracted on one GPU type.

  python -m crux.crux_extract {i|ii} [--part k --nparts n] [--cells look_,disg_]
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch

from common.jsonl import append, done_keys
from common.render import n_turns, render_turn
from crux.disguise import cap_per_problem, transform
from crux.strip_comments import FENCE, _strip_prose, strip_code
from crux.strip_run import last_files
from detect.judge import TEMPLATE, first_user_turn
from extract.pooled import ROOT, check_pooled_means, load_model, pooled_means
from label.dedup import key_of

RAW = ROOT / "data/raw/hacktrace"
OUT = ROOT / "acts/qwen3-8b/hacktrace/crux2"
SEED, CAP = 20261016, 20


def one_per_key(rows, key):
    seen, out = set(), []
    for r in sorted(rows, key=lambda r: r["id"]):
        if key[r["id"]] not in seen:
            seen.add(key[r["id"]])
            out.append(r)
    return out


def jobs_i():
    cols = ["id", "variant", "tier", "problem_name", "ACT", "judge_category", "visible_tests_pass", "hidden_tests_pass"]
    te = pq.read_table(RAW / "trajectories/test.parquet", columns=cols).to_pylist()
    key = {r["id"]: r["key"] for r in pq.read_table(ROOT / "results/hacktrace/labels/dedup_test.parquet", columns=["id", "key"]).to_pylist()}
    pk = lambda r: (r["tier"], r["problem_name"])
    hc = [r for r in te if not r["ACT"] and r["visible_tests_pass"] and r["hidden_tests_pass"]]
    hc_probs = {pk(r) for r in hc}
    over = [r for r in te if not r["ACT"] and r["visible_tests_pass"] and r["hidden_tests_pass"] is False and pk(r) in hc_probs]
    n1, nhc = defaultdict(int), defaultdict(int)
    for r in te:
        n1[pk(r)] += bool(r["ACT"])
    for r in hc:
        nhc[pk(r)] += 1
    s48 = {p for p in hc_probs if n1[p] >= 5 and nhc[p] >= 5}
    ref_probs = {pk(r) for r in over} | s48
    jobs = [("overfit", "test", r["id"], "orig") for r in one_per_key(over, key)]
    jobs += [("hc_ref", "test", r["id"], "orig") for r in cap_per_problem([r for r in hc if pk(r) in ref_probs], CAP, "hc")]
    jobs += [("s48_act1", "test", r["id"], "orig") for r in cap_per_problem([r for r in te if r["ACT"] and pk(r) in s48], CAP, "s48")]
    return jobs


def jobs_ii():
    look = pq.read_table(ROOT / "results/hacktrace/crux/lookalikes.parquet").to_pylist()
    jobs = [("look_broad", "test", r["id"], "orig") for r in look if r["broad"]]
    jobs += [("look_unflagged", "test", r["id"], "orig") for r in look if r["unflagged"]]
    lp = {(r["tier"], r["problem_name"]) for r in look if r["broad"]}
    ded = {r["id"]: r for r in pq.read_table(ROOT / "results/hacktrace/labels/dedup_test.parquet", columns=["id", "key", "artefact"]).to_pylist()}
    te = pq.read_table(RAW / "trajectories/test.parquet", columns=["id", "tier", "problem_name", "ACT"]).to_pylist()
    hacks = [r for r in te if r["ACT"] and (r["tier"], r["problem_name"]) in lp and not ded[r["id"]]["artefact"]]
    hacks = cap_per_problem(one_per_key(hacks, {i: d["key"] for i, d in ded.items()}), CAP, "look")
    jobs += [("look_hacks", "test", r["id"], "orig") for r in hacks]
    dg = [r for r in pq.read_table(ROOT / "results/hacktrace/crux/disguised.parquet").to_pylist() if r["parse_ok"]]
    test = [r for r in dg if r["split"] == "test"]
    test = [r for r in test if r.get("executed")]  # every hack + the S4.3 per-problem cap of ACT=0 programs
    for r in test:
        jobs += [("disg_orig", "test", r["id"], "orig"), ("disg_unparse", "test", r["id"], "unparse")]
        if r["keep_surface"]:
            jobs.append(("disg_surface", "test", r["id"], "surface"))
        if r["keep_full"]:
            jobs.append(("disg_full", "test", r["id"], "full"))
    jobs += [("disg_full_train", "train", r["id"], "full") for r in dg if r["split"] == "train" and r["keep_full"]]
    return jobs


def disguise_messages(messages, final_solution, exec_code, how, rid):
    """Transform every fenced block of every assistant turn; the final solution's last block becomes exec_code."""
    fk = key_of(final_solution or "")
    where = None
    for t, m in enumerate(messages):
        if m["role"] == "assistant":
            for i, b in enumerate(FENCE.finditer(m["content"])):
                if key_of(b.group(2)) == fk:
                    where = (t, i)
    new, n_prose = [], 0
    for t, m in enumerate(messages):
        if m["role"] != "assistant":
            new.append(m)
            continue
        text, pieces, last = m["content"], [], 0
        for i, b in enumerate(FENCE.finditer(text)):
            before, n = _strip_prose(text[last: b.start()])
            n_prose += n
            if (t, i) == where:
                code = exec_code
            else:
                tr = transform(b.group(2), f"{rid}|{t}|{i}")
                code = tr[0][how] if tr else strip_code(b.group(2))[0]
            if code and not code.endswith("\n"):
                code += "\n"
            pieces.append(before + b.group(1) + code + b.group(3))
            last = b.end()
        tail, n = _strip_prose(text[last:])
        n_prose += n
        pieces.append(tail)
        new.append({**m, "content": "".join(pieces)})
    return new, {"final_block_found": where is not None, "prose_removed": n_prose}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", choices=["i", "ii"])
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--cells", default="", help="comma-separated prefixes; only these cells (every row of a cell stays on one machine)")
    a = ap.parse_args()
    jobs = jobs_i() if a.run == "i" else jobs_ii()
    if a.cells:
        jobs = [j for j in jobs if any(j[0].startswith(c) for c in a.cells.split(","))]
    jobs = [j for k, j in enumerate(sorted(jobs)) if k % a.nparts == a.part]
    c_names = set(json.loads((ROOT / "private/confirmation_set_C.json").read_text()))
    raw = {}
    for split in ("test", "train"):
        want = {j[2] for j in jobs if j[1] == split}
        if want:
            for r in pq.read_table(RAW / f"trajectories/{split}.parquet", columns=["id", "problem_name", "ACT", "messages", "final_solution", "final_test"]).to_pylist():
                if r["id"] in want:
                    raw[r["id"]] = r
    assert not any(raw[j[2]]["problem_name"] in c_names for j in jobs if j[1] == "train"), "train row from set C"
    codes = {}
    if a.run == "ii":
        for r in pq.read_table(ROOT / "results/hacktrace/crux/disguised.parquet", columns=["id", "code_unparse", "code_surface", "code_full"]).to_pylist():
            codes[r["id"]] = r
    tok, model = load_model()
    yes_id, no_id = tok.encode("Yes", add_special_tokens=False)[0], tok.encode("No", add_special_tokens=False)[0]
    checked = False
    by_cell = defaultdict(list)
    for j in jobs:
        by_cell[j[0]].append(j)
    for cell, cj in sorted(by_cell.items()):
        d = OUT / cell / (f"part_{a.part:02d}" if a.nparts > 1 else "")
        d.mkdir(parents=True, exist_ok=True)
        index = d / "index.jsonl"
        done = done_keys(str(index))
        todo = [j for j in cj if j[2] not in done]
        shard_no = len({json.loads(l)["shard"] for l in open(index)}) if index.exists() else 0
        print(f"{cell}: {len(todo)} to do", flush=True)
        for start in range(0, len(todo), 256):
            chunk = todo[start: start + 256]
            code = np.zeros((len(chunk), 37, 4096), np.float16)
            gen = np.zeros_like(code)
            meta = []
            for k, (_, split, rid, how) in enumerate(chunk):
                r = raw[rid]
                st = {}
                if how == "orig":
                    msgs = r["messages"]
                else:
                    msgs, st = disguise_messages(r["messages"], r["final_solution"], codes[rid][f"code_{how}"], how, rid)
                rr = render_turn(tok, msgs, n_turns(msgs))
                cm = torch.tensor(rr.code_mask, device="cuda")
                gm = torch.zeros_like(cm)
                gm[rr.assistant_start:] = True
                if not cm.any():
                    cm = gm
                if not checked and len(rr.ids) < 3000:
                    check_pooled_means(model, rr.ids, [cm, gm])
                    checked = True
                hc, hg = pooled_means(model, rr.ids, [cm, gm])
                torch.cuda.empty_cache()
                code[k], gen[k] = hc.half().cpu().numpy(), hg.half().cpu().numpy()
                sol, test = last_files(msgs)
                user = TEMPLATE.format(first_user_turn=first_user_turn(msgs), solution=sol, test=test)
                text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                with torch.no_grad():
                    lg = model(tok(text, add_special_tokens=False, return_tensors="pt").input_ids.cuda(), logits_to_keep=1).logits[0, -1]
                meta.append({"id": rid, "ACT": int(r["ACT"]), "how": how, **st, "solution": sol, "test": test,
                             "judge": float(lg[yes_id] - lg[no_id])})
            sd = d / f"shard_{shard_no:04d}"
            sd.mkdir(exist_ok=True)
            for layer in range(37):
                np.save(sd / f"L{layer:02d}_meancode.npy", code[:, layer])
                np.save(sd / f"L{layer:02d}_meangen.npy", gen[:, layer])
            with open(sd / "meta.jsonl", "w") as f:
                for rec in meta:
                    f.write(json.dumps(rec) + "\n")
            for off, (_, _, rid, _) in enumerate(chunk):
                append(str(index), {"id": rid, "shard": shard_no, "offset": off})
            print(f"  {cell} shard {shard_no}: {start + len(chunk)}/{len(todo)}", flush=True)
            shard_no += 1


def load_cell(cell, ids, pos):
    """Activations [n, 37, 4096] and meta of one cell, gathered over part folders."""
    dirs = [OUT / cell] + sorted((OUT / cell).glob("part_*"))
    where, meta, arrs = {}, {}, {}
    for d in dirs:
        if not (d / "index.jsonl").exists():
            continue
        for line in open(d / "index.jsonl"):
            r = json.loads(line)
            where[r["id"]] = (d, r["shard"], r["offset"])
        for sd in sorted(d.glob("shard_*")):
            for line in open(sd / "meta.jsonl"):
                m = json.loads(line)
                meta[m["id"]] = m
    out = np.zeros((len(ids), 37, 4096), np.float16)
    for k, rid in enumerate(ids):
        d, sh, off = where[rid]
        if (d, sh) not in arrs:
            arrs[(d, sh)] = np.stack([np.load(d / f"shard_{sh:04d}/L{l:02d}_{pos}.npy", mmap_mode="r") for l in range(37)], 1)
        out[k] = arrs[(d, sh)][off]
    return out, [meta[i] for i in ids]


if __name__ == "__main__":
    main()
