"""S2.3 (per-token part): rows and stored token spans for the lead-time steps.

Read-only test split:
  stratum A (t*=1 panel): every problem with >= 5 located t*=1 hacks and
    >= 5 ACT=0 rows; 5 hacks + 5 ACT=0 rows each.
  stratum B (t*>=2 panel): every problem with >= 1 located t*>=2 hack and
    >= 1 ACT=0 row that reached turn 2; up to 3 + 3 each.
Train (fits the per-token probe): the stratum-A rule on read-only, non-conf
  rows of train minus C, up to 100 problems.
Stored positions are spans in turn t*: positives max(0, p*-400) .. p*-1;
negatives 0 .. (largest p* among the problem's sampled positives) + 31, so the
+/-32-token matching of S3.6 always has room. Seed 20261016.
"""

import json
import random
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/hacktrace/trajectories"
LAB = ROOT / "results/hacktrace/labels"
OUT = ROOT / "results/hacktrace/splits"
K = 400


def build(split, rng, max_problems=None, with_b=True):
    fs = {r["id"]: r for r in pq.read_table(LAB / f"fsrt_{split}.parquet").to_pylist()}
    filters = [("variant", "=", "read-only-test")]
    c_names = set()
    if split in ("train", "C"):
        c_names = set(json.loads((ROOT / "private/confirmation_set_C.json").read_text()))
        filters.append(("problem_name", "not in" if split == "train" else "in", sorted(c_names)))
    rows = [r for r in pq.read_table(RAW / f"{'train' if split == 'C' else split}.parquet", columns=["id", "tier", "problem_name", "ACT", "num_turns"],
                                     filters=filters).to_pylist() if "conf" not in r["tier"]]
    if split == "C":
        assert {r["problem_name"] for r in rows} <= c_names
    else:
        assert not {r["problem_name"] for r in rows} & c_names
    by = defaultdict(lambda: {"h1": [], "h2": [], "neg": []})
    for r in rows:
        p = by[r["problem_name"]]
        if r["ACT"]:
            f = fs.get(r["id"])
            if f and f["fsrt"] is not None:
                (p["h1"] if f["hack_turn"] == 1 else p["h2"]).append(r | {"t_star": f["hack_turn"], "p_star": f["fsrt"],
                                                                         "fsrt_tok": f["fsrt_tok"], "fsrt_prev_tok": f["fsrt_prev_tok"]})
        else:
            p["neg"].append(r)
    out = []
    names = sorted(by)
    rng.shuffle(names)
    a_count = 0
    for name in names:
        p = by[name]
        if len(p["h1"]) >= 5 and len(p["neg"]) >= 5 and (max_problems is None or a_count < max_problems):
            a_count += 1
            hacks, negs = rng.sample(p["h1"], 5), rng.sample(p["neg"], 5)
            top = max(h["p_star"] for h in hacks)
            out += [h | {"stratum": "A", "label": 1, "span_start": max(0, h["p_star"] - K), "span_end": h["p_star"]} for h in hacks]
            out += [n | {"stratum": "A", "label": 0, "t_star": 1, "p_star": None, "span_start": 0, "span_end": top + 32} for n in negs]
        if with_b:
            negs2 = [n for n in p["neg"] if n["num_turns"] >= 2]
            if p["h2"] and negs2:
                hacks = rng.sample(p["h2"], min(3, len(p["h2"])))
                for h in hacks:
                    out.append(h | {"stratum": "B", "label": 1, "span_start": max(0, h["p_star"] - K), "span_end": h["p_star"]})
                for t in sorted({h["t_star"] for h in hacks}):
                    pool = [n for n in negs2 if n["num_turns"] >= t]
                    top = max(h["p_star"] for h in hacks if h["t_star"] == t)
                    for n in rng.sample(pool, min(3, len(pool))):
                        out.append(n | {"stratum": "B", "label": 0, "t_star": t, "p_star": None, "span_start": 0, "span_end": top + 32})
    # strata A and B draw negatives independently, so one ACT=0 row can appear at t*=1 (A) and t*>=2 (B);
    # (id, t*) is unique and keys everything downstream
    for r in out:
        r["key"] = f'{r["id"]}@{r["t_star"]}'
    assert len({r["key"] for r in out}) == len(out)
    return out


def main():
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "C":  # S8.0: the same rule on set C (PREREG amendment 2)
        rows = build("C", random.Random(20261016))
        pq.write_table(pa.Table.from_pylist(rows), OUT / "anchor_pertoken_C.parquet")
        print({s: {"rows": sum(r["stratum"] == s for r in rows), "problems": len({r["problem_name"] for r in rows if r["stratum"] == s}),
                   "hacks": sum(r["label"] for r in rows if r["stratum"] == s)} for s in ("A", "B")})
        return
    rng = random.Random(20261016)
    test = build("test", rng)
    train = build("train", rng, max_problems=100, with_b=False)
    pq.write_table(pa.Table.from_pylist(test), OUT / "anchor_pertoken.parquet")
    pq.write_table(pa.Table.from_pylist(train), OUT / "anchor_pertoken_train.parquet")

    def counts(rows):
        out = {}
        for s in ("A", "B"):
            sub = [r for r in rows if r["stratum"] == s]
            out[s] = {"rows": len(sub), "problems": len({r["problem_name"] for r in sub}),
                      "hacks": sum(r["label"] for r in sub), "tokens_stored": sum(r["span_end"] - r["span_start"] for r in sub)}
        return out
    summary = {"test": counts(test), "train": counts(train)}
    (OUT / "anchor_pertoken.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
