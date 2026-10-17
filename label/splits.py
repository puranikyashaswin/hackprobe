"""S2.3 (anchor part): train and test sets for the first probe.

Anchor train: ~8,000 rows from train minus C, conf tiers excluded, balanced
across shortcut categories and variants (honest_fix_test counts as honest),
with 5 folds assigned by problem_name. Anchor test: up to 4 hacks + 4 honest
rows from each of the 239 problems that have both (S2.6).

Asserts: no problem in two folds, in both train and test, or in C.
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/hacktrace/trajectories"
OUT = ROOT / "results/hacktrace/splits"
C_FILE = ROOT / "private/confirmation_set_C.json"
SEED, N_POS, N_NEG, FOLDS = 20261015, 4000, 4000, 5
POS_CATS = ["hardcode_output", "special_casing", "test_tamper"]
COLS = ["id", "variant", "tier", "problem_name", "ACT", "judge_category"]


def main() -> None:
    rng = random.Random(SEED)
    c_names = set(json.loads(C_FILE.read_text()))

    train = [r for r in pq.read_table(RAW / "train.parquet", columns=COLS).to_pylist()
             if "conf" not in r["tier"] and r["problem_name"] not in c_names]
    pos = defaultdict(list)
    neg = defaultdict(list)
    for r in train:
        (pos[r["judge_category"]] if r["ACT"] else neg[r["variant"]]).append(r)
    picked = []
    other_pos = [r for cat, rows in pos.items() if cat not in POS_CATS for r in rows]
    per_cat = (N_POS - len(other_pos)) // len(POS_CATS)
    for cat in POS_CATS:
        rng.shuffle(pos[cat])
        picked += pos[cat][:per_cat]
    picked += other_pos
    for v in sorted(neg):
        rng.shuffle(neg[v])
        picked += neg[v][: N_NEG // len(neg)]

    names = sorted({r["problem_name"] for r in picked})
    rng.shuffle(names)
    fold = {n: i % FOLDS for i, n in enumerate(names)}
    for r in picked:
        r["fold"] = fold[r["problem_name"]]

    test = pq.read_table(RAW / "test.parquet", columns=COLS).to_pylist()
    eligible = {(r["variant"], r["tier"], r["problem_name"])
                for r in pq.read_table(OUT / "within_problem.parquet").to_pylist() if r["both_classes"]}
    groups = defaultdict(lambda: ([], []))
    for r in test:
        key = (r["variant"], r["tier"], r["problem_name"])
        if key in eligible:
            groups[key][0 if r["ACT"] else 1].append(r)
    anchor_test = []
    for key in sorted(groups):
        hacks, honest = groups[key]
        rng.shuffle(hacks)
        rng.shuffle(honest)
        anchor_test += hacks[:4] + honest[:4]

    tr_names, te_names = {r["problem_name"] for r in picked}, {r["problem_name"] for r in anchor_test}
    assert not tr_names & te_names, "problem in both train and test"
    assert not tr_names & c_names and not te_names & c_names, "problem from C"
    assert all(len({fold[n]}) == 1 for n in tr_names), "problem in two folds"

    pq.write_table(pa.Table.from_pylist(picked), OUT / "anchor_train.parquet")
    pq.write_table(pa.Table.from_pylist(anchor_test), OUT / "anchor_test.parquet")
    n_pos = sum(r["ACT"] for r in picked)
    cats = Counter(r["judge_category"] for r in picked if r["ACT"])
    summary = {
        "train_rows": len(picked), "train_pos": n_pos, "train_neg": len(picked) - n_pos,
        "train_problems": len(tr_names), "train_by_variant": dict(Counter(r["variant"] for r in picked)),
        "train_pos_share_by_category": {k: round(v / n_pos, 3) for k, v in cats.most_common()},
        "categories_below_15pct": [k for k in POS_CATS if cats[k] / n_pos < 0.15],
        "test_rows": len(anchor_test), "test_problems": len(groups),
        "test_by_variant": dict(Counter(r["variant"] for r in anchor_test)),
        "asserts": "no problem in two folds, in both train and test, or in C",
    }
    (OUT / "anchor.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
