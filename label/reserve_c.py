"""S2.0: reserve the confirmation set C from train problems nobody has opened.

Reads only problem_name, tier, variant and ACT (no messages, no code).
Rule (PREREG.md §4, step S2.0): seed 20261014, tier-stratified sample of the
unopened train problems, start at 45% and grow by 5% until at least 60 problems
in C have both ACT classes in the read-only variant, keeping >= 250 train
problems for training. Otherwise drop C (confirmation moves to fresh generation).

Writes the problem names to private/ (gitignored) and only the sha256, seed and
counts to results/ (public).
"""

import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data/raw/hacktrace/trajectories/train.parquet"
# train problems whose trajectories were viewed before registration (excluded from C)
OPENED = ROOT / "results/hacktrace/splits/opened_train_problems.json"
PRIVATE = ROOT / "private/confirmation_set_C.json"
PUBLIC = ROOT / "results/hacktrace/splits/confirmation_set_C.meta.json"
SEED, START, STEP, NEED_BOTH, MIN_TRAIN = 20261014, 0.45, 0.05, 60, 250


def main() -> None:
    cols = pq.read_table(TRAIN, columns=["problem_name", "tier", "variant", "ACT"]).to_pydict()
    tier_of, ro_classes = {}, defaultdict(set)
    for name, tier, variant, act in zip(cols["problem_name"], cols["tier"], cols["variant"], cols["ACT"]):
        if "conf" not in tier:
            tier_of[name] = tier
        else:
            tier_of.setdefault(name, tier.replace("_conf", ""))
        if variant == "read-only-test":
            ro_classes[name].add(bool(act))

    opened = {r["problem_name"] for r in json.loads(OPENED.read_text())}
    names = sorted(tier_of)
    candidates = [n for n in names if n not in opened]
    by_tier = defaultdict(list)
    for n in candidates:
        by_tier[tier_of[n]].append(n)
    rng = random.Random(SEED)
    for tier in sorted(by_tier):
        rng.shuffle(by_tier[tier])

    both = lambda c: sum(1 for n in c if len(ro_classes[n]) == 2)
    frac, chosen = START, None
    while frac <= 1.0 + 1e-9:
        c = [n for tier in sorted(by_tier) for n in by_tier[tier][: math.ceil(frac * len(by_tier[tier]))]]
        if len(names) - len(c) < MIN_TRAIN:
            break
        if both(c) >= NEED_BOTH:
            chosen = c
            break
        frac = round(frac + STEP, 2)

    meta = {
        "seed": SEED, "train_problems": len(names), "opened": len(opened & set(names)),
        "candidates": len(candidates), "candidate_both_class_readonly": both(candidates),
        "candidates_by_tier": {t: len(v) for t, v in sorted(by_tier.items())},
    }
    if chosen is None:
        meta.update(decision="drop C: confirmation moves to fresh generation in November (S11)")
    else:
        body = json.dumps(sorted(chosen), indent=0)
        PRIVATE.parent.mkdir(exist_ok=True)
        PRIVATE.write_text(body + "\n")
        meta.update(
            fraction=frac, size=len(chosen), both_class_readonly=both(chosen),
            train_left=len(names) - len(chosen),
            sha256=hashlib.sha256(PRIVATE.read_bytes()).hexdigest(),
            by_tier={t: sum(1 for n in chosen if tier_of[n] == t) for t in sorted(by_tier)},
        )
    PUBLIC.write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
