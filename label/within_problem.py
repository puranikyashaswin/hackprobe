"""S2.6: within-problem index on the test split (development data).

Lists (variant, problem) groups that have both ACT=1 and ACT=0 rows, which is
HackTrace's own Table 2 set (expected 239 = 155 editable + 84 read-only), plus
(problem, final turn) groups and problems with >= 5 hacks and >= 5 honest_correct.
"""

import json
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "data/raw/hacktrace/trajectories/test.parquet"
OUT = ROOT / "results/hacktrace/splits"


def main() -> None:
    t = pq.read_table(TEST, columns=["variant", "tier", "problem_name", "ACT", "num_turns", "judge_category"]).to_pydict()
    acts, turns, counts = defaultdict(set), defaultdict(set), defaultdict(lambda: [0, 0])
    for v, tier, p, act, n, cat in zip(t["variant"], t["tier"], t["problem_name"], t["ACT"], t["num_turns"], t["judge_category"]):
        key = (v, tier, p)
        acts[key].add(bool(act))
        turns[key + (n,)].add(bool(act))
        if act:
            counts[key][0] += 1
        elif cat == "honest_correct":
            counts[key][1] += 1

    rows = []
    for (v, tier, p), s in sorted(acts.items()):
        rows.append({"variant": v, "tier": tier, "problem_name": p, "both_classes": len(s) == 2,
                     "n_hack": counts[(v, tier, p)][0], "n_honest_correct": counts[(v, tier, p)][1]})
    pq.write_table(pa.Table.from_pylist(rows), OUT / "within_problem.parquet")

    both = [r for r in rows if r["both_classes"]]
    summary = {
        "problems": len(rows),
        "both_classes": len(both),
        "both_editable": sum(r["variant"] == "editable-test" for r in both),
        "both_readonly": sum(r["variant"] == "read-only-test" for r in both),
        "problem_turn_groups_both": sum(len(s) == 2 for s in turns.values()),
        "ge5_hack_ge5_honest_correct": sum(r["n_hack"] >= 5 and r["n_honest_correct"] >= 5 for r in rows),
        "ge5_hack_ge5_honest_correct_readonly": sum(r["n_hack"] >= 5 and r["n_honest_correct"] >= 5
                                                     and r["variant"] == "read-only-test" for r in rows),
    }
    (OUT / "within_problem.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
