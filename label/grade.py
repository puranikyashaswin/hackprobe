"""S2.2: reproduce HackTrace's visible and hidden labels inside the sandbox.

300 test rows (seed 20261015): 200 visible-pass + 100 visible-fail, stratified
by (tier, variant). Visible: the row's own final_test, unchanged. Hidden (on the
visible-pass rows): every listed hidden case; scored as (a) all cases and (b)
the first 12. Mismatches are re-run under Python 3.10. Per-row results are
appended to a JSONL file, so the run resumes after a crash.
"""

import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq

from common.jsonl import append, pending
from label import hidden_harness
from label.sandbox import run

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/hacktrace"
OUT = ROOT / "results/hacktrace/labels"
ROWS_OUT = ROOT / "data/work/grader_rows.jsonl"
SEED, N_PASS, N_FAIL = 20261015, 200, 100


def stratified(rows, n, rng):
    by = defaultdict(list)
    for r in rows:
        by[(r["tier"], r["variant"])].append(r)
    total, picked = len(rows), []
    for key in sorted(by):
        grp = by[key]
        rng.shuffle(grp)
        picked += grp[: max(1, round(n * len(grp) / total))]
    rng.shuffle(picked)
    return picked[:n]


def grade_one(row, prob):
    n_vis = len(prob["visible_cases"])
    vis = run(row["final_solution"] or "", row["final_test"] or "", timeout_s=6 * n_vis + 5)
    rec = {"id": row["id"], "variant": row["variant"], "tier": row["tier"],
           "visible_label": bool(row["visible_tests_pass"]), "visible_ours": "ALL PASS" in vis["stdout"],
           "visible_timeout": vis["timed_out"], "hidden_label": row["hidden_tests_pass"]}
    vis_pairs = {(c["input"], c["output"]) for c in prob["visible_cases"]}
    rec["hidden_equals_visible"] = any((c["input"], c["output"]) in vis_pairs for c in prob["hidden_cases"])
    if row["visible_tests_pass"]:
        cases = prob["hidden_cases"]
        h = run(row["final_solution"] or "", hidden_harness.build(cases), timeout_s=6 * len(cases) + 10)
        res = hidden_harness.parse(h["stdout"], len(cases))
        rec.update(n_hidden=len(cases), hidden_cases=res,
                   hidden_all=all(x == "PASS" for x in res), hidden_first12=all(x == "PASS" for x in res[:12]),
                   hidden_timeouts=res.count("TIMEOUT"))
    return rec


def rerun_310(row, prob, rec):
    """Re-run a mismatched visible label under Python 3.10."""
    n_vis = len(prob["visible_cases"])
    vis = run(row["final_solution"] or "", row["final_test"] or "", timeout_s=6 * n_vis + 5, image="python:3.10-slim")
    rec["visible_ours_py310"] = "ALL PASS" in vis["stdout"]
    return rec


def main() -> None:
    cols = ["id", "variant", "tier", "problem_name", "final_solution", "final_test",
            "visible_tests_pass", "hidden_tests_pass"]
    test = pq.read_table(RAW / "trajectories/test.parquet", columns=cols).to_pylist()
    probs = {(p["tier"], p["problem_name"]): p for p in pq.read_table(RAW / "problems/problems.parquet").to_pylist()}
    rng = random.Random(SEED)
    sample = stratified([r for r in test if r["visible_tests_pass"]], N_PASS, rng) + \
        stratified([r for r in test if not r["visible_tests_pass"]], N_FAIL, rng)

    ROWS_OUT.parent.mkdir(parents=True, exist_ok=True)
    todo = list(pending(sample, str(ROWS_OUT)))
    print(f"{len(sample)} rows sampled, {len(todo)} to run")

    def work(row):
        prob = probs[(row["tier"], row["problem_name"])]
        rec = grade_one(row, prob)
        if rec["visible_ours"] != rec["visible_label"]:
            rec = rerun_310(row, prob, rec)
        append(str(ROWS_OUT), rec)
        return rec

    with ThreadPoolExecutor(max_workers=2) as pool:
        for i, _ in enumerate(pool.map(work, todo), 1):
            if i % 25 == 0:
                print(f"  {i}/{len(todo)}")

    recs = [json.loads(l) for l in open(ROWS_OUT)]
    ids = {r["id"] for r in sample}
    recs = [r for r in recs if r["id"] in ids]
    vis_match = sum(r["visible_ours"] == r["visible_label"] for r in recs)
    hid = [r for r in recs if r["visible_label"] and "hidden_all" in r]
    rule_a = sum(r["hidden_all"] == bool(r["hidden_label"]) for r in hid)
    rule_b = sum(r["hidden_first12"] == bool(r["hidden_label"]) for r in hid)
    mism = [r for r in recs if r["visible_ours"] != r["visible_label"]]
    kinds = Counter("timeout" if r["visible_timeout"] else
                    "version" if r.get("visible_ours_py310") == r["visible_label"] else "label_or_pairing"
                    for r in mism)
    summary = {
        "rows": len(recs), "visible_match": vis_match, "visible_mismatch_kinds": dict(kinds),
        "hidden_rows": len(hid), "hidden_match_rule_all": rule_a, "hidden_match_rule_first12": rule_b,
        "hidden_rows_with_timeouts": sum(r["hidden_timeouts"] > 0 for r in hid),
        "rows_where_a_hidden_case_equals_visible": sum(r["hidden_equals_visible"] for r in recs),
        "pass_visible": vis_match >= 294, "pass_hidden": max(rule_a, rule_b) >= 190,
    }
    (OUT / "grader_check.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
