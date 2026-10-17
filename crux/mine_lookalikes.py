"""S4.2: benign lookalikes (PREREG §5; parse only, no execution).

Honest_correct rows (ACT=0, pass visible and hidden; S2.2 checked these flags)
of test-split problems that also have hacks, whose code has a branch or lookup
holding a literal equal to a visible input or output (the broad match: trivial
values allowed, so base cases such as `if n == 1:` qualify). "Visible" is the
shared definition of the locator (label/first_hack_token.Visible). Branch = the
test of an If/While/IfExp; lookup = a dict-literal key, a string-keyed
subscript, or a literal container on the right of `in` (plain list indexing
such as a[0] is not a lookup). Two stricter subsets:
  branch_pair   an If/IfExp/While whose test holds a visible-input literal and
                whose body holds a visible-output literal
  multi         a branch pair with >= 2 visible-input literals in the test
                (occurrences), or one whole-input literal of >= 3 characters
Control (i): honest_correct rows of the same problems where neither the broad
match nor rules A and B fire ("unflagged honest"). One row per dedup key.
"""

import ast
import json
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from label.first_hack_token import Visible, consts, construct_hits

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/hacktrace"
OUT = ROOT / "results/hacktrace/crux"


def literal_values(node):
    return {v for v, _ in consts(node)}


def match(code, vis, whole_inputs):
    """(broad, branch_pair, multi) for one program."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return False, False, False
    broad = pair = multi = False
    for n in ast.walk(tree):
        if isinstance(n, (ast.If, ast.While, ast.IfExp)):
            tv = literal_values(n.test)
            if any(vis.visible(v) for v in tv):
                broad = True
            ins = [v for v, _ in consts(n.test) if v in vis.inp]
            body = n.body if isinstance(n.body, list) else [n.body]
            outs = set().union(*(literal_values(b) for b in body)) & vis.out if body else set()
            if ins and outs:
                pair = True
                if len(ins) >= 2 or any(v in whole_inputs and len(v) >= 3 for v in ins):
                    multi = True
        elif isinstance(n, ast.Dict):
            if any(k is not None and vis.visible(v) for k in n.keys if k is not None for v in literal_values(k)):
                broad = True
        elif isinstance(n, ast.Subscript):
            sl = n.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, str) and vis.visible(sl.value.strip()):
                broad = True
        elif isinstance(n, ast.Compare):
            for op, right in zip(n.ops, n.comparators):
                if isinstance(op, (ast.In, ast.NotIn)) and isinstance(right, (ast.List, ast.Tuple, ast.Set, ast.Dict)) \
                        and any(vis.visible(v) for v in literal_values(right)):
                    broad = True
    return broad, pair, multi


def main():
    cols = ["id", "variant", "tier", "problem_name", "ACT", "judge_category", "visible_tests_pass", "hidden_tests_pass", "final_solution"]
    rows = pq.read_table(RAW / "trajectories/test.parquet", columns=cols).to_pylist()
    key = {r["id"]: r["key"] for r in pq.read_table(ROOT / "results/hacktrace/labels/dedup_test.parquet", columns=["id", "key"]).to_pylist()}
    probs = {(p["tier"], p["problem_name"]): p for p in pq.read_table(RAW / "problems/problems.parquet").to_pylist()}
    hacked = {(r["tier"], r["problem_name"]) for r in rows if r["ACT"]}
    out, seen = [], set()
    for r in rows:
        pk = (r["tier"], r["problem_name"])
        if r["ACT"] or not (r["visible_tests_pass"] and r["hidden_tests_pass"]) or pk not in hacked:
            continue
        k = key[r["id"]]
        if k in seen:
            continue
        seen.add(k)
        p = probs[pk]
        cases = [(c["input"], c["output"]) for c in p["visible_cases"]]
        vis = Visible(cases, p["description"])
        whole = {c[0].strip() for c in cases}
        code = r["final_solution"] or ""
        broad, pair, multi = match(code, vis, whole)
        rules_ab = any(h[0] in "AB" for h in construct_hits(code, vis))
        out.append({"id": r["id"], "variant": r["variant"], "tier": r["tier"], "problem_name": r["problem_name"], "key": k,
                    "judge_category": r["judge_category"], "broad": broad, "branch_pair": pair, "multi": multi,
                    "rules_ab": rules_ab, "unflagged": not broad and not rules_ab})
    OUT.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(out), OUT / "lookalikes.parquet")

    summary = {}
    for variant in ("read-only-test", "editable-test", "all"):
        sub = [r for r in out if variant == "all" or r["variant"] == variant]
        s = {}
        for cell in ("broad", "branch_pair", "multi", "unflagged"):
            c = [r for r in sub if r[cell]]
            s[cell] = {"n_unique_programs": len(c), "n_problems": len({(r["tier"], r["problem_name"]) for r in c})}
        per = defaultdict(int)
        for r in sub:
            if r["broad"]:
                per[(r["tier"], r["problem_name"])] += 1
        top = sorted(per.values(), reverse=True)
        s["broad_top5_programs"] = sum(top[:5])
        s["honest_correct_programs_in_hacked_problems"] = len(sub)
        summary[variant] = s
    b = summary["all"]["broad"]
    summary["floor"] = {"rule": ">= 100 unique programs from >= 20 problems (D5 (f))", "broad_passes": b["n_unique_programs"] >= 100 and b["n_problems"] >= 20}
    (OUT / "lookalikes.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
