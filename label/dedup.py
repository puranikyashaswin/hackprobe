"""S2.3a: dedup keys and the editable artefact flag (test split by default; `train` reads train minus C).

Key = ast.dump of the code after removing comments and docstrings (editable:
final_solution + final_test, because test tampering lives in the test file).
Code that does not parse falls back to whitespace-stripped text. Code is parsed,
never executed.

artefact = editable row whose final_test is empty, or lacks "ALL PASS" while
the final assistant turn ends inside an unclosed test.py fence.
"""

import ast
import hashlib
import json
import sys
import warnings
from collections import defaultdict
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
SPLIT = sys.argv[1] if len(sys.argv) > 1 else "test"
SRC = ROOT / f"data/raw/hacktrace/trajectories/{SPLIT}.parquet"
OUT = ROOT / "results/hacktrace/labels"


def strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return tree


def key_of(code: str) -> str:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            norm = ast.dump(strip_docstrings(ast.parse(code)))  # comments vanish in parsing
    except (SyntaxError, ValueError, RecursionError):
        norm = "".join(code.split())
    return hashlib.sha1(norm.encode()).hexdigest()


def is_artefact(variant: str, final_test: str, messages: list) -> bool:
    if variant != "editable-test":
        return False
    if not (final_test or "").strip():
        return True
    last = next((m["content"] for m in reversed(messages) if m["role"] == "assistant"), "")
    tail = last[last.rfind("```test.py"):] if "```test.py" in last else ""
    unclosed = bool(tail) and tail.count("```") == 1
    return "ALL PASS" not in final_test and unclosed


def main() -> None:
    cols = ["id", "variant", "tier", "problem_name", "ACT", "judge_category", "final_solution", "final_test", "messages"]
    filters = None
    if SPLIT == "train":  # never read set C's code
        c_names = sorted(json.loads((ROOT / "private/confirmation_set_C.json").read_text()))
        filters = [("problem_name", "not in", c_names)]
    t = pq.read_table(SRC, columns=cols, filters=filters).to_pylist()
    rows = []
    for r in t:
        code = r["final_solution"] or ""
        if r["variant"] == "editable-test":
            code += "\n# --- test.py ---\n" + (r["final_test"] or "")
        rows.append({"id": r["id"], "variant": r["variant"], "tier": r["tier"], "problem_name": r["problem_name"],
                     "ACT": bool(r["ACT"]), "judge_category": r["judge_category"], "key": key_of(code),
                     "artefact": is_artefact(r["variant"], r["final_test"], r["messages"])})

    groups = defaultdict(list)
    for r in rows:
        groups[(r["variant"], r["tier"], r["problem_name"], r["key"])].append(r)
    for g in groups.values():
        for r in g:
            r["cluster_size"] = len(g)
    pq.write_table(pa.Table.from_pylist(rows), OUT / f"dedup_{SPLIT}.parquet")

    def cell(sel):
        sub = [r for r in rows if sel(r)]
        uniq = {(r["variant"], r["tier"], r["problem_name"], r["key"]) for r in sub}
        n = len(sub)
        return {"n_rows": n, "n_unique_programs": len(uniq), "n_problems": len({r["problem_name"] for r in sub}),
                "dup_share": round(1 - len(uniq) / n, 3) if n else None}

    ro, ed = "read-only-test", "editable-test"
    summary = {
        "split": SPLIT,
        "readonly_ACT1": cell(lambda r: r["variant"] == ro and r["ACT"]),
        "readonly_ACT0": cell(lambda r: r["variant"] == ro and not r["ACT"]),
        "editable_ACT1": cell(lambda r: r["variant"] == ed and r["ACT"]),
        "editable_ACT0": cell(lambda r: r["variant"] == ed and not r["ACT"]),
        "artefact_test_tamper": sum(r["artefact"] for r in rows if r["judge_category"] == "test_tamper"),
        "test_tamper_rows": sum(r["judge_category"] == "test_tamper" for r in rows),
        "artefact_editable_ACT0": sum(r["artefact"] for r in rows if r["variant"] == ed and not r["ACT"]),
    }
    (OUT / f"dedup_{SPLIT}.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
