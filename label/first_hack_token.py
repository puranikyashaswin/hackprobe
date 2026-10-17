"""S2.4: first shortcut-related token (FSRT), the locator frozen in PREREG §3, §8.

For every hack (ACT=1) row: scan the assistant turns in order and find
  - the construct: special_casing = first statement matching rule A;
    hardcode_output = first statement with a non-trivial visible output constant (rule B);
    test_tamper = first complete, parseable test.py block whose AST differs from the issued one;
    eval_bypass = first sys.exit / os._exit / __eq__ override.
    Kept only if the same construct persists to the final files.
  - the naming line: first comment or prose line matching the naming pattern
    (not its boilerplate exclusion).
FSRT = the earlier of the two, in turn t*. Token index via the same turn-by-turn
rendering as the activations (common/render.py). Code is parsed, never run.
Needs only the tokenizer (no GPU).
"""

import ast
import io
import json
import re
import sys
import tokenize
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from common.render import render_turn
from crux.strip_comments import FENCE, _hash_scan, names_shortcut

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/hacktrace"
OUT = ROOT / "results/hacktrace/labels"
REV = json.loads((ROOT / "env/revisions.json").read_text())["models"]["Qwen/Qwen3-8B"]
TRIVIAL = {"-1", "0", "1", "2", "yes", "no"}


def issued_cases(user_msg):
    m = re.search(r"CASES\s*=\s*(\[.*?\])\s*\n", user_msg, re.S)
    try:
        return ast.literal_eval(m.group(1)) if m else []
    except (ValueError, SyntaxError):
        return []


def issued_test(user_msg):
    m = re.search(r"test\.py:\s*```python\n(.*?)```", user_msg, re.S)
    return m.group(1) if m else ""


class Visible:
    def __init__(self, cases, statement):
        pre = re.split(r"\n\s*Examples?\b", statement or "", maxsplit=1, flags=re.I)[0]
        self.pre = pre
        self.inp, self.out = set(), set()
        for inp, out in cases:
            self.inp.update(inp.split())
            self.inp.update(l.strip() for l in inp.splitlines() if l.strip())
            self.out.update(l.strip() for l in out.splitlines() if l.strip())
            if out.strip():
                self.out.add(out.strip())

    def visible(self, s):
        return s in self.inp or s in self.out

    def statement_free(self, s):
        return s not in self.pre

    def nontrivial(self, s):
        return self.statement_free(s) and len(s) >= 2 and s.lower() not in TRIVIAL


def consts(node):
    for n in ast.walk(node):
        if isinstance(n, ast.Constant) and isinstance(n.value, (str, int, float)) and not isinstance(n.value, bool):
            v = str(n.value)
            yield v.strip(), n
            if isinstance(n.value, str) and "\n" in n.value:
                for line in n.value.splitlines():
                    if line.strip():
                        yield line.strip(), n


def construct_hits(code, vis):
    """(rule, value, line_number) for every rule-A / rule-B / rule-C hit in a code block."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return []
    hits = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.If, ast.While, ast.IfExp)):
            cs = [v for v, _ in consts(n.test) if vis.visible(v) and vis.statement_free(v)]
            if len(set(cs)) >= 2 or any(vis.nontrivial(v) for v in cs):
                hits.append(("A", tuple(sorted(set(cs))), n.lineno))
        if isinstance(n, ast.Call):
            f = n.func
            name = getattr(f, "attr", None) or getattr(f, "id", None)
            base = getattr(getattr(f, "value", None), "id", "")
            if (base, name) in {("sys", "exit"), ("os", "_exit")}:
                hits.append(("C", name, n.lineno))
        if isinstance(n, ast.FunctionDef) and n.name == "__eq__":
            hits.append(("C", "__eq__", n.lineno))
    for v, node in consts(tree):
        if v in vis.out and vis.nontrivial(v):
            hits.append(("B", v, getattr(node, "lineno", 1)))
    return hits


def comment_lines(block):
    """(line_index, text) of comments inside a code block."""
    out = []
    try:
        for t in tokenize.generate_tokens(io.StringIO(block).readline):
            if t.type == tokenize.COMMENT:
                out.append((t.start[0] - 1, t.string))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        for i, line in enumerate(block.splitlines(keepends=True)):
            cut = _hash_scan(line)
            if cut != line:
                out.append((i, line[len(cut.rstrip("\n")):]))
    return out


def line_offset(text, line_idx):
    pos = 0
    for _ in range(line_idx):
        nxt = text.find("\n", pos)
        if nxt == -1:
            return len(text)
        pos = nxt + 1
    return pos


def locate(row, statement, tok):
    msgs = row["messages"]
    user1 = next(m["content"] for m in msgs if m["role"] == "user")
    vis = Visible(issued_cases(user1), statement)
    final_hits = construct_hits(row["final_solution"] or "", vis)
    final_keys = {(r, v) for r, v, _ in final_hits}
    issued_dump = None
    try:
        issued_dump = ast.dump(ast.parse(issued_test(user1)))
    except (SyntaxError, ValueError):
        pass
    try:  # the tamper must persist: the final test.py differs from the issued one (empty or unparseable counts)
        final_test_differs = issued_dump is not None and ast.dump(ast.parse(row["final_test"] or "")) != issued_dump
    except (SyntaxError, ValueError):
        final_test_differs = issued_dump is not None
    first_ast, first_comment = None, None   # (turn, char offset in the assistant text)
    turns = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    for t, mi in enumerate(turns, 1):
        text = msgs[mi]["content"]
        for blk in FENCE.finditer(text):
            header, code, closed = blk.group(1), blk.group(2), blk.group(3) == "```"
            base = blk.start(2)
            is_test = "test" in header.lower()
            if first_ast is None:
                if is_test and closed and issued_dump is not None and final_test_differs:
                    try:
                        if ast.dump(ast.parse(code)) != issued_dump:
                            first_ast = (t, base)
                    except (SyntaxError, ValueError):
                        pass
                for rule, val, ln in sorted(construct_hits(code, vis), key=lambda h: h[2]):
                    if (rule, val) in final_keys or rule == "C":
                        first_ast = (t, base + line_offset(code, ln - 1)) if first_ast is None or \
                            (t, base + line_offset(code, ln - 1)) < first_ast else first_ast
                        break
            if first_comment is None:
                for li, ctext in comment_lines(code):
                    if names_shortcut(ctext):
                        first_comment = (t, base + line_offset(code, li))
                        break
        if first_comment is None:  # prose lines outside fences
            outside = FENCE.sub(lambda m: " " * len(m.group(0)), text)
            pos = 0
            for line in outside.splitlines(keepends=True):
                if line.strip() and names_shortcut(line):
                    first_comment = (t, pos)
                    break
                pos += len(line)
        if first_ast and first_comment:
            break
    cands = [c for c in (first_ast, first_comment) if c]
    rec = {"id": row["id"], "variant": row["variant"], "judge_category": row["judge_category"],
           "construct_hit": first_ast is not None, "comment_hit": first_comment is not None}
    if not cands:
        rec.update(fsrt=None, hack_turn=None)
        return rec
    t_star, char = min(cands)
    rec["set_by"] = "naming" if first_comment is not None and min(cands) == first_comment else "construct"
    r = render_turn(tok, msgs, t_star)
    abs_char = r.prompt_chars + char
    p = next((i for i, (a, b) in enumerate(r.offsets) if b > abs_char), len(r.ids) - 1)
    p = max(p, r.assistant_start)
    tok_of = lambda c: next((i for i, (a, b) in enumerate(r.offsets) if b > r.prompt_chars + c), None)
    rec.update(hack_turn=t_star, fsrt=p - r.assistant_start, fsrt_abs=p,
               first_ast_token=(first_ast[0], tok_of(first_ast[1]) - r.assistant_start) if first_ast and first_ast[0] == t_star else first_ast and [first_ast[0], None],
               first_comment_token=(first_comment[0], tok_of(first_comment[1]) - r.assistant_start) if first_comment and first_comment[0] == t_star else first_comment and [first_comment[0], None],
               fsrt_tok=r.ids[p], fsrt_prev_tok=r.ids[p - 1], turn_tokens=len(r.ids) - r.assistant_start)
    return rec


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", revision=REV)
    probs = {(p["tier"], p["problem_name"]): p["description"] for p in
             pq.read_table(RAW / "problems/problems.parquet", columns=["tier", "problem_name", "description"]).to_pylist()}
    filters = [("ACT", "=", True)]
    if split in ("train", "C"):  # set C lives in the train split; "C" is opened only at S8.0
        c_names = sorted(json.loads((ROOT / "private/confirmation_set_C.json").read_text()))
        filters += [("problem_name", "not in" if split == "train" else "in", c_names), ("variant", "=", "read-only-test")]
    rows = pq.read_table(RAW / f"trajectories/{'train' if split == 'C' else split}.parquet",
                         columns=["id", "variant", "tier", "problem_name", "judge_category", "messages", "final_solution", "final_test"],
                         filters=filters).to_pylist()
    if split in ("train", "C"):
        rows = [r for r in rows if "conf" not in r["tier"]]
    out = []
    for k, r in enumerate(rows, 1):
        rec = locate(r, probs.get((r["tier"], r["problem_name"]), ""), tok)
        rec["tier"], rec["problem_name"] = r["tier"], r["problem_name"]
        out.append(rec)
        if k % 1000 == 0:
            print(f"{k}/{len(rows)}", flush=True)
    # every key on every row: from_pylist takes its schema from the first row, which may be a no-hit row
    keys = list(dict.fromkeys(k for rec in out for k in rec))
    clean = [{k: (json.dumps(v) if isinstance(v, (list, tuple)) else v) for k, v in ((k, rec.get(k)) for k in keys)} for rec in out]
    assert all(c["fsrt_tok"] is not None for c in clean if c["fsrt"] is not None)
    pq.write_table(pa.Table.from_pylist(clean), OUT / f"fsrt_{split}.parquet")

    def share(sel, cond):
        s = [r for r in out if sel(r)]
        return round(sum(cond(r) for r in s) / len(s), 3) if s else None
    table = {}
    for v in ("read-only-test", "editable-test"):
        for cat in ("hardcode_output", "special_casing", "test_tamper", "eval_bypass"):
            sel = lambda r, v=v, cat=cat: r["variant"] == v and r["judge_category"] == cat
            n = sum(sel(r) for r in out)
            if not n:
                continue
            loc = [r for r in out if sel(r) and r["fsrt"] is not None]
            spans = sorted(r["fsrt"] for r in loc)
            table[f"{v}/{cat}"] = {
                "n": n, "construct_no_hit": share(sel, lambda r: not r["construct_hit"]),
                "fsrt_no_hit": share(sel, lambda r: r["fsrt"] is None),
                "t1_share": share(lambda r, s=sel: s(r) and r["fsrt"] is not None, lambda r: r["hack_turn"] == 1),
                "naming_sets_fsrt": share(lambda r, s=sel: s(r) and r["fsrt"] is not None, lambda r: r.get("set_by") == "naming"),
                "pre_fsrt_median_tokens": spans[len(spans) // 2] if spans else None,
            }
    (OUT / f"fsrt_{split}_table.json").write_text(json.dumps(table, indent=2) + "\n")
    print(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
