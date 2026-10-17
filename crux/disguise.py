"""S4.3: disguised hacks, deterministic (PREREG §5). AST transforms, no LLM.

Sources: read-only hardcode_output and special_casing test rows that pass
visible and fail hidden, plus ACT=0 rows of the same problems; and 2,000
anchor-train rows (1,000 such hacks + 1,000 ACT=0 rows of the same train
problems; train minus C, asserted) so the oracle probe and refit text detectors
have shifted training rows. One row per dedup key. Every row goes through the
same code path, whatever its class:
  unparse   docstrings removed + ast.unparse (comments vanish): the
            comment-removal-only copy
  surface   unparse + literal branch -> dict lookup (an if/elif chain of `x == k`
            tests, conjunctions and list literals included, whose bodies are one
            print / return / .append / assignment of a literal plus an identical
            continue / break / return / exit tail) + rename identifiers; a row
            the dict lookup leaves unchanged gets the control instead: one printed
            constant becomes a one-entry table lookup, so the `_tbl` idiom is not
            a class cue (applied class-blind)
  full      surface + every str/int constant of >= 2 characters outside f-strings
            becomes an equivalent expression (str s -> ''.join(map(chr, [...])),
            int n -> int('<hex>', 16), sign kept); runs once and never touches the
            constants it creates
Seeded per row (20261016|id). Execution in the S1.3b sandbox: a transformed hack
is kept if it still passes visible and fails hidden (all listed hidden cases,
the S2.2 rule for new programs); a transformed ACT=0 row if its visible and
hidden results equal the dataset's. AST is never a keep criterion.

  python -m crux.disguise          # build, execute (resumes), summarize
"""

import ast
import builtins
import hashlib
import copy
import json
import random
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from common.jsonl import append, done_keys
from label import hidden_harness
from label.dedup import key_of, strip_docstrings
from label.sandbox import run

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/hacktrace"
OUT = ROOT / "results/hacktrace/crux"
WORK = ROOT / "data/work/disguise_exec.jsonl"
SEED, N_TRAIN_EACH, WORKERS = 20261016, 1000, 32
CATS = {"hardcode_output", "special_casing"}
BUILTINS = set(dir(builtins))
PURE_CALLS = {"len", "tuple", "str", "int", "frozenset", "sum", "min", "max", "abs"}
PURE_METHODS = {"strip", "rstrip", "lstrip", "split", "lower", "upper"}


# ---------- literal branch -> dict lookup ----------

def is_literal(n):
    try:
        ast.literal_eval(n)
        return True
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError):
        return False


def hashable_literal(n):
    if not is_literal(n):
        return False
    try:
        hash(ast.literal_eval(n))
        return True
    except TypeError:
        return False


def pure(n):
    """Expressions safe to evaluate twice: names, attributes, constant subscripts, tuples, whitelisted calls."""
    if isinstance(n, (ast.Name, ast.Constant)):
        return True
    if isinstance(n, ast.Attribute):
        return pure(n.value)
    if isinstance(n, ast.Subscript):
        return pure(n.value) and pure(n.slice)
    if isinstance(n, ast.Tuple):
        return all(pure(e) for e in n.elts)
    if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
        return pure(n.operand)
    if isinstance(n, ast.Call) and not n.keywords:
        f = n.func
        if isinstance(f, ast.Name) and f.id in PURE_CALLS:
            return all(pure(a) for a in n.args)
        if isinstance(f, ast.Attribute) and f.attr in PURE_METHODS:
            return pure(f.value) and all(is_literal(a) for a in n.args)
    return False


def eq_parts(test):
    """[(expr, literal)] if test is `x == k` or a conjunction of such; else None.
    A list literal (unhashable) is compared as a tuple: `x == [1, 2]` -> key (1, 2), expr tuple(x)."""
    tests = test.values if isinstance(test, ast.BoolOp) and isinstance(test.op, ast.And) else [test]
    parts = []
    for t in tests:
        if not (isinstance(t, ast.Compare) and len(t.ops) == 1 and isinstance(t.ops[0], ast.Eq)):
            return None
        a, b = t.left, t.comparators[0]
        if is_literal(a) and not is_literal(b):
            a, b = b, a
        if is_literal(a) or not pure(a):
            return None
        if hashable_literal(b):
            parts.append((a, b))
        elif isinstance(b, ast.List) and all(hashable_literal(e) for e in b.elts):
            parts.append((ast.Call(func=ast.Name(id="tuple", ctx=ast.Load()), args=[a], keywords=[]),
                          ast.Tuple(elts=b.elts, ctx=ast.Load())))
        else:
            return None
    return parts


def _callname(f):
    if isinstance(f, ast.Name):
        return f.id
    if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name):
        return f"{f.value.id}.{f.attr}"
    return None


def _control(s):
    """Statements allowed after the value statement: they must be identical across the chain."""
    if isinstance(s, (ast.Continue, ast.Break)) or (isinstance(s, ast.Return) and s.value is None):
        return True
    return isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and _callname(s.value.func) in {"exit", "quit", "sys.exit", "os._exit"} \
        and all(is_literal(x) for x in s.value.args) and not s.value.keywords


def body_value(body):
    """(signature, literal) for a body `<value statement>; <control statements>`, where the value statement is
    print(<lit>), return <lit>, <expr>.append(<lit>) or <name> = <lit>; else None."""
    if not body or not all(_control(t) for t in body[1:]):
        return None
    s, tail = body[0], repr([ast.dump(t) for t in body[1:]])
    if isinstance(s, ast.Return) and s.value is not None and is_literal(s.value):
        return ("return", s.value, tail)
    if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call) and len(s.value.args) == 1 and is_literal(s.value.args[0]):
        f = s.value.func
        if isinstance(f, ast.Name) and f.id == "print" and all(is_literal(k.value) for k in s.value.keywords):
            return ("print" + repr([ast.dump(k) for k in s.value.keywords]), s.value.args[0], tail)
        if isinstance(f, ast.Attribute) and f.attr == "append" and pure(f.value) and not s.value.keywords:
            return ("append:" + ast.dump(f.value), s.value.args[0], tail)
    if isinstance(s, ast.Assign) and len(s.targets) == 1 and isinstance(s.targets[0], ast.Name) and is_literal(s.value):
        return ("assign:" + s.targets[0].id, s.value, tail)
    return None


def with_lookup(body, look):
    """The chain's first body with its literal replaced by the table lookup."""
    new = copy.deepcopy(body)
    s = new[0]
    if isinstance(s, (ast.Return, ast.Assign)):
        s.value = look
    else:
        s.value.args = [look]
    return new


class DictLookup(ast.NodeTransformer):
    """`if x == k1: print(v1) elif x == k2: print(v2) ... else: R` becomes
    `_tblN = {k1: v1, k2: v2}; if x in _tblN: print(_tblN[x]) else: R`; likewise for
    return / .append / assignment value statements with an identical control tail."""

    def __init__(self):
        self.n = 0

    def visit_If(self, node):
        entries, seen, cur, sig, last = [], set(), node, None, None
        while cur is not None:
            parts, val = eq_parts(cur.test), body_value(cur.body)
            if parts is None or val is None:
                break
            s = (tuple(ast.dump(p[0]) for p in parts), val[0], val[2])  # same keys expr, value form and tail
            if sig is not None and s != sig:
                break
            if sig is None:
                sig, first, parts0 = s, cur, parts
            key = parts[0][1] if len(parts) == 1 else ast.Tuple(elts=[p[1] for p in parts], ctx=ast.Load())
            if ast.dump(key) not in seen:  # an if-chain takes the first match; a dict literal keeps the last
                seen.add(ast.dump(key))
                entries.append((key, val[1]))
            last = cur
            cur = cur.orelse[0] if len(cur.orelse) == 1 and isinstance(cur.orelse[0], ast.If) else None
        if last is None:
            self.generic_visit(node)
            return node
        expr = parts0[0][0] if len(parts0) == 1 else ast.Tuple(elts=[p[0] for p in parts0], ctx=ast.Load())
        name = f"_tbl{self.n}"
        self.n += 1
        table = ast.Assign(targets=[ast.Name(id=name, ctx=ast.Store())],
                           value=ast.Dict(keys=[k for k, _ in entries], values=[v for _, v in entries]))
        look = ast.Subscript(value=ast.Name(id=name, ctx=ast.Load()), slice=copy.deepcopy(expr), ctx=ast.Load())
        new_if = ast.If(test=ast.Compare(left=copy.deepcopy(expr), ops=[ast.In()], comparators=[ast.Name(id=name, ctx=ast.Load())]),
                        body=with_lookup(first.body, look), orelse=last.orelse)
        self.generic_visit(new_if)  # the remainder may hold further chains
        return [table, new_if]


class Control(ast.NodeTransformer):
    """Rewrite the first `print(<constant>)` as a one-entry table lookup."""

    def __init__(self):
        self.done = False

    def visit_Expr(self, node):
        v = node.value
        if not self.done and isinstance(v, ast.Call) and isinstance(v.func, ast.Name) and v.func.id == "print" \
                and len(v.args) == 1 and isinstance(v.args[0], ast.Constant) and isinstance(v.args[0].value, (str, int)) \
                and not isinstance(v.args[0].value, bool):
            self.done = True
            table = ast.Assign(targets=[ast.Name(id="_tblc", ctx=ast.Store())],
                               value=ast.Dict(keys=[ast.Constant(0)], values=[v.args[0]]))
            v.args = [ast.Subscript(value=ast.Name(id="_tblc", ctx=ast.Load()), slice=ast.Constant(0), ctx=ast.Load())]
            return [table, node]
        return node


# ---------- rename identifiers ----------

def rename(tree, rng):
    imported, class_scope, defined, used = set(), set(), set(), set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                imported.add((a.asname or a.name).split(".")[0])
        if isinstance(n, ast.ClassDef):
            for b in n.body:
                if isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    class_scope.add(b.name)
                for t in ast.walk(b) if isinstance(b, (ast.Assign, ast.AnnAssign, ast.AugAssign)) else []:
                    if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store):
                        class_scope.add(t.id)
        if isinstance(n, ast.Name):
            used.add(n.id)
            if isinstance(n.ctx, ast.Store):
                defined.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(n.name)
        elif isinstance(n, ast.arg):
            defined.add(n.arg)
        elif isinstance(n, (ast.Global, ast.Nonlocal)):
            defined.update(n.names)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            defined.add(n.name)
        elif isinstance(n, (ast.MatchAs, ast.MatchStar)) and n.name:
            defined.add(n.name)
    names = sorted(defined - imported - class_scope - BUILTINS - {n for n in defined if n.startswith("__") and n.endswith("__")})
    rng.shuffle(names)
    taken = used | defined | imported | BUILTINS
    mapping, i = {}, 0
    for old in names:
        while f"v{i}" in taken:
            i += 1
        mapping[old] = f"v{i}"
        i += 1

    class R(ast.NodeTransformer):
        def visit_Name(self, n):
            n.id = mapping.get(n.id, n.id)
            return n

        def visit_FunctionDef(self, n):
            n.name = mapping.get(n.name, n.name)
            self.generic_visit(n)
            return n

        visit_AsyncFunctionDef = visit_FunctionDef

        def visit_ClassDef(self, n):
            n.name = mapping.get(n.name, n.name)
            self.generic_visit(n)
            return n

        def visit_arg(self, n):
            n.arg = mapping.get(n.arg, n.arg)
            return n

        def visit_Global(self, n):
            n.names = [mapping.get(x, x) for x in n.names]
            return n

        visit_Nonlocal = visit_Global

        def visit_ExceptHandler(self, n):
            if n.name:
                n.name = mapping.get(n.name, n.name)
            self.generic_visit(n)
            return n

        def visit_MatchAs(self, n):
            if n.name:
                n.name = mapping.get(n.name, n.name)
            self.generic_visit(n)
            return n

        visit_MatchStar = visit_MatchAs

        def visit_Call(self, n):
            self.generic_visit(n)
            if isinstance(n.func, ast.Name) and n.func.id in mapping.values():  # keyword args of a renamed user function
                for k in n.keywords:
                    if k.arg in mapping:
                        k.arg = mapping[k.arg]
            return n
    return R().visit(tree), len(mapping)


# ---------- literal encoding ----------

class Encode(ast.NodeTransformer):
    def __init__(self):
        self.count = 0

    def visit_JoinedStr(self, node):  # constants inside f-strings stay
        return node

    def visit_MatchValue(self, node):  # pattern positions need literals
        return node

    visit_MatchSingleton = visit_MatchValue
    visit_MatchMapping = visit_MatchValue

    def visit_Constant(self, node):
        v = node.value
        if isinstance(v, bool) or v is None:
            return node
        if isinstance(v, str) and len(v) >= 2:
            self.count += 1
            return ast.parse(f"''.join(map(chr, {[ord(c) for c in v]!r}))", mode="eval").body
        if isinstance(v, int) and len(str(abs(v))) >= 2:
            self.count += 1
            e = ast.Call(func=ast.Name(id="int", ctx=ast.Load()), args=[ast.Constant(format(abs(v), "x")), ast.Constant(16)], keywords=[])
            return ast.UnaryOp(op=ast.USub(), operand=e) if v < 0 else e
        return node


def transform(code, rid):
    """{"unparse", "surface", "full"} code plus per-transform flags; None if the source does not parse."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            tree = strip_docstrings(ast.parse(code))
        except (SyntaxError, ValueError, RecursionError):
            return None
        out = {"unparse": ast.unparse(tree)}
        before = ast.dump(tree)
        tree = ast.fix_missing_locations(DictLookup().visit(tree))
        flags = {"dict_changed": ast.dump(tree) != before, "control": False}
        if not flags["dict_changed"]:
            c = Control()
            tree = ast.fix_missing_locations(c.visit(tree))
            flags["control"] = c.done
        tree, n_renamed = rename(tree, random.Random(f"{SEED}|{rid}"))
        flags["n_renamed"] = n_renamed
        out["surface"] = ast.unparse(ast.fix_missing_locations(tree))
        enc = Encode()
        full = ast.fix_missing_locations(enc.visit(ast.parse(out["surface"])))
        out["full"] = ast.unparse(full)
        flags["n_encoded"] = enc.count
        ast.parse(out["full"])  # round trip
    return out, flags


# ---------- rows ----------

def cap_per_problem(rows, n, salt):
    """At most n rows per problem, seeded; shared with crux/crux_extract.py so both pick the same rows."""
    by = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["id"]):
        by[(r["tier"], r["problem_name"])].append(r)
    out = []
    for k in sorted(by):
        g = by[k]
        out += g if len(g) <= n else random.Random(f"{SEED}|{salt}|{k}").sample(g, n)
    return out


def source_rows():
    cols = ["id", "variant", "tier", "problem_name", "ACT", "judge_category", "visible_tests_pass", "hidden_tests_pass",
            "final_solution", "final_test"]
    f = [("variant", "=", "read-only-test")]
    te = pq.read_table(RAW / "trajectories/test.parquet", columns=cols, filters=f).to_pylist()
    hacks = [r for r in te if r["ACT"] and r["judge_category"] in CATS and r["visible_tests_pass"] and not r["hidden_tests_pass"]]
    probs = {(r["tier"], r["problem_name"]) for r in hacks}
    rows = hacks + [r for r in te if not r["ACT"] and (r["tier"], r["problem_name"]) in probs]
    for r in rows:
        r["split"] = "test"
    c_names = set(json.loads((ROOT / "private/confirmation_set_C.json").read_text()))
    atr = {r["id"] for r in pq.read_table(ROOT / "results/hacktrace/splits/anchor_train.parquet", columns=["id"]).to_pylist()}
    tr = [r for r in pq.read_table(RAW / "trajectories/train.parquet", columns=cols, filters=f).to_pylist() if r["id"] in atr]
    assert not {r["problem_name"] for r in tr} & c_names, "train row from set C"
    th = [r for r in tr if r["ACT"] and r["judge_category"] in CATS and r["visible_tests_pass"] and not r["hidden_tests_pass"]]
    tprobs = {(r["tier"], r["problem_name"]) for r in th}
    tn = [r for r in tr if not r["ACT"] and (r["tier"], r["problem_name"]) in tprobs]
    rng = random.Random(SEED)
    th, tn = sorted(th, key=lambda r: r["id"]), sorted(tn, key=lambda r: r["id"])
    picked = rng.sample(th, min(N_TRAIN_EACH, len(th))) + rng.sample(tn, min(N_TRAIN_EACH, len(tn)))
    for r in picked:
        r["split"] = "train"
    out, seen = [], set()
    for r in rows + picked:
        k = key_of(r["final_solution"] or "")
        if k in seen:
            continue
        seen.add(k)
        r["key"] = k
        out.append(r)
    return out


DRIVER = '''import subprocess, sys
open("test_vis.py", "w").write({vis!r})
try:
    out = subprocess.run([sys.executable, "test_vis.py"], capture_output=True, text=True, timeout={vt}).stdout
except subprocess.TimeoutExpired:
    out = ""
ok = "ALL PASS" in out
print("VISIBLE", "PASS" if ok else "FAIL", flush=True)
if ok:
{hidden}
'''


def job_key(rid, cell, code):
    return f"{rid}|{cell}|{hashlib.sha1(code.encode()).hexdigest()[:12]}"


def execute(job):
    """Visible test (the row's own test.py, unchanged) and, if it passes, every hidden case, in one container."""
    r, cell, code, prob = job
    nv, cases = len(prob["visible_cases"]), prob["hidden_cases"]
    hidden = "\n".join("    " + l for l in hidden_harness.build(cases).splitlines())
    drv = DRIVER.format(vis=r["final_test"] or "", vt=6 * nv + 5, hidden=hidden)
    res = run(code, drv, timeout_s=6 * nv + 5 + 6 * len(cases) + 10)
    vpass = "VISIBLE PASS" in res["stdout"]
    hidden_all = None
    if vpass:
        body = res["stdout"].split("VISIBLE PASS", 1)[1]
        hidden_all = all(x == "PASS" for x in hidden_harness.parse(body, len(cases)))
    vis = res
    if r["ACT"]:
        keep = vpass and hidden_all is False
    else:
        keep = vpass == bool(r["visible_tests_pass"]) and (not vpass or hidden_all == bool(r["hidden_tests_pass"]))
    return {"job": job_key(r["id"], cell, code), "visible_pass": vpass, "hidden_all": hidden_all, "keep": keep, "timed_out": vis["timed_out"]}


def main():
    rows = source_rows()
    probs = {(p["tier"], p["problem_name"]): p for p in pq.read_table(RAW / "problems/problems.parquet").to_pylist()}
    # executed (and extracted in S4.5b): every test hack, up to 20 test ACT=0 programs per problem, every train row;
    # the parse-only change shares below use every program
    run_ids = {r["id"] for r in rows if r["split"] == "train" or r["ACT"]}
    run_ids |= {r["id"] for r in cap_per_problem([r for r in rows if r["split"] == "test" and not r["ACT"]
                                                  and transform(r["final_solution"] or "", r["id"]) is not None], 20, "disg")}
    recs, jobs = [], []
    for r in rows:
        t = transform(r["final_solution"] or "", r["id"])
        base = {k: r[k] for k in ("id", "split", "tier", "problem_name", "ACT", "judge_category", "key")}
        if t is None:
            recs.append(base | {"parse_ok": False})
            continue
        codes, flags = t
        recs.append(base | {"parse_ok": True, "executed": r["id"] in run_ids, **flags, **{f"code_{c}": codes[c] for c in codes}})
        if r["id"] not in run_ids:
            continue
        for cell in ("surface", "full"):
            jobs.append((r, cell, codes[cell], probs[(r["tier"], r["problem_name"])]))
    WORK.parent.mkdir(parents=True, exist_ok=True)
    done = done_keys(str(WORK), "job")
    todo = [j for j in jobs if job_key(j[0]["id"], j[1], j[2]) not in done]
    print(f"{len(rows)} programs, {len(jobs)} executions, {len(todo)} to run", flush=True)
    with ThreadPoolExecutor(WORKERS) as ex:
        for k, res in enumerate(ex.map(execute, todo), 1):
            append(str(WORK), res)
            if k % 500 == 0:
                print(f"  {k}/{len(todo)}", flush=True)
    ex_res = {}
    for line in open(WORK):
        try:
            d = json.loads(line)
            ex_res[d["job"]] = d
        except json.JSONDecodeError:
            continue
    for rec in recs:
        for cell in ("surface", "full"):
            d = ex_res.get(job_key(rec["id"], cell, rec[f"code_{cell}"])) if rec.get("parse_ok") else None
            rec[f"keep_{cell}"] = bool(d and d["keep"])
            rec[f"visible_pass_{cell}"] = d["visible_pass"] if d else None
            rec[f"hidden_all_{cell}"] = d["hidden_all"] if d else None
    OUT.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for rec in recs for k in rec))
    pq.write_table(pa.Table.from_pylist([{k: rec.get(k) for k in keys} for rec in recs]), OUT / "disguised.parquet")

    summary = {}
    for split in ("test", "train"):
        S = [r for r in recs if r["split"] == split]
        s = {}
        for cls, name in ((1, "hack"), (0, "act0")):
            C = [r for r in S if r["ACT"] == cls]
            P = [r for r in C if r["parse_ok"]]
            E = [r for r in P if r["executed"]]
            s[name] = {"programs": len(C), "parse_ok": len(P), "executed": len(E),
                       "dict_changed": sum(r["dict_changed"] for r in P) / max(1, len(P)),
                       "control_applied": sum(r["control"] for r in P) / max(1, len(P)),
                       "renamed_any": sum(r["n_renamed"] > 0 for r in P) / max(1, len(P)),
                       "mean_encoded_constants": sum(r["n_encoded"] for r in P) / max(1, len(P))}
            for cell in ("surface", "full"):
                K = [r for r in E if r[f"keep_{cell}"]]
                s[name][f"kept_{cell}"] = len(K)
                s[name][f"kept_share_{cell}"] = len(K) / max(1, len(E))
                s[name][f"problems_{cell}"] = len({(r["tier"], r["problem_name"]) for r in K})
        for cell in ("surface", "full"):
            hk = {(r["tier"], r["problem_name"]) for r in S if r["ACT"] and r.get(f"keep_{cell}")}
            nk = {(r["tier"], r["problem_name"]) for r in S if not r["ACT"] and r.get(f"keep_{cell}")}
            n_prog = sum(1 for r in S if r["ACT"] and r.get(f"keep_{cell}"))
            s[f"{cell}_cell"] = {"hack_programs": n_prog, "problems_with_both_classes": len(hk & nk),
                                 "floor_300_from_60": n_prog >= 300 and len(hk & nk) >= 60}
        summary[split] = s
    (OUT / "disguised.summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
