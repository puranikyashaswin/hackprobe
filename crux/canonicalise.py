"""Canonicalisation / constant-decoding baseline (PREREG §3).

Before the visible-literal rules and TF-IDF run on disguised cells, an AST pass
  - folds constant expressions (arithmetic on constants, string concatenation),
  - decodes chr(n), int('<digits>', base), str(c) and ''.join(...) over constants
    (so ''.join(map(chr, [...])) and int('7b', 16) become their literals),
  - inlines single-use helpers whose body is a single `return <expr>`,
  - normalises identifiers (v0, v1, ... in order of first appearance).
A disguise this pass reverses is "reversible". Code that does not parse is returned unchanged.
"""

import ast
import copy
import warnings

from crux.disguise import rename

MAX_LEN = 10_000


def _const(n):
    return isinstance(n, ast.Constant) and not isinstance(n.value, bool) and n.value is not None


def _const_seq(n):
    """Values of a list/tuple of constants, or of map(chr, <list of ints>)."""
    if isinstance(n, (ast.List, ast.Tuple)) and all(_const(e) for e in n.elts):
        return [e.value for e in n.elts]
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "map" and len(n.args) == 2 \
            and isinstance(n.args[0], ast.Name) and n.args[0].id == "chr" and isinstance(n.args[1], (ast.List, ast.Tuple)) \
            and all(_const(e) and isinstance(e.value, int) for e in n.args[1].elts):
        try:
            return [chr(e.value) for e in n.args[1].elts]
        except (ValueError, OverflowError):
            return None
    return None


class Fold(ast.NodeTransformer):
    def visit_JoinedStr(self, node):
        return node

    def _wrap(self, node, value):
        if isinstance(value, (str, int, float)) and not isinstance(value, bool) and len(str(value)) <= MAX_LEN:
            return ast.copy_location(ast.Constant(value), node)
        return node

    def visit_BinOp(self, node):
        self.generic_visit(node)
        if _const(node.left) and _const(node.right) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)):
            a, b = node.left.value, node.right.value
            if isinstance(node.op, ast.Mult) and (isinstance(a, str) or isinstance(b, str)):
                if (isinstance(a, int) and abs(a) > 100) or (isinstance(b, int) and abs(b) > 100):
                    return node
            try:
                return self._wrap(node, {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b,
                                         ast.FloorDiv: lambda: a // b, ast.Mod: lambda: a % b}[type(node.op)]())
            except Exception:
                return node
        return node

    def visit_UnaryOp(self, node):
        self.generic_visit(node)
        if _const(node.operand) and isinstance(node.op, (ast.USub, ast.UAdd)) and isinstance(node.operand.value, (int, float)):
            return self._wrap(node, -node.operand.value if isinstance(node.op, ast.USub) else node.operand.value)
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        f, args = node.func, node.args
        if node.keywords:
            return node
        try:
            if isinstance(f, ast.Name) and f.id == "chr" and len(args) == 1 and _const(args[0]) and isinstance(args[0].value, int):
                return self._wrap(node, chr(args[0].value))
            if isinstance(f, ast.Name) and f.id == "int" and len(args) in (1, 2) and all(_const(a) for a in args) \
                    and isinstance(args[0].value, str):
                return self._wrap(node, int(args[0].value, *(a.value for a in args[1:])))
            if isinstance(f, ast.Name) and f.id == "str" and len(args) == 1 and _const(args[0]):
                return self._wrap(node, str(args[0].value))
            if isinstance(f, ast.Attribute) and f.attr == "join" and _const(f.value) and isinstance(f.value.value, str) and len(args) == 1:
                seq = _const_seq(args[0])
                if seq is not None and all(isinstance(x, str) for x in seq):
                    return self._wrap(node, f.value.value.join(seq))
        except Exception:
            return node
        return node


def inline_helpers(tree):
    """Inline functions `def f(a, b): return <expr>` that are called exactly once and referenced nowhere else."""
    defs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef) and len(n.body) == 1
            and isinstance(n.body[0], ast.Return) and n.body[0].value is not None and not n.decorator_list
            and not n.args.vararg and not n.args.kwarg and not n.args.kwonlyargs and not n.args.defaults}
    refs = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and n.id in defs:
            refs[n.id] = refs.get(n.id, 0) + 1
    calls = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in defs and not n.keywords:
            calls.setdefault(n.func.id, []).append(n)
    single = {k for k in defs if refs.get(k) == 1 and len(calls.get(k, [])) == 1
              and len(calls[k][0].args) == len(defs[k].args.args)}

    class Sub(ast.NodeTransformer):
        def __init__(self, m):
            self.m = m

        def visit_Name(self, n):
            return copy.deepcopy(self.m[n.id]) if n.id in self.m and isinstance(n.ctx, ast.Load) else n

    class Inline(ast.NodeTransformer):
        def visit_Call(self, n):
            self.generic_visit(n)
            if isinstance(n.func, ast.Name) and n.func.id in single:
                d = defs[n.func.id]
                m = {a.arg: v for a, v in zip(d.args.args, n.args)}
                return Sub(m).visit(copy.deepcopy(d.body[0].value))
            return n
    tree = Inline().visit(tree)
    tree.body = [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name in single)]
    return tree


class _FirstSeen:
    """Stands in for the seeded shuffle in disguise.rename: orders names by first appearance."""

    def __init__(self, tree):
        self.order = {}
        for n in ast.walk(tree):
            name = getattr(n, "id", None) or getattr(n, "arg", None) or (n.name if isinstance(n, (ast.FunctionDef, ast.ClassDef)) else None)
            if isinstance(name, str) and name not in self.order:
                self.order[name] = len(self.order)

    def shuffle(self, names):
        names.sort(key=lambda x: self.order.get(x, 1 << 30))


def canonicalise(code):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            tree = ast.parse(code or "")
        except (SyntaxError, ValueError, RecursionError):
            return code or ""
        try:
            tree = Fold().visit(tree)
            tree = inline_helpers(tree)
            tree = Fold().visit(tree)
            tree, _ = rename(tree, _FirstSeen(tree))
            return ast.unparse(ast.fix_missing_locations(tree))
        except (RecursionError, ValueError, TypeError):
            return code or ""
