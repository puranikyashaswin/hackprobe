"""S4.1: strip comments and docstrings from code, changing nothing else.

Comments are removed with the tokenize module and docstring statements are
deleted by line span, so every other character stays where it was. If tokenize
raises part-way, the comments it already yielded are removed and the rest of
the block is stripped by a quote-aware "#" scan, so no row is ever excluded
(S4.1, PREREG §5). Prose lines outside code that match the frozen naming
pattern (PREREG §8) are deleted unless they match the boilerplate exclusion.
"""

import ast
import io
import re
import tokenize

FENCE = re.compile(r"(```[^\n]*\n)(.*?)(```|\Z)", re.S)
NAMING = re.compile(
    r"hard[- _]?cod|special[- _]?cas|expected[ _](output|result|answer|value)|example|sample|"
    r"(given|provided|specific|visible|known)[ _](input|output|test|case|answer)|test[ _]cases?|"
    r"pass(es|ing)? (the|all|these)? ?tests?|directly (output|return|print)|placeholder", re.I)
BOILERPLATE = re.compile(
    r"(number of|read(ing)?( in)?( all| the)?|each|every|all|between|per|loop over|iterate over|"
    r"process(ing)?( each)?|this) test[ _]cases?|^#\s*for example\b|^#\s*example:", re.I)


def names_shortcut(line: str) -> bool:
    s = line.strip()
    return bool(NAMING.search(s)) and not BOILERPLATE.search(s)


def _hash_scan(line: str) -> str:
    """Cut a line at the first '#' outside a string literal."""
    quote = None
    i = 0
    while i < len(line):
        ch = line[i]
        if quote:
            if ch == "\\":
                i += 2
                continue
            if line.startswith(quote, i):
                i += len(quote)
                quote = None
                continue
        else:
            if line.startswith('"""', i) or line.startswith("'''", i):
                quote = line[i: i + 3]
                i += 3
                continue
            if ch in "\"'":
                quote = ch
            elif ch == "#":
                nl = "\n" if line.endswith("\n") else ""
                return line[:i].rstrip(" \t") + nl
        i += 1
    return line


def _docstring_lines(code: str):
    drop, keep_pass = set(), {}
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        return drop, keep_pass
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and body \
                and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
                and isinstance(body[0].value.value, str):
            drop.update(range(body[0].lineno - 1, body[0].end_lineno))
            if len(body) == 1 and not isinstance(node, ast.Module):
                keep_pass[body[0].lineno - 1] = " " * body[0].col_offset + "pass\n"
    return drop, keep_pass


def strip_code(code: str) -> tuple:
    """Return (stripped_code, mode). mode is "tokenize" or "fallback" (tokenize raised part-way)."""
    lines = code.splitlines(keepends=True)
    out = [list(l) for l in lines]
    touched, mode, done_upto = set(), "tokenize", len(lines)
    last_line = 0
    try:
        for t in tokenize.generate_tokens(io.StringIO(code).readline):
            if t.type == tokenize.COMMENT:
                (r, c0), (_, c1) = t.start, t.end
                touched.add(r - 1)
                for c in range(c0, c1):
                    out[r - 1][c] = ""
            last_line = t.end[0]
    except (tokenize.TokenError, IndentationError, SyntaxError):
        mode, done_upto = "fallback", last_line
    drop, keep_pass = _docstring_lines(code)
    result = []
    for i, chars in enumerate(out):
        if i in drop:
            if i in keep_pass:
                result.append(keep_pass[i])
            continue
        line = "".join(chars)
        if mode == "fallback" and i >= done_upto:
            new = _hash_scan(line)
            if new != line:
                touched.add(i)
            line = new
        if i in touched:
            if line.strip() == "":
                continue  # a line that held only a comment disappears
            nl = "\n" if line.endswith("\n") else ""
            line = line.rstrip("\n").rstrip(" \t") + nl  # spaces that stood before the comment
        result.append(line)
    return "".join(result), mode


def _strip_prose(text: str) -> tuple:
    kept, n = [], 0
    for line in text.splitlines(keepends=True):
        if names_shortcut(line):
            n += 1
            continue
        kept.append(line)
    return "".join(kept), n


def strip_message(text: str) -> tuple:
    """Strip every fenced block (an unclosed fence runs to the end) and naming prose outside fences.
    Returns (text, n_fallback_blocks, n_prose_lines_removed)."""
    fallback = prose = 0
    pieces, last = [], 0
    for m in FENCE.finditer(text):
        before, n = _strip_prose(text[last: m.start()])
        prose += n
        code, mode = strip_code(m.group(2))
        fallback += mode == "fallback"
        pieces.append(before + m.group(1) + code + m.group(3))
        last = m.end()
    tail, n = _strip_prose(text[last:])
    prose += n
    pieces.append(tail)
    return "".join(pieces), fallback, prose
