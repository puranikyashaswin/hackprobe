"""Text baselines, built to HackTrace's specification (arXiv 2610.03055, App. B and E).

- TF-IDF: word uni- and bigrams of "# solution.py ... # test.py ...", min_df 3,
  200,000 features, sublinear tf; logistic regression C=1, balanced.
- Visible-literal LR: seven features of solution.py (share of visible expected
  outputs and inputs that occur as literals, reads stdin, counts of literal
  prints, literal comparisons and branches, length); logistic regression.
- Length only.
All fitted on training rows and scored once on test rows.
"""

import ast

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def artifact_text(solution: str, test: str) -> str:
    return f"# solution.py\n{solution or ''}\n# test.py\n{test or ''}"


def tfidf_lr(train_texts, y_train, test_texts):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True,
                          token_pattern=r"(?u)\b\w+\b")
    xtr, xte = vec.fit_transform(train_texts), vec.transform(test_texts)
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000).fit(xtr, y_train)
    return lr.decision_function(xte)


def _literals(tree):
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Constant) and isinstance(n.value, (str, int, float)) and not isinstance(n.value, bool):
            out.append(str(n.value).strip())
    return out


def literal_features(solution: str, visible_cases: list) -> list:
    """The seven solution-side features. Unparseable code gets literal counts of 0."""
    code = solution or ""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError, RecursionError):
        tree = None
    lits = set(_literals(tree)) if tree is not None else set()
    outs = [c["output"].strip() for c in visible_cases if c["output"].strip()]
    ins = [c["input"].strip() for c in visible_cases if c["input"].strip()]
    share = lambda xs: sum(x in lits or x in code for x in xs) / len(xs) if xs else 0.0
    prints = comps = branches = 0
    if tree is not None:
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "print" \
                    and n.args and all(isinstance(a, ast.Constant) for a in n.args):
                prints += 1
            if isinstance(n, ast.Compare) and any(isinstance(c, ast.Constant) for c in n.comparators):
                comps += 1
            if isinstance(n, ast.If):
                branches += 1
    reads_stdin = int("input(" in code or "sys.stdin" in code)
    return [share(outs), share(ins), reads_stdin, prints, comps, branches, len(code)]


def literal_lr(train_feats, y_train, test_feats):
    sc = StandardScaler().fit(train_feats)
    lr = LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)
    lr.fit(sc.transform(train_feats), y_train)
    return lr.decision_function(sc.transform(test_feats))


def length_only(test_lengths):
    return np.asarray(test_lengths, float)
