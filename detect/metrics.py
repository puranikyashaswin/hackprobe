"""Metrics used for every detector.

Primary: within-problem AUROC = mean over problems that have both classes of
the AUROC computed inside that problem (HackTrace Table 2). Confidence
intervals come from resampling problems, never rows.
"""

from collections import defaultdict

import numpy as np


def auroc(y, s) -> float:
    """Rank-based AUROC with ties counted half. Returns nan if one class is missing."""
    from scipy.stats import rankdata
    y, s = np.asarray(y, bool), np.asarray(s, float)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = rankdata(s)  # average ranks over ties
    return float((r[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def per_problem(y, s, groups) -> dict:
    by = defaultdict(list)
    for yi, si, g in zip(y, s, groups):
        by[g].append((yi, si))
    out = {}
    for g, pairs in by.items():
        yy, ss = zip(*pairs)
        a = auroc(yy, ss)
        if not np.isnan(a):
            out[g] = a
    return out


def within_problem_auroc(y, s, groups) -> float:
    vals = list(per_problem(y, s, groups).values())
    return float(np.mean(vals)) if vals else float("nan")


def bootstrap_ci(y, s, groups, n=2000, seed=0, alpha=0.05) -> tuple:
    """Problem-level bootstrap CI for the within-problem AUROC."""
    pp = per_problem(y, s, groups)
    vals = np.array(list(pp.values()))
    rng = np.random.default_rng(seed)
    means = vals[rng.integers(0, len(vals), size=(n, len(vals)))].mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def paired_bootstrap_diff(y, s1, s2, groups, n=2000, seed=0) -> tuple:
    """Difference in within-problem AUROC (s1 - s2), resampling problems; returns (diff, lo, hi)."""
    p1, p2 = per_problem(y, s1, groups), per_problem(y, s2, groups)
    keys = sorted(set(p1) & set(p2))
    d = np.array([p1[k] - p2[k] for k in keys])
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    return float(d.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def tpr_at_fpr(y, s, fpr=0.01) -> float:
    y, s = np.asarray(y, bool), np.asarray(s, float)
    thr = np.quantile(s[~y], 1 - fpr)
    return float((s[y] > thr).mean())
