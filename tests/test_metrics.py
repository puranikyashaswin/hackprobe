import numpy as np

from detect.metrics import auroc, bootstrap_ci, paired_bootstrap_diff, tpr_at_fpr, within_problem_auroc


def test_auroc_basic_and_ties():
    assert auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    assert auroc([0, 1], [0.5, 0.5]) == 0.5
    assert np.isnan(auroc([1, 1], [0.1, 0.2]))


def test_auroc_matches_pair_count():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200).astype(bool)
    s = rng.normal(size=200) + y
    pairs = np.mean([(a > b) + 0.5 * (a == b) for a in s[y] for b in s[~y]])
    assert abs(auroc(y, s) - pairs) < 1e-12


def test_within_problem_ignores_between_problem_signal():
    # problem A: all scores high, problem B: all low; inside each, random -> about 0.5
    rng = np.random.default_rng(1)
    y = np.tile([0, 1], 200)
    g = np.repeat(["A", "B"], 200)
    s = rng.normal(size=400) + (g == "A") * 10
    assert abs(within_problem_auroc(y, s, g) - 0.5) < 0.1


def test_bootstrap_and_paired_diff():
    rng = np.random.default_rng(2)
    g = np.repeat(np.arange(40), 10)
    y = np.tile([0, 1], 200)
    good = rng.normal(size=400) + 2 * y
    bad = rng.normal(size=400)
    lo, hi = bootstrap_ci(y, good, g, n=500)
    assert lo < within_problem_auroc(y, good, g) < hi
    d, dlo, dhi = paired_bootstrap_diff(y, good, bad, g, n=500)
    assert d > 0 and dlo > 0


def test_tpr_at_fpr():
    y = np.array([0] * 100 + [1] * 10)
    s = np.concatenate([np.linspace(0, 1, 100), np.full(10, 2.0)])
    assert tpr_at_fpr(y, s) == 1.0
