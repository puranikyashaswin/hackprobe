"""Synthetic data only: the signal lives in layer 2, so selection must find it."""

import numpy as np

from detect.metrics import within_problem_auroc
from detect.probe import fit_and_score, select


def make(n=400, layers=4, d=16, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    g = np.repeat(np.arange(n // 10), 10)
    acts = rng.normal(size=(n, layers, d)).astype(np.float32)
    acts[:, 2, 0] += 2.0 * y  # the only informative feature
    return acts, y, g


def test_selection_finds_the_signal_layer():
    acts, y, g = make()
    folds = g % 5
    sel = select(acts, y, g, folds)
    assert sel["layer"] == 2 and sel["cv_auroc"] > 0.8


def test_fit_and_score_on_held_out():
    acts, y, g = make(seed=1)
    te_acts, te_y, te_g = make(seed=2)
    out = fit_and_score(acts, y, te_acts, layer=2, c=1.0)
    assert within_problem_auroc(te_y, out["probe"], te_g) > 0.8
    assert within_problem_auroc(te_y, out["diff_of_means"], te_g) > 0.8
