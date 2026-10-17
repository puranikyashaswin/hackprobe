"""Linear probes on pooled activations.

For each layer: an L2 logistic regression (torch, LBFGS) and a difference-of-
means direction. Layer and regularisation strength are chosen by 5-fold CV on
the training rows only, grouped by problem, scoring within-problem AUROC.
The chosen probe is refit on all training rows and scored once on test.
"""

import numpy as np
import torch

from detect.metrics import within_problem_auroc

CS = (0.01, 0.1, 1.0, 10.0)


def standardize(xtr, xte):
    mu, sd = xtr.mean(0, keepdims=True), xtr.std(0, keepdims=True) + 1e-6
    return (xtr - mu) / sd, (xte - mu) / sd


def fit_logreg(x, y, c, w=None, iters=200, device="cpu"):
    """L2 logistic regression; returns (weight, bias). c = inverse regularisation strength."""
    X = torch.as_tensor(x, dtype=torch.float32, device=device)
    Y = torch.as_tensor(y, dtype=torch.float32, device=device)
    W = torch.ones_like(Y) if w is None else torch.as_tensor(w, dtype=torch.float32, device=device)
    W = W / W.mean()
    theta = torch.zeros(X.shape[1] + 1, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([theta], max_iter=iters, line_search_fn="strong_wolfe")

    def loss_fn():
        opt.zero_grad()
        z = X @ theta[:-1] + theta[-1]
        loss = (W * torch.nn.functional.binary_cross_entropy_with_logits(z, Y, reduction="none")).mean() \
            + (theta[:-1] ** 2).sum() / (2 * c * len(Y))
        loss.backward()
        return loss

    opt.step(loss_fn)
    t = theta.detach().cpu().numpy()
    return t[:-1], float(t[-1])


def diff_of_means(x, y):
    d = x[y == 1].mean(0) - x[y == 0].mean(0)
    return d / (np.linalg.norm(d) + 1e-9)


def select(acts, y, groups, folds, weights=None, layers=None, device="cpu"):
    """Grid over (layer, C) by grouped CV on training rows. acts: [n, L, d]."""
    layers = range(acts.shape[1]) if layers is None else layers
    best = (-1.0, None, None)
    table = {}
    for layer in layers:
        x = acts[:, layer].astype(np.float32)
        for c in CS:
            scores = np.zeros(len(y))
            for f in np.unique(folds):
                tr, va = folds != f, folds == f
                xtr, xva = standardize(x[tr], x[va])
                w, b = fit_logreg(xtr, y[tr], c, None if weights is None else weights[tr], device=device)
                scores[va] = xva @ w + b
            a = within_problem_auroc(y, scores, groups)
            table[(layer, c)] = a
            if a > best[0]:
                best = (a, layer, c)
    return {"cv_auroc": best[0], "layer": best[1], "C": best[2], "grid": table}


def fit_and_score(acts_tr, y_tr, acts_te, layer, c, weights=None, device="cpu"):
    xtr, xte = standardize(acts_tr[:, layer].astype(np.float32), acts_te[:, layer].astype(np.float32))
    w, b = fit_logreg(xtr, y_tr, c, weights, device=device)
    dom = diff_of_means(xtr, y_tr)
    return {"probe": xte @ w + b, "diff_of_means": xte @ dom}
