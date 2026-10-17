"""Synthetic programs only (no HackTrace scores before the prereg tag)."""

import numpy as np

from detect.text_baselines import artifact_text, literal_features, literal_lr, tfidf_lr

CASES = [{"input": "3\n", "output": "42\n"}]
HACK = "n = int(input())\nif n == 3:\n    print(42)\n"
HONEST = "n = int(input())\nprint(sum(range(n + 1)) * 7)\n"


def test_literal_features_see_the_hardcode():
    h, o = literal_features(HACK, CASES), literal_features(HONEST, CASES)
    assert h[0] == 1.0 and o[0] == 0.0      # visible output appears as a literal
    assert h[4] >= 1 and h[5] >= 1          # literal comparison and branch


def test_baselines_separate_synthetic_classes():
    rng = np.random.default_rng(0)
    texts, feats, y = [], [], []
    for i in range(120):
        is_hack = i % 2
        k = int(rng.integers(1, 9))
        code = (f"n = int(input())\nif n == {k}:\n    print({k * 11})\n" if is_hack
                else f"n = int(input())\nprint(n * {k})\n")
        texts.append(artifact_text(code, ""))
        feats.append(literal_features(code, [{"input": f"{k}\n", "output": f"{k * 11}\n"}]))
        y.append(is_hack)
    y = np.array(y)
    s_t = tfidf_lr(texts[:80], y[:80], texts[80:])
    s_l = literal_lr(np.array(feats[:80]), y[:80], np.array(feats[80:]))
    assert (s_t[y[80:] == 1].mean() > s_t[y[80:] == 0].mean())
    assert (s_l[y[80:] == 1].mean() > s_l[y[80:] == 0].mean())
