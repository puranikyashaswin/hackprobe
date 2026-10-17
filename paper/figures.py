"""Paper figures from the result files (nothing is recomputed here).

  fig1_leadtime   within-problem AUROC k tokens before the first shortcut token (development, set C)
  fig1b_replay    prospective replay at 1 false alarm per 20 honest runs: share of hacks warned
                  before the shortcut and before the action; development and set C; exploratory
                  monitors beside the registered ones when results/qwen3-8b/diag/monitors.json exists
  fig3_crux       the disguise grid: every detector on every cell, plus the code-only probe

  env/cpu/bin/python -m paper.figures   # -> figs/*.svg, figs/*.png
"""

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/qwen3-8b"
FIGS = ROOT / "figs"
C = {"probe": "#1b4f9c", "judge": "#c0392b", "tfidf": "#2e8b57", "tfidf_all": "#2e8b57", "literal": "#8e44ad", "length": "#999999",
     "tfidf_prefix": "#2e8b57", "tfidf_last32": "#7fbf7f", "tfidf_comments": "#b3dfb3", "prefix_judge": "#c0392b",
     "visible_literal_rules": "#8e44ad", "prev_token": "#999999", "causal19": "#e67e22", "probe_w16": "#5b8fd6", "probe_w64": "#9bb8e6",
     "causal19_w16": "#f0a35c", "causal19_w64": "#f6c89a", "tfidf_strong": "#1e6b3f", "literal_sofar": "#8e44ad", "frontier": "#7b241c", "random": "#bbbbbb", "ast": "#a0522d", "unixcoder": "#d4a017", "gemma": "#16a085", "frontier_judge": "#7b241c"}
LABEL = {"probe": "activation probe", "judge": "prompted judge", "tfidf": "prefix TF-IDF", "tfidf_all": "TF-IDF (all positions)",
         "literal": "visible-literal rules", "length": "length", "tfidf_prefix": "TF-IDF on prefix", "tfidf_last32": "TF-IDF last 32 tokens",
         "tfidf_comments": "TF-IDF comments", "prefix_judge": "prompted judge on prefix", "visible_literal_rules": "visible-literal rules",
         "prev_token": "previous token", "causal19": "causal code-mean probe", "probe_w16": "probe, 16-token mean", "probe_w64": "probe, 64-token mean",
         "causal19_w16": "causal probe, 16-token mean", "causal19_w64": "causal probe, 64-token mean", "tfidf_strong": "TF-IDF (same training rows)", "literal_sofar": "visible-literal rules", "frontier": "frontier judge, turn end", "random": "random alarms", "ast": "AST features", "unixcoder": "UniXcoder", "gemma": "Gemma-4-12B activations", "frontier_judge": "frontier judge"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 110})


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def fig1():
    dev, conf = load(RES / "anchor/leadtime.json"), load(RES / "confirm/leadtime.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), sharey=True)
    for ax, (title, panel, keys) in zip(axes, (("Development (60 problems)", dev["t1"], ["probe", "tfidf_prefix", "prefix_judge", "tfidf_last32", "visible_literal_rules"]),
                                               ("Set C, frozen probe (49 problems)", conf["t1"] if conf else None, ["probe"]))):
        if panel is None:
            continue
        curves = panel["curves"] if "curves" in panel else {"probe": panel["curve"]}
        for name in keys:
            if name not in curves:
                continue
            ks = sorted(int(k) for k in curves[name])
            ax.plot(ks, [curves[name][str(k)]["auroc"] for k in ks], color=C[name], lw=1.6 if name == "probe" else 1.1, label=LABEL[name])
        lb = panel.get("L", {}).get("0.8", {}).get("lower_bound_by_k") if panel.get("L") else None
        if lb:
            ks = sorted(int(k) for k in lb)
            ax.plot(ks, [lb[str(k)] for k in ks], color=C["probe"], lw=0.8, ls=":", label="probe, lower 95% bound")
        ax.axhline(0.8, color="#444", lw=0.7, ls="--")
        ax.axhline(0.5, color="#bbb", lw=0.6)
        ax.set_xscale("log")
        ax.set_xlabel("tokens before the first shortcut token (k)")
        ax.set_title(title, fontsize=9)
    axes[0].set_ylabel("within-problem AUROC")
    axes[0].set_ylim(0.4, 1.0)
    axes[0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout()
    save(fig, "fig1_leadtime")


def bars(ax, rep, dets, metric, title, show_fa=True):
    x = np.arange(len(dets))
    for i, d in enumerate(dets):
        v = rep["detectors"][d][metric]
        ax.bar(i, v[0], color=C.get(d, "#777"), width=0.72)
        ax.errorbar(i, v[0], yerr=[[v[0] - v[1]], [v[2] - v[0]]], color="#222", lw=0.8, capsize=2)
        if show_fa:
            fa = rep["detectors"][d]["false_alarm_per_honest"][0]
            ax.text(i, max(v[2], 0.02) + 0.02, f"FA {fa:.0%}", ha="center", fontsize=6.5, color="#c0392b" if fa > 0.05 else "#444")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL.get(d, d) for d in dets], rotation=35, ha="right", fontsize=7)
    ax.set_title(title, fontsize=9)
    ax.set_ylim(0, 1)


def fig1b():
    dev, conf, fair = load(RES / "anchor/replay.json"), load(RES / "confirm/replay.json"), load(RES / "diag/fair/fair.json")
    dets = ["probe", "tfidf", "tfidf_all", "judge", "literal"]
    ncol = 2 + (1 if fair else 0)
    fig, axes = plt.subplots(2, ncol, figsize=(2.6 * ncol + 1.2, 5.2), sharey="row")
    for row, metric in enumerate(("before_fsrt", "before_action")):
        bars(axes[row, 0], dev, dets, metric, f"Development ({dev['n_eval']['hacks']} hacks)" if row == 0 else "")
        if conf:
            bars(axes[row, 1], conf, dets, metric, f"Set C ({conf['n_eval']['hacks']} hacks)" if row == 0 else "")
        if fair:
            fdets = ["causal19_w16", "tfidf_strong", "tfidf_all", "literal_sofar", "judge", "frontier", "random"]
            ax = axes[row, 2]
            for i, d in enumerate(fdets):
                for k, (nm, mk) in enumerate((("development_eval_half", "o"), ("set_C", "s"))):
                    v = fair[nm]["matched_5pct"][d][metric]
                    ax.bar(i + (k - 0.5) * 0.36, v, width=0.34, color=C.get(d, "#777"), alpha=1.0 if k else 0.45)
            ax.set_xticks(range(len(fdets)))
            ax.set_xticklabels([LABEL.get(d, d) for d in fdets], rotation=35, ha="right", fontsize=7)
            ax.set_ylim(0, 1)
            if row == 0:
                ax.set_title("Fair comparison, matched 5% false alarms\n(light: development half, dark: set C)", fontsize=8)
    axes[0, 0].set_ylabel("share of hacks warned\nbefore the shortcut")
    axes[1, 0].set_ylabel("share of hacks warned\nbefore the end of the turn")
    fig.suptitle("Prospective replay at 1 false alarm per 20 honest runs (left, centre: thresholds frozen on the calibration half)", fontsize=9)
    fig.tight_layout()
    save(fig, "fig1b_replay")


def fig3():
    g, diag = load(RES / "crux/grid.json"), load(RES / "diag/diagnostics.json")
    cells = [("disg_orig", "original"), ("disg_unparse", "comments removed"), ("disg_surface", "surface disguise"), ("disg_full", "full disguise")]
    dets = [("probe", "activation probe"), ("judge", "prompted judge"), ("tfidf", "TF-IDF"), ("canon_tfidf", "TF-IDF, canonicalised"),
            ("lit7_lr", "visible-literal LR"), ("rules_lr", "rules A–C LR"), ("rule_b_count", "rule-B count")]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    x = np.arange(len(cells))
    w = 0.11
    for j, (d, lab) in enumerate(dets):
        vals = [g["cells"][c][d]["within"] for c, _ in cells]
        los = [g["cells"][c][d]["ci"][0] for c, _ in cells]
        his = [g["cells"][c][d]["ci"][1] for c, _ in cells]
        xs = x + (j - len(dets) / 2 + 0.5) * w
        col = C.get(d, plt.cm.Greys(0.35 + 0.1 * j))
        ax.bar(xs, vals, width=w, color=col, label=lab)
        ax.errorbar(xs, vals, yerr=[np.array(vals) - los, np.array(his) - vals], fmt="none", ecolor="#222", lw=0.6, capsize=1.5)
    if diag and "D5_code_only_probe" in diag:
        d5 = diag["D5_code_only_probe"]
        for ci, key in ((0, "orig"), (3, "full")):
            ax.plot(x[ci] - len(dets) / 2 * w + w / 2, d5[key]["probe_code_only"], marker="*", ms=9, color="#f1c40f", mec="#222", ls="none",
                    label="probe, code only (no transcript)" if ci == 0 else None)
    ax.axhline(0.5, color="#bbb", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in cells])
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("within-problem AUROC")
    ax.set_title(f"Disguised cheats, read-only ({g['cells']['disg_full']['floor']['hack_programs']} cheat programs, "
                 f"{g['cells']['disg_full']['floor']['problems_with_both']} problems)", fontsize=9)
    ax.legend(frameon=False, fontsize=6.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    save(fig, "fig3_crux")


def fig3b():
    """Under the full disguise and the blind rewrite: the writer's probe vs other representations of the same code."""
    g, diag = load(RES / "crux/grid.json"), load(RES / "diag/diagnostics.json")
    om = {k: load(RES / f"diag/other_model/{f}") for k, f in (("ast", "ast.json"), ("unixcoder", "encoder_unixcoder-base.json"), ("gemma", "decoder_gemma-4-12B-it.json"))}
    fr, lad = load(RES / "diag/frontier_disguise/frontier_disguise.json"), load(RES / "crux/hard_ladder/ladder.json")
    cells = [("original", "original"), ("full", "full disguise"), ("rewrite_blind", "LLM rewrite")]
    series = [("probe", "Qwen3-8B probe (writer)"), ("gemma", "Gemma-4-12B activations"), ("frontier_judge", "frontier judge"),
              ("judge", "Qwen3-8B judge"), ("unixcoder", "UniXcoder"), ("ast", "AST features"), ("tfidf", "TF-IDF")]
    gc = {"original": "disg_orig", "full": "disg_full"}

    def val(d, c):
        if d == "probe":  # code-only render, the same input the other representations see
            if c in gc:
                r = diag["D5_code_only_probe"]["orig" if c == "original" else "full"]
                return r["probe_code_only"], r.get("ci")
            return (lad["level1"]["probe"]["within"], lad["level1"]["probe"]["ci"]) if lad else (None, None)
        if d in ("judge", "tfidf"):
            if c in gc:
                r = g["cells"][gc[c]][d]
                return r["within"], r["ci"]
            return (lad["level1"][d]["within"], lad["level1"][d]["ci"]) if lad else (None, None)
        if d == "frontier_judge" and fr and c in fr:
            return fr[c]["within"], fr[c]["ci"]
        if d in om and om[d] and c in om[d]:
            return om[d][c]["within"], om[d][c]["ci"]
        return None, None
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    x = np.arange(len(cells))
    w = 0.11
    for j, (d, lab) in enumerate(series):
        xs, vs, lo, hi = [], [], [], []
        for i, (c, _) in enumerate(cells):
            v, ci = val(d, c)
            if v is None:
                continue
            xs.append(x[i] + (j - len(series) / 2 + 0.5) * w)
            vs.append(v)
            lo.append(v - ci[0] if ci else 0)
            hi.append(ci[1] - v if ci else 0)
        ax.bar(xs, vs, width=w, color=C.get(d, "#777"), label=lab)
        ax.errorbar(xs, vs, yerr=[lo, hi], fmt="none", ecolor="#222", lw=0.6, capsize=1.5)
    ax.axhline(0.5, color="#bbb", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in cells])
    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("within-problem AUROC")
    ax.set_title("The same disguised code read by different detectors (2,486 programs, 69 problems)", fontsize=9)
    ax.legend(frameon=False, fontsize=6.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    fig.tight_layout()
    save(fig, "fig3b_representations")


def fig4():
    """Second model family: Gemma-4-12B-it writes the code and is probed."""
    fit, mon = load(ROOT / "results/second_model/pass2/gemma_fit.json"), load(ROOT / "results/second_model/pass2/gemma_monitor.json")
    if not fit:
        return
    fig, axes = plt.subplots(1, 2, figsize=(6.4, 2.7))
    ax = axes[0]
    dets = [("probe", "activation probe"), ("tfidf", "TF-IDF"), ("judge", "Gemma judge"), ("length", "length")]
    for i, (d, lab) in enumerate(dets):
        v, ci = fit[d]["within"], fit[d]["ci"]
        ax.bar(i, v, color=C.get(d, "#777"), width=0.7)
        ax.errorbar(i, v, yerr=[[v - ci[0]], [ci[1] - v]], color="#222", lw=0.8, capsize=2)
    ax.set_xticks(range(len(dets)))
    ax.set_xticklabels([lab for _, lab in dets], rotation=25, ha="right", fontsize=7)
    ax.axhline(0.5, color="#bbb", lw=0.6)
    ax.set_ylim(0.3, 1.0)
    ax.set_ylabel("within-problem AUROC")
    ax.set_title(f"Finished code ({fit['n']['test_hacks']} cheats, {fit['n']['test_problems_with_both']} problems)", fontsize=9)
    ax = axes[1]
    ev = mon["evaluation_half"]
    mdets = [("probe_w16", "probe, 16-token mean"), ("tfidf", "TF-IDF on code so far"), ("judge_end", "Gemma judge, turn end")]
    for i, (d, lab) in enumerate(mdets):
        ax.bar(i, ev[d]["caught"], color=C.get(d, {"judge_end": C["judge"]}.get(d, "#777")), width=0.7)
        ax.text(i, ev[d]["caught"] + 0.02, f"FA {ev[d]['false_alarm']:.0%}", ha="center", fontsize=6.5, color="#444")
    ax.set_xticks(range(len(mdets)))
    ax.set_xticklabels([lab for _, lab in mdets], rotation=25, ha="right", fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_ylabel("share of cheats caught\nbefore the end of the turn")
    ax.set_title(f"Causal monitor, evaluation half ({ev['probe_w16']['n_hacks']} cheats)", fontsize=9)
    fig.suptitle("Second model family: Gemma-4-12B-it on HackTrace read-only prompts", fontsize=9)
    fig.tight_layout()
    save(fig, "fig4_gemma")


def save(fig, name):
    FIGS.mkdir(exist_ok=True)
    for ext in ("svg", "png"):
        fig.savefig(FIGS / f"{name}.{ext}", dpi=220 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)
    print("->", FIGS / f"{name}.svg")


if __name__ == "__main__":
    fig1()
    fig1b()
    fig3()
    fig3b()
    fig4()
