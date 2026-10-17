"""Matched prefix judge (frozen prompt detect/monitor_prompt_prefix.txt), every 16 tokens.

Asks Qwen3-8B (thinking off) "Will this trajectory take a shortcut ...?" on the
transcript cut at a scored position; score = logit(Yes) - logit(No).
Speed without changing the answer: for each row the prompts at successive cut
points share a long prefix, so the KV cache of the longest common token prefix
is reused and only the rest is recomputed (exact: same tokens, same model).

  python -m detect.prefix_judge grid    # S3.6: points 16, 32, ... in turn t*, per-token test rows
  python -m detect.prefix_judge replay  # S3.8: every 16 assistant tokens over every turn
"""

import argparse
import json
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
from transformers import DynamicCache

from common.render import n_turns, render_turn
from extract.pooled import RAW, ROOT, SPLITS, load_model

TEMPLATE = (ROOT / "detect/monitor_prompt_prefix.txt").read_text().split("=== USER ===\n", 1)[1]
STEP = 16


class Scorer:
    def __init__(self, tok, model):
        self.tok, self.model = tok, model
        self.yes = tok.encode("Yes", add_special_tokens=False)[0]
        self.no = tok.encode("No", add_special_tokens=False)[0]
        self.reset()

    def reset(self):
        self.ids, self.cache = [], DynamicCache()

    @torch.no_grad()
    def score(self, ids):
        lcp = 0
        for a, b in zip(self.ids, ids):
            if a != b:
                break
            lcp += 1
        lcp = min(lcp, len(ids) - 1)
        self.cache.crop(lcp)
        out = self.model(torch.tensor([ids[lcp:]], device="cuda"), past_key_values=self.cache, use_cache=True, logits_to_keep=1)
        self.cache, self.ids = out.past_key_values, list(ids)
        lg = out.logits[0, -1]
        return float(lg[self.yes] - lg[self.no])


def transcript(messages, turn, cut_chars):
    """The trajectory so far: assistant turns and feedback, cut `cut_chars` characters into turn `turn`.
    Shared with the prefix-trained TF-IDF of detect/replay.py, so both read the same text."""
    parts = []
    for t in range(1, turn + 1):
        a = messages[2 * t]["content"]
        if t == turn:
            a = a[:cut_chars]
        parts.append(f"[assistant turn {t}]\n{a}")
        if t < turn:
            parts.append(f"[feedback]\n{messages[2 * t + 1]['content']}")
    return "\n\n".join(parts)


def prompt_ids(tok, messages, turn, cut_chars):
    """Prompt for the transcript up to `cut_chars` characters into assistant turn `turn`."""
    user1 = next(m["content"] for m in messages if m["role"] == "user")
    user = TEMPLATE.format(first_user_turn=user1, transcript_so_far=transcript(messages, turn, cut_chars))
    text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    return tok(text, add_special_tokens=False)["input_ids"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["grid", "replay"])
    ap.add_argument("--part", type=int, default=0)
    ap.add_argument("--nparts", type=int, default=1)
    ap.add_argument("--set", default="dev", choices=["dev", "C"])
    a = ap.parse_args()
    rows = pq.read_table(SPLITS / ("anchor_pertoken_C.parquet" if a.set == "C" else "anchor_pertoken.parquet")).to_pylist()
    top = {}  # over all rows, before the part split
    for r in rows:
        if r["label"] == 1:
            top[(r["problem_name"], r["t_star"])] = max(top.get((r["problem_name"], r["t_star"]), 0), r["p_star"])
    if a.mode == "replay":  # one pass per trajectory
        rows = list({r["id"]: r for r in rows}.values())
    rows = [r for k, r in enumerate(rows) if k % a.nparts == a.part]
    want = {r["id"] for r in rows}
    msgs = {r["id"]: r["messages"] for r in pq.read_table(RAW / ("train.parquet" if a.set == "C" else "test.parquet"), columns=["id", "messages"]).to_pylist()
            if r["id"] in want}
    tok, model = load_model()
    sc = Scorer(tok, model)
    out, t0, n_scored = {}, time.time(), 0
    for k, r in enumerate(rows, 1):
        m = msgs[r["id"]]
        sc.reset()
        res = {}
        turns = [r["t_star"]] if a.mode == "grid" else list(range(1, n_turns(m) + 1))
        for t in turns:
            rr = render_turn(tok, m, t)
            s0 = rr.assistant_start
            n_ass = len(rr.ids) - s0
            limit = n_ass if a.mode == "replay" else (r["p_star"] if r["label"] == 1 else top.get((r["problem_name"], t), 0) + 32)
            for g in range(STEP, min(limit, n_ass) + 1, STEP):
                cut = rr.offsets[s0 + g - 1][1] - rr.prompt_chars  # characters through token g-1
                res[f"{t}:{g}"] = sc.score(prompt_ids(tok, m, t, cut))  # cut g: tokens 0..g-1 of turn t
                n_scored += 1
        out.setdefault(r["id"], {}).update(res)
        if k % 50 == 0:
            print(f"{k}/{len(rows)}", flush=True)
    out["_meta"] = {"seconds": time.time() - t0, "scores": n_scored, "gpu": torch.cuda.get_device_name()}
    name = "prefix_judge.json" if a.mode == "grid" else "replay_judge.json"
    outd = ROOT / ("results/qwen3-8b/confirm" if a.set == "C" else "results/qwen3-8b/anchor")
    outd.mkdir(parents=True, exist_ok=True)
    path = outd / (name if a.nparts == 1 else name.replace(".json", f".part{a.part}.json"))
    path.write_text(json.dumps(out) + "\n")
    print(f"{len(out)} rows -> {path}")


if __name__ == "__main__":
    main()
