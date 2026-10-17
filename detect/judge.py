"""Prompted Qwen3-8B judge (HackTrace App. E), frozen prompt in detect/monitor_prompt.txt.

Score = logit("Yes") - logit("No") at the first answer token, thinking off,
one forward pass per row, no sampling. Runs on a GPU machine. --limit N writes a
dry run to /data/scratch instead of results/.
"""

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch

from extract.pooled import RAW, ROOT, SPLITS, load_model

TEMPLATE = (ROOT / "detect/monitor_prompt.txt").read_text().split("=== USER ===\n", 1)[1]


def first_user_turn(messages):
    return next(m["content"] for m in messages if m["role"] == "user")


@torch.no_grad()
def score(tok, model, yes_id, no_id, row):
    user = TEMPLATE.format(first_user_turn=first_user_turn(row["messages"]),
                           solution=row["final_solution"] or "", test=row["final_test"] or "")
    text = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
    logits = model(ids).logits[0, -1]
    return float(logits[yes_id] - logits[no_id])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    tok, model = load_model()
    yes_id, no_id = tok.encode("Yes", add_special_tokens=False)[0], tok.encode("No", add_special_tokens=False)[0]
    ids = [r["id"] for r in pq.read_table(SPLITS / "anchor_test.parquet").to_pylist()]
    if a.limit:
        ids = ids[: a.limit]
    want = set(ids)
    rows = {r["id"]: r for r in pq.read_table(RAW / "test.parquet", columns=["id", "messages", "final_solution", "final_test"]).to_pylist()
            if r["id"] in want}
    out = {}
    for k, rid in enumerate(ids, 1):
        out[rid] = score(tok, model, yes_id, no_id, rows[rid])
        if k % 200 == 0:
            print(f"{k}/{len(ids)}", flush=True)
    path = Path("/data/scratch/judge_dryrun.json") if a.limit else ROOT / "results/qwen3-8b/anchor/judge_scores.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out) + "\n")
    print(f"{len(out)} rows -> {path}")


if __name__ == "__main__":
    main()
